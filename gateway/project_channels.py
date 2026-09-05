"""Mirror Hermes projects as Discord channels.

Each project becomes one text channel (``#proj-<slug>``) under a dedicated
category; each Hermes session inside it becomes a thread (Discord's own
auto-thread behaviour already provides that half — see
``discord.auto_thread``). This module owns only the *project → channel*
direction: provisioning the channel, remembering the binding on the project
row, and resolving a channel back to the project whose working directory the
agent should operate in.

Design constraints this respects:

- **Config-gated, default off.** Nothing here runs unless
  ``discord.project_channels.enabled`` is true. A bug must not be able to
  spray channels into a server that never asked for the feature.
- **Create-only.** We create channels and categories; we never delete, never
  rename, and never touch a channel we did not create. MANAGE_CHANNELS is a
  broad permission and this module deliberately uses the smallest slice of it.
- **Idempotent.** Provisioning an already-bound project is a no-op, and a
  pre-existing channel with the expected name is adopted rather than
  duplicated. Safe to run repeatedly (startup backfill, CLI, retries).
- **Never fatal.** Discord being down, a revoked permission, or a rate limit
  must degrade to "no channel binding" — never break project creation or
  message handling. Every public entry point swallows and logs.

The REST calls go through ``httpx`` directly rather than discord.py because
provisioning runs from the CLI too, where no gateway client is live.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"

# Last ``retry_after`` Discord reported per endpoint path, so post_message can
# wait the bucket's real cooldown instead of a guess before retrying a 429.
_LAST_RETRY_AFTER: Dict[str, float] = {}

# Discord rejects channel names with uppercase/spaces/most punctuation and caps
# them at 100 chars. Slugs from projects_db are already lowercase-hyphenated,
# but a hand-edited DB row could carry anything, so normalise defensively.
_CHANNEL_SAFE_RE = re.compile(r"[^a-z0-9-]+")

DEFAULT_CHANNEL_PREFIX = "proj-"
DEFAULT_CATEGORY_NAME = "Projects"

# Where a project created FROM Discord gets its folder. Discord has no way to
# pick a filesystem path, so new projects land in one configured root as
# ``<root>/<Project Name>``. Overridable via
# ``discord.project_channels.projects_root``.
DEFAULT_PROJECTS_ROOT = os.path.join(os.path.expanduser("~"), "Hermes Projects")

# Discord channel type constants (numeric in the REST API).
_CHANNEL_TYPE_TEXT = 0
_CHANNEL_TYPE_CATEGORY = 4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _discord_config(config: Optional[dict] = None) -> dict:
    """Return the ``discord:`` block from config.yaml (never None)."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception as exc:  # pragma: no cover - config load is best-effort
            logger.debug("project_channels: config load failed: %s", exc)
            return {}
    block = config.get("discord") if isinstance(config, dict) else None
    return block if isinstance(block, dict) else {}


def settings(config: Optional[dict] = None) -> dict:
    """Resolve ``discord.project_channels`` settings with defaults applied."""
    raw = _discord_config(config).get("project_channels")
    raw = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "guild_id": str(raw.get("guild_id") or "").strip(),
        "category_name": str(
            raw.get("category_name") or DEFAULT_CATEGORY_NAME
        ).strip(),
        "channel_prefix": str(
            raw.get("channel_prefix")
            if raw.get("channel_prefix") is not None
            else DEFAULT_CHANNEL_PREFIX
        ).strip(),
        # When true, a message in a project channel makes the agent operate in
        # that project's primary folder instead of the global TERMINAL_CWD.
        "bind_cwd": bool(raw.get("bind_cwd", True)),
        # Root folder for projects created from Discord (see create_project_from_discord).
        "projects_root": str(
            raw.get("projects_root") or DEFAULT_PROJECTS_ROOT
        ).strip(),
        # Copy new turns from mirrored (desktop/TUI/CLI) sessions into their
        # Discord thread so both surfaces show the same conversation.
        "relay_messages": bool(raw.get("relay_messages", True)),
        # Seconds between mirror/relay passes. An idle pass is local-only
        # (SQLite + JSON, no HTTP), so this can be low without hitting
        # Discord's rate limits. Floored at 5s to keep a typo from spinning.
        "poll_interval": max(5, int(raw.get("poll_interval") or 15)),
        # How many recent user/assistant turns to post into a thread the first
        # time a session is mirrored, so the thread shows the actual
        # conversation instead of only a header. 0 disables backfill (only
        # NEW turns relay). Capped at 50: each message is one REST call, and
        # a 466-message session would otherwise flood the channel and the
        # rate limiter.
        "backfill_limit": max(0, min(50, int(raw.get("backfill_limit", 20)))),
    }


def is_enabled(config: Optional[dict] = None) -> bool:
    s = settings(config)
    return bool(s["enabled"] and s["guild_id"])


def channel_name_for(slug: str, config: Optional[dict] = None) -> str:
    """Deterministic Discord channel name for a project slug."""
    prefix = settings(config)["channel_prefix"]
    raw = f"{prefix}{str(slug or '').strip().lower()}"
    name = _CHANNEL_SAFE_RE.sub("-", raw).strip("-")
    return (name[:100] or "proj").strip("-") or "proj"


# ---------------------------------------------------------------------------
# Discord REST
# ---------------------------------------------------------------------------


def _bot_token() -> str:
    """Read the bot token from the environment (loaded from .env at startup)."""
    return (os.environ.get("DISCORD_BOT_TOKEN") or "").strip()


def _headers(token: str) -> Dict[str, str]:
    # Discord blocks the default urllib/httpx UA on some edges; send a real one.
    return {
        "Authorization": f"Bot {token}",
        "User-Agent": "HermesAgent (https://github.com/NousResearch/hermes-agent, 1.0)",
        "Content-Type": "application/json",
    }


def _request(
    method: str, path: str, token: str, payload: Optional[dict] = None
) -> Optional[Any]:
    """Perform one Discord REST call. Returns parsed JSON, or None on failure."""
    import httpx

    try:
        resp = httpx.request(
            method,
            f"{DISCORD_API}{path}",
            headers=_headers(token),
            json=payload,
            timeout=15.0,
        )
    except Exception as exc:
        logger.warning("project_channels: %s %s failed: %s", method, path, exc)
        return None

    if resp.status_code == 403:
        logger.warning(
            "project_channels: Discord denied %s %s (403). The bot is missing "
            "MANAGE_CHANNELS in this guild.",
            method,
            path,
        )
        return None
    if resp.status_code == 429:
        retry_after = ""
        try:
            retry_after = str(resp.json().get("retry_after", ""))
            _LAST_RETRY_AFTER[path] = float(retry_after)
        except Exception:
            pass
        logger.warning(
            "project_channels: rate limited on %s %s (retry_after=%s); "
            "skipping this pass",
            method,
            path,
            retry_after,
        )
        return None
    if resp.status_code >= 400:
        logger.warning(
            "project_channels: %s %s -> HTTP %s: %s",
            method,
            path,
            resp.status_code,
            resp.text[:300],
        )
        return None
    if not resp.content:
        return {}
    try:
        return resp.json()
    except Exception:
        return {}


def list_guild_channels(guild_id: str, token: str) -> List[dict]:
    data = _request("GET", f"/guilds/{guild_id}/channels", token)
    return data if isinstance(data, list) else []


def ensure_category(
    guild_id: str, token: str, name: str, channels: Optional[List[dict]] = None
) -> Optional[str]:
    """Return the id of the category ``name``, creating it if absent."""
    if not name:
        return None
    if channels is None:
        channels = list_guild_channels(guild_id, token)
    for ch in channels:
        if (
            ch.get("type") == _CHANNEL_TYPE_CATEGORY
            and str(ch.get("name", "")).strip().lower() == name.strip().lower()
        ):
            return str(ch.get("id"))
    created = _request(
        "POST",
        f"/guilds/{guild_id}/channels",
        token,
        {"name": name, "type": _CHANNEL_TYPE_CATEGORY},
    )
    if isinstance(created, dict) and created.get("id"):
        logger.info("project_channels: created category '%s'", name)
        return str(created["id"])
    return None


def ensure_channel(
    guild_id: str,
    token: str,
    channel_name: str,
    *,
    category_id: Optional[str] = None,
    topic: str = "",
    channels: Optional[List[dict]] = None,
) -> Optional[str]:
    """Return the id of text channel ``channel_name``, creating it if absent.

    Adopts a pre-existing channel with the same name rather than creating a
    duplicate — re-running provisioning after a manual channel rename or a
    lost DB binding converges instead of spamming.
    """
    if channels is None:
        channels = list_guild_channels(guild_id, token)
    wanted = channel_name.strip().lower()
    for ch in channels:
        if (
            ch.get("type") == _CHANNEL_TYPE_TEXT
            and str(ch.get("name", "")).strip().lower() == wanted
        ):
            logger.info(
                "project_channels: adopting existing channel #%s (%s)",
                channel_name,
                ch.get("id"),
            )
            return str(ch.get("id"))

    payload: Dict[str, Any] = {"name": channel_name, "type": _CHANNEL_TYPE_TEXT}
    if category_id:
        payload["parent_id"] = str(category_id)
    if topic:
        payload["topic"] = topic[:1024]
    created = _request("POST", f"/guilds/{guild_id}/channels", token, payload)
    if isinstance(created, dict) and created.get("id"):
        logger.info(
            "project_channels: created channel #%s (%s)", channel_name, created["id"]
        )
        return str(created["id"])
    return None


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def provision_project(
    project_id: str, *, config: Optional[dict] = None, conn: Any = None
) -> Optional[str]:
    """Ensure a Discord channel exists for ``project_id`` and record the binding.

    Returns the channel id (existing or newly created), or None when the
    feature is disabled, the token/guild is missing, or Discord refused. Never
    raises — project creation must succeed even when Discord is unreachable.
    """
    try:
        s = settings(config)
        if not s["enabled"]:
            return None
        guild_id = s["guild_id"]
        if not guild_id:
            logger.warning(
                "project_channels: enabled but discord.project_channels.guild_id "
                "is unset; skipping provisioning"
            )
            return None
        token = _bot_token()
        if not token:
            logger.warning(
                "project_channels: DISCORD_BOT_TOKEN is not set; skipping "
                "provisioning"
            )
            return None

        from hermes_cli import projects_db as pdb

        owns_conn = conn is None
        conn = conn if conn is not None else pdb.connect()
        try:
            project = pdb.get_project(conn, project_id)
            if project is None:
                logger.warning("project_channels: no such project %s", project_id)
                return None
            if project.discord_channel_id:
                return project.discord_channel_id

            channels = list_guild_channels(guild_id, token)
            category_id = ensure_category(
                guild_id, token, s["category_name"], channels=channels
            )
            topic = f"Hermes project: {project.name}"
            if project.primary_path:
                topic += f" — {project.primary_path}"
            channel_id = ensure_channel(
                guild_id,
                token,
                channel_name_for(project.slug, config),
                category_id=category_id,
                topic=topic,
                channels=channels,
            )
            if not channel_id:
                return None
            pdb.update_project(conn, project_id, discord_channel_id=channel_id)
            return channel_id
        finally:
            if owns_conn:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception as exc:
        logger.warning(
            "project_channels: provisioning failed for %s: %s",
            project_id,
            exc,
            exc_info=True,
        )
        return None


def sync_all_projects(
    *, config: Optional[dict] = None, include_archived: bool = False
) -> List[Tuple[str, Optional[str]]]:
    """Provision channels for every project missing one.

    Returns ``[(slug, channel_id_or_None), ...]``. Used by the startup backfill
    and by ``hermes project sync-discord``.
    """
    results: List[Tuple[str, Optional[str]]] = []
    try:
        if not is_enabled(config):
            return results
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            for project in pdb.list_projects(conn, include_archived=include_archived):
                if project.discord_channel_id:
                    results.append((project.slug, project.discord_channel_id))
                    continue
                cid = provision_project(project.id, config=config, conn=conn)
                results.append((project.slug, cid))
    except Exception as exc:
        logger.warning("project_channels: sync_all_projects failed: %s", exc)
    return results


# ---------------------------------------------------------------------------
# Resolution (channel -> project working directory)
# ---------------------------------------------------------------------------


def create_project_from_discord(
    name: str, *, config: Optional[dict] = None
) -> Dict[str, Any]:
    """Create a project from Discord: make the folder, the row, and the channel.

    Discord can't supply a filesystem path, so the folder is derived as
    ``<projects_root>/<name>``. Returns a result dict the slash handler renders;
    it never raises so a bad name or a Discord failure reports cleanly.

    Ordering matters: the folder is created FIRST, because a project row whose
    primary_path doesn't exist would silently fail cwd binding later.
    """
    result: Dict[str, Any] = {"ok": False, "error": "", "created_folder": False}
    clean = str(name or "").strip()
    if not clean:
        result["error"] = "Project name must not be empty."
        return result
    # Reject path separators / traversal outright — the name becomes a folder.
    if any(ch in clean for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|')) or ".." in clean:
        result["error"] = (
            "Project name can't contain path separators or these characters: "
            r"/ \ : * ? \" < > |"
        )
        return result

    try:
        s = settings(config)
        root = os.path.abspath(os.path.expanduser(s["projects_root"]))
        folder = os.path.join(root, clean)

        # Containment check: the resolved folder must stay under the root even
        # after normalisation, so a crafted name can't escape it.
        if os.path.commonpath([root, os.path.abspath(folder)]) != root:
            result["error"] = "Project name resolves outside the projects root."
            return result

        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            result["created_folder"] = True

        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            existing = pdb.project_for_path(conn, folder)
            if existing is not None:
                result.update(
                    ok=True,
                    already_existed=True,
                    project_id=existing.id,
                    slug=existing.slug,
                    name=existing.name,
                    folder=folder,
                    channel_id=existing.discord_channel_id,
                )
                return result
            pid = pdb.create_project(
                conn, name=clean, folders=[folder], primary_path=folder
            )
            channel_id = provision_project(pid, config=config, conn=conn)
            project = pdb.get_project(conn, pid)

        result.update(
            ok=True,
            already_existed=False,
            project_id=pid,
            slug=project.slug if project else "",
            name=clean,
            folder=folder,
            channel_id=channel_id,
        )
        return result
    except Exception as exc:
        logger.warning(
            "project_channels: create_project_from_discord(%r) failed: %s",
            name, exc, exc_info=True,
        )
        result["error"] = str(exc)
        return result


def cwd_for_channel(
    channel_id: str, parent_id: Optional[str] = None, config: Optional[dict] = None
) -> Optional[str]:
    """Return the project working directory bound to a channel, if any.

    ``parent_id`` lets a *thread* inherit its parent channel's project, which
    is the common case: the channel is the project, each thread inside it is a
    session. Returns None when the feature is off, nothing is bound, or the
    directory no longer exists on disk.
    """
    try:
        s = settings(config)
        if not (s["enabled"] and s["bind_cwd"]):
            return None
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            project = None
            for cid in (channel_id, parent_id):
                if not cid:
                    continue
                project = pdb.project_for_channel(conn, str(cid))
                if project is not None:
                    break
            if project is None:
                return None
            path = project.primary_path
            if path and os.path.isdir(path):
                return path
            if path:
                logger.warning(
                    "project_channels: project '%s' primary path does not exist: %s",
                    project.slug,
                    path,
                )
            return None
    except Exception as exc:
        logger.debug("project_channels: cwd_for_channel failed: %s", exc)
        return None


def channel_for_cwd(cwd: str, config: Optional[dict] = None) -> Optional[str]:
    """Return the Discord channel bound to the project owning ``cwd``.

    The handoff path uses this to deliver a CLI/desktop/TUI session into its
    project's channel instead of the single global home channel.
    """
    try:
        if not is_enabled(config) or not cwd:
            return None
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            project = pdb.project_for_path(conn, str(cwd))
            if project is None:
                return None
            return project.discord_channel_id or None
    except Exception as exc:
        logger.debug("project_channels: channel_for_cwd failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Session -> thread mirroring
# ---------------------------------------------------------------------------

# Sessions whose thread we've already created, keyed by session id. Persisted
# in state.db's session row (thread_id) for real bindings; this in-memory set
# only suppresses duplicate work within one gateway process.
_MIRRORED_SESSIONS: set = set()


def mirror_state_path():
    """Path of the JSON file recording session_id -> mirrored thread id."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "gateway" / "project_session_threads.json"


def _read_mirror_state() -> Dict[str, str]:
    import json

    try:
        path = mirror_state_path()
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_mirror_state(state: Dict[str, str]) -> None:
    import json

    try:
        path = mirror_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        tmp.replace(path)
    except Exception as exc:
        logger.debug("project_channels: could not persist mirror state: %s", exc)


def create_thread_in_channel(
    channel_id: str, name: str, token: Optional[str] = None
) -> Optional[str]:
    """Create a public thread under ``channel_id`` via REST. Returns its id.

    Uses the threads endpoint directly (type 11 = PUBLIC_THREAD) so this works
    from the mirror loop without a live discord.py client object.
    """
    tok = token or _bot_token()
    if not tok:
        return None
    clean = (name or "session").strip()[:100] or "session"
    created = _request(
        "POST",
        f"/channels/{channel_id}/threads",
        tok,
        {"name": clean, "type": 11, "auto_archive_duration": 10080},
    )
    if isinstance(created, dict) and created.get("id"):
        return str(created["id"])
    return None


def post_message(channel_id: str, content: str, token: Optional[str] = None) -> bool:
    """Post a plain message to a channel/thread, retrying a 429 once.

    Discord's per-channel message bucket is ~5 posts / 5s. ``_request`` treats
    a 429 as a soft failure and returns None, so a burst (a first-sight
    backfill) would silently lose messages. Honour the ``retry_after`` the API
    hands back and try once more before giving up.
    """
    tok = token or _bot_token()
    if not tok:
        return False
    payload = {"content": content[:1900]}
    path = f"/channels/{channel_id}/messages"
    res = _request("POST", path, tok, payload)
    if isinstance(res, dict):
        return True
    # Retry once after the bucket's own cooldown (plus a little headroom).
    time.sleep(_LAST_RETRY_AFTER.get(path, 1.0) + 0.25)
    res = _request("POST", path, tok, payload)
    return isinstance(res, dict)


def sessions_needing_threads(session_db: Any, *, limit: int = 500) -> List[dict]:
    """Return session rows that belong to a project but have no mirrored thread.

    A session qualifies when its cwd resolves to a project that has a bound
    Discord channel, and we have not already created a thread for it. Sessions
    that already LIVE in Discord (source='discord') are skipped: they already
    have their own thread from auto-thread, so mirroring them would duplicate.
    """
    from hermes_cli import projects_db as pdb

    out: List[dict] = []
    try:
        rows = session_db.list_sessions_rich(
            source=None, exclude_sources=["tool"], limit=limit
        )
    except Exception as exc:
        logger.debug("project_channels: could not list sessions: %s", exc)
        return out

    mirrored = _read_mirror_state()
    with pdb.connect_closing() as conn:
        for row in rows:
            sid = str(row.get("id") or "")
            if not sid or sid in mirrored or sid in _MIRRORED_SESSIONS:
                continue
            # Discord-native sessions already have a thread of their own.
            if str(row.get("source") or "") == "discord":
                continue
            cwd = row.get("cwd") or row.get("git_repo_root") or ""
            if not cwd:
                continue
            project = pdb.project_for_path(conn, str(cwd))
            if project is None or not project.discord_channel_id:
                continue
            out.append(
                {
                    "session_id": sid,
                    "title": row.get("title") or sid[:8],
                    "source": row.get("source") or "",
                    "channel_id": project.discord_channel_id,
                    "project_name": project.name,
                }
            )
    return out


def mirrored_session_for_thread(thread_id: str) -> Optional[str]:
    """Return the session id a mirrored thread points at, if any.

    Inverse of the mirror state written by :func:`mirror_sessions_to_threads`.
    Lets the gateway bind an inbound message in a mirrored thread to the
    session that thread represents, so the user can just type instead of
    running ``/resume <id>`` first.
    """
    tid = str(thread_id or "").strip()
    if not tid:
        return None
    try:
        for session_id, mapped in _read_mirror_state().items():
            if str(mapped) == tid:
                return session_id
    except Exception as exc:
        logger.debug("project_channels: mirrored_session_for_thread failed: %s", exc)
    return None


def relay_state_path():
    """Path of the JSON file recording the last relayed message id per session."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "gateway" / "project_session_relay.json"


def _read_relay_state() -> Dict[str, int]:
    import json

    try:
        path = relay_state_path()
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): int(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_relay_state(state: Dict[str, int]) -> None:
    import json

    try:
        path = relay_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        tmp.replace(path)
    except Exception as exc:
        logger.debug("project_channels: could not persist relay state: %s", exc)


# Marker prepended to every relayed message. Any message carrying it is a copy
# we posted, never original content — the echo-loop guard depends on it.
RELAY_MARKER = "\u2937"  # ⤷

# Discord's hard cap is 2000; leave room for the marker and role prefix.
_RELAY_BODY_LIMIT = 1500

# Discord's per-channel message bucket is ~5 / 5s. Pace a backfill under it
# rather than leaning on post_message's 429 retry to claw each one back.
_RELAY_POST_SPACING = 1.1

_RELAY_ROLES = frozenset({"user", "assistant"})


def _relayable(row: dict) -> bool:
    """Whether a message row should be copied to Discord.

    Excludes tool/system turns, blanks, and anything already carrying
    ``RELAY_MARKER`` (i.e. a copy we posted — re-relaying it would loop).
    """
    if str(row.get("role") or "") not in _RELAY_ROLES:
        return False
    content = str(row.get("content") or "").strip()
    return bool(content) and RELAY_MARKER not in content


def _relay_text(row: dict) -> str:
    """Render one message row as the Discord message body."""
    content = str(row.get("content") or "").strip()
    if len(content) > _RELAY_BODY_LIMIT:
        content = content[:_RELAY_BODY_LIMIT] + " …"
    who = "🧑 **You**" if row.get("role") == "user" else "🤖 **Hermes**"
    return f"{RELAY_MARKER} {who}\n{content}"


def relay_new_messages(
    session_db: Any, *, config: Optional[dict] = None, max_per_pass: int = 8
) -> int:
    """Copy new turns from mirrored sessions into their Discord thread.

    Makes a session started on desktop/TUI/CLI show up in Discord as it
    happens, so both surfaces read the same conversation.

    ``max_per_pass`` throttles the STEADY-STATE relay so a burst of new turns
    is spread across passes rather than fired at Discord at once. A first-sight
    backfill is deliberately exempt: its own ``backfill_limit`` already bounds
    it, and truncating it here would advance the high-water mark past messages
    that were never posted, losing them permanently.

    Echo-loop safety, in layers:

    1. Only sessions in the mirror state are relayed, and those are non-Discord
       by construction (``sessions_needing_threads`` skips ``source='discord'``).
       A Discord-native session is therefore never a relay source.
    2. A per-session high-water mark (last relayed message row id) means a
       message is copied at most once, ever.
    3. Relayed text carries ``RELAY_MARKER``; anything already marked is
       skipped, so a copy can never be re-copied.

    Returns the number of messages relayed. Never raises.
    """
    relayed = 0
    try:
        s = settings(config)
        if not (s["enabled"] and s["guild_id"] and s["relay_messages"]):
            return 0
        token = _bot_token()
        if not token:
            return 0

        mirror = _read_mirror_state()
        if not mirror:
            return 0
        state = _read_relay_state()
        dirty = False

        for session_id, thread_id in mirror.items():
            try:
                rows = session_db.get_messages(session_id) or []
            except Exception:
                continue
            if not rows:
                continue

            last_seen = state.get(session_id)
            highest = max(int(r.get("id") or 0) for r in rows)

            if last_seen is None:
                # First sight: post the last N turns so the thread shows the
                # conversation rather than just a header, then mark past
                # everything so the steady-state loop won't duplicate them.
                # Deliberately exempt from max_per_pass — truncating here would
                # advance the mark past messages never posted.
                limit = int(s.get("backfill_limit") or 0)
                if limit > 0:
                    convo = [r for r in rows if _relayable(r)]
                    tail = convo[-limit:]
                    if len(convo) > len(tail):
                        post_message(
                            str(thread_id),
                            f"{RELAY_MARKER} _…{len(convo) - len(tail)} earlier "
                            f"message(s) not shown._",
                            token=token,
                        )
                    for r in tail:
                        if post_message(str(thread_id), _relay_text(r), token=token):
                            relayed += 1
                        time.sleep(_RELAY_POST_SPACING)
                state[session_id] = highest
                dirty = True
                continue

            for row in rows:
                rid = int(row.get("id") or 0)
                if rid <= last_seen:
                    continue
                # Advance the mark for every row we consider, relayable or not:
                # a skipped tool turn must never be re-examined next pass.
                state[session_id] = max(state.get(session_id, 0), rid)
                dirty = True
                if not _relayable(row):
                    continue
                if post_message(str(thread_id), _relay_text(row), token=token):
                    relayed += 1
                if relayed >= max_per_pass:
                    break
            if relayed >= max_per_pass:
                break

        if dirty:
            _write_relay_state(state)
    except Exception as exc:
        logger.warning("project_channels: relay pass failed: %s", exc)
    return relayed


def mirror_sessions_to_threads(
    session_db: Any, *, config: Optional[dict] = None, limit: int = 50
) -> List[Tuple[str, Optional[str]]]:
    """Create a Discord thread for each project-linked session missing one.

    Returns ``[(session_id, thread_id_or_None), ...]``. Idempotent: a session
    is recorded in the mirror state once its thread exists, so repeat runs are
    no-ops. Never raises.
    """
    results: List[Tuple[str, Optional[str]]] = []
    try:
        if not is_enabled(config):
            return results
        token = _bot_token()
        if not token:
            return results

        pending = sessions_needing_threads(session_db)
        if not pending:
            return results

        state = _read_mirror_state()
        for item in pending[:limit]:
            sid = item["session_id"]
            thread_name = f"{item['title']}"[:100]
            tid = create_thread_in_channel(
                item["channel_id"], thread_name, token=token
            )
            if tid:
                state[sid] = tid
                _MIRRORED_SESSIONS.add(sid)
                post_message(
                    tid,
                    f"🔗 **{item['title']}**\n"
                    f"Session `{sid}` · started on `{item['source']}`\n"
                    f"Just type here to continue this conversation.",
                    token=token,
                )
                logger.info(
                    "project_channels: mirrored session %s -> thread %s (%s)",
                    sid, tid, item["project_name"],
                )
            results.append((sid, tid))
        _write_mirror_state(state)
    except Exception as exc:
        logger.warning("project_channels: session mirroring failed: %s", exc)
    return results


__all__ = [
    "channel_for_cwd",
    "channel_name_for",
    "create_project_from_discord",
    "create_thread_in_channel",
    "cwd_for_channel",
    "ensure_category",
    "ensure_channel",
    "is_enabled",
    "list_guild_channels",
    "mirror_sessions_to_threads",
    "mirrored_session_for_thread",
    "post_message",
    "provision_project",
    "relay_new_messages",
    "sessions_needing_threads",
    "settings",
    "sync_all_projects",
]
