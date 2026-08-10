"""Profiles dashboard routes (extracted verbatim from web_server.py).

Two routers because the original registration points are far apart and route
order matters: ``sessions_router`` (/api/profiles/sessions*) was registered
long before the generic ``/api/profiles/{name}`` routes on ``router`` — if the
literal-path routes were appended after ``{name}`` in one router, Starlette
would still match literals first here, but we preserve the original global
registration order exactly rather than rely on that.

Handler bodies are byte-identical; web_server-owned helpers are reached via the
late-binding seam in :mod:`hermes_cli.web_deps` so tests that
``monkeypatch.setattr(web_server, "_helper", ...)`` keep working.
"""

import asyncio  # noqa: F401 — used by handlers
import json
import logging
import re
import subprocess  # noqa: F401
import sys  # noqa: F401
import time  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

from fastapi import APIRouter, HTTPException, Query  # noqa: F401

from hermes_cli.web_deps import late
from hermes_cli.web_models import (
    ProfileCreate,
    ProfileActiveUpdate,
    ProfileExport,
    ProfileImport,
    ProfileRename,
    ProfileSoulUpdate,
    ProfileDescriptionUpdate,
    ProfileModelUpdate,
    ProfileDescribeAuto,
    SessionPrScanBody,
)

# Same logger the handlers used before extraction (identical logger object).
_log = logging.getLogger("hermes_cli.web_server")

# Per-profile session reads report failures in the response's ``errors``
# array, which the desktop sidebar does not currently surface — during the
# stale-schema incident that made an empty sidebar look healthy while
# /api/sessions (which logs) was the only diagnosable trace. Warn once per
# (profile, message) per process so a persistent failure is loud in
# errors.log without turning every sidebar poll into log spam.
_profile_read_warned: set = set()


def _warn_profile_read_error(profile: str, exc: Exception) -> None:
    key = (profile, str(exc))
    if key in _profile_read_warned:
        return
    _profile_read_warned.add(key)
    _log.warning(
        "profile session read failed for %r (reported only in the response "
        "errors array): %s", profile, exc,
    )

sessions_router = APIRouter()
router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_cron_profile_home = late("_cron_profile_home")
_disable_unselected_skills = late("_disable_unselected_skills")
_fallback_profile_dicts = late("_fallback_profile_dicts")
_hub_action_name = late("_hub_action_name")
_open_session_db_at_path = late("_open_session_db_at_path")
_profile_setup_command = late("_profile_setup_command")
_profile_to_dict = late("_profile_to_dict")
_resolve_profile_dir = late("_resolve_profile_dir")
_spawn_hermes_action = late("_spawn_hermes_action")
_strip_session_list_rows = late("_strip_session_list_rows")
_write_profile_mcp_servers = late("_write_profile_mcp_servers")
_write_profile_model = late("_write_profile_model")


@sessions_router.get("/api/profiles/sessions")
def get_profiles_sessions(
    # ``le=500`` caps the per-request page size (idea from #39200) — this
    # endpoint fans the query out across EVERY profile's state.db, so an
    # unbounded limit multiplies the damage. 500 (not 100) because real
    # desktop callers use limit=200 (sessions-settings ARCHIVED_FETCH_LIMIT,
    # command palette) and the electron remote-merge over-fetches
    # ``limit + offset``.
    limit: int = Query(20, ge=0, le=500),
    offset: int = Query(0, ge=0),
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "recent",
    profile: str = "all",
    source: str = None,
    sources: str = None,
    exclude_sources: str = None,
    full: bool = False,
):
    """Unified, read-only session list aggregated across ALL profiles.

    Intentionally process-light: this opens each profile's ``state.db`` directly
    from disk — it does NOT spawn a dashboard backend per profile. Each returned
    session is tagged with its owning ``profile`` so the desktop renders one
    browsable list and only spins up a profile's backend when the user actually
    interacts (sends a message). A user with a single (default) profile gets the
    same rows as ``/api/sessions``, just tagged ``profile="default"``.

    Rows omit ``system_prompt``/``model_config`` unless ``full=1`` — same
    list projection as ``/api/sessions``.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(status_code=400, detail="archived must be one of: exclude, only, include")
    if order not in ("created", "recent"):
        raise HTTPException(status_code=400, detail="order must be one of: created, recent")

    from hermes_cli import profiles as profiles_mod

    targets: List[Tuple[str, Path]] = []
    if profile and profile != "all":
        name, home = _cron_profile_home(profile)
        targets.append((name, home))
    else:
        try:
            infos = profiles_mod.list_profiles()
            targets = [(info.name, info.path) for info in infos]
        except Exception:
            _log.exception("GET /api/profiles/sessions: list_profiles failed")
            targets = []
        if not targets:
            targets.append(("default", profiles_mod.get_profile_dir("default")))

    min_message_count = max(0, min_messages)
    archived_only = archived == "only"
    include_archived = archived == "include"
    # Source scoping (see /api/sessions): recents pass exclude_sources=cron,
    # the cron-jobs section passes source=cron — two independent lists so
    # newest cron sessions can't starve the recents page.
    source_filter = source or None
    source_list = [s.strip() for s in (sources or "").split(",") if s.strip()]
    exclude_list = [s.strip() for s in (exclude_sources or "").split(",") if s.strip()]
    # Over-fetch per profile so the merged+sorted window is correct for the
    # requested page. Capped so a huge profile can't blow up the response.
    per_profile = min(max(limit + offset, limit), 500)

    merged: List[Dict[str, Any]] = []
    total = 0
    profile_totals: Dict[str, int] = {}
    errors: List[Dict[str, str]] = []
    now = time.time()
    for name, home in targets:
        db_path = Path(home) / "state.db"
        if not db_path.exists():
            continue
        try:
            # Read-only on the healthy path: this loop runs on every sidebar
            # refresh, so it must not routinely DDL/write-lock another
            # profile's live DB (see SessionDB read_only docstring). The
            # helper's stale-schema probe performs a ONE-TIME writable open
            # when the store predates a schema addition — the same reconcile
            # that profile's own backend runs at startup — because read-only
            # opens skip column reconciliation and would otherwise fail here
            # on every refresh until something else opened the DB writable.
            db = _open_session_db_at_path(db_path, read_only=True)
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
            continue
        try:
            rows = db.list_sessions_rich(
                source=source_filter,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
                limit=per_profile,
                offset=0,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
                # Same SQL-level blob skip as /api/sessions (see above).
                compact_rows=not full,
                include_pinned=True,
            )
            profile_total = db.session_count(
                source=source_filter,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            total += profile_total
            profile_totals[name] = profile_total
            for s in rows:
                s["profile"] = name
                s["is_default_profile"] = name == "default"
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
                s["archived"] = bool(s.get("archived"))
                s["pinned"] = bool(s.get("pinned"))
                merged.append(s)
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
        finally:
            db.close()

    sort_key = "last_active" if order == "recent" else "started_at"
    merged.sort(key=lambda s: s.get(sort_key) or s.get("started_at") or 0, reverse=True)
    # Pinned rows are back-filled past each profile's LIMIT on purpose; keep
    # them in the merged window instead of re-dropping them on recency.
    window = merged[offset:offset + limit]
    if len(merged) > offset + limit:
        seen = {id(s) for s in window}
        window.extend(s for s in merged[offset + limit:] if s.get("pinned") and id(s) not in seen)
    if not full:
        _strip_session_list_rows(window)
    return {
        "sessions": window,
        "total": total,
        "profile_totals": profile_totals,
        "limit": limit,
        "offset": offset,
        "errors": errors,
    }


@sessions_router.get("/api/profiles/sessions/sidebar")
def get_profiles_sessions_sidebar(
    recents_profile: str = "all",
    recents_limit: int = 20,
    recents_exclude: str = None,
    cron_limit: int = 50,
    messaging_limit: int = 100,
    messaging_exclude: str = None,
):
    """Batched sidebar session slices — one profile-DB open per refresh.

    The desktop sidebar needs three source-scoped windows per refresh: recents
    (local chats), cron sessions, and messaging-platform sessions. Served as
    three separate ``/api/profiles/sessions`` calls they reopened every
    profile's ``state.db`` three times and re-counted each refresh. This opens
    each DB once and runs the three filtered queries together, returning the
    three windows in one payload. Read-only and process-light, same row
    projection and 300s active heuristic as ``/api/profiles/sessions``.

    ``recents_profile`` scopes the whole payload, not just recents. Cron and
    messaging used to come back cross-profile unconditionally, which is what
    made a concrete profile show another profile's Telegram threads and
    cronjobs (#65710, #42651, #70629) — the sidebar has one scope, so every
    slice answers to it, and ``all`` is how the caller asks for everything.

    The caller passes the source taxonomy (``recents_exclude`` /
    ``messaging_exclude`` CSV, ``source=cron`` is implicit) so this stays
    taxonomy-agnostic like the per-slice endpoint. All three slices use
    ``min_messages=1`` / ``archived=exclude`` / recency order, matching the
    desktop's per-slice calls.
    """
    from hermes_cli import profiles as profiles_mod

    try:
        infos = profiles_mod.list_profiles()
        targets: List[Tuple[str, Path]] = [(info.name, info.path) for info in infos]
    except Exception:
        _log.exception("GET /api/profiles/sessions/sidebar: list_profiles failed")
        targets = []
    if not targets:
        targets.append(("default", profiles_mod.get_profile_dir("default")))

    recents_scope = (recents_profile or "all").strip() or "all"
    recents_exclude_list = [s for s in (recents_exclude or "").split(",") if s.strip()]
    messaging_exclude_list = [s for s in (messaging_exclude or "").split(",") if s.strip()]

    recents_cap = min(max(recents_limit, 1), 500)
    cron_cap = min(max(cron_limit, 1), 500)
    messaging_cap = min(max(messaging_limit, 1), 500)

    recents_rows: List[Dict[str, Any]] = []
    cron_rows: List[Dict[str, Any]] = []
    messaging_rows: List[Dict[str, Any]] = []
    recents_truncated: Dict[str, bool] = {}
    profile_totals: Dict[str, Dict[str, float]] = {}
    errors: List[Dict[str, str]] = []
    now = time.time()

    def _tag(rows: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
        for s in rows:
            s["profile"] = name
            s["is_default_profile"] = name == "default"
            s["is_active"] = (
                s.get("ended_at") is None
                and (now - s.get("last_active", s.get("started_at", 0))) < 300
            )
            s["archived"] = bool(s.get("archived"))
            # SQLite stores the pin as 0/1; the sidebar needs a real boolean to
            # render the Pinned section from server state.
            s["pinned"] = bool(s.get("pinned"))
        return rows

    def _slice(db, *, source=None, exclude=None, cap):
        return db.list_sessions_rich(
            source=source,
            exclude_sources=exclude or None,
            limit=cap,
            offset=0,
            min_message_count=1,
            include_archived=False,
            archived_only=False,
            order_by_last_active=True,
            compact_rows=True,
            # A pinned conversation must reach the sidebar even when it has
            # aged past the window — otherwise its Pinned row renders empty.
            include_pinned=True,
        )

    for name, home in targets:
        if recents_scope != "all" and name != recents_scope:
            continue
        db_path = Path(home) / "state.db"
        if not db_path.exists():
            continue
        try:
            # Read-only with the stale-schema heal — same contract as the
            # per-slice endpoint above (one-time writable reconcile when the
            # store predates a schema addition, plain read-only otherwise).
            db = _open_session_db_at_path(db_path, read_only=True)
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
            continue
        try:
            profile_rows = _slice(db, exclude=recents_exclude_list, cap=recents_cap)
            # A full window means more rows remain on disk. That is all the
            # sidebar's "load more" needs, and unlike an exact COUNT(*) per
            # profile per refresh it costs nothing beyond the rows already
            # read. Discount pinned back-fills — they arrive past the LIMIT
            # and would otherwise fake a full page on a short list.
            unpinned_count = sum(1 for s in profile_rows if not s.get("pinned"))
            recents_truncated[name] = unpinned_count >= recents_cap
            recents_rows.extend(_tag(profile_rows, name))
            # Aggregated in SQL rather than over the window above: the window is
            # a page, and a total that shrank when you scrolled would be worse
            # than no total at all.
            profile_totals[name] = db.usage_totals()
            cron_rows.extend(_tag(_slice(db, source="cron", cap=cron_cap), name))
            messaging_rows.extend(
                _tag(_slice(db, exclude=messaging_exclude_list, cap=messaging_cap), name)
            )
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
        finally:
            db.close()

    def _window(rows: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
        rows.sort(key=lambda s: s.get("last_active") or s.get("started_at") or 0, reverse=True)
        # Pinned rows survive the cap. The per-profile queries deliberately
        # back-fill them past the LIMIT, so truncating the merged window on
        # recency alone would throw away exactly what the back-fill fetched.
        win = rows[:cap]
        if len(rows) > cap:
            seen = {id(s) for s in win}
            win.extend(s for s in rows[cap:] if s.get("pinned") and id(s) not in seen)
        _strip_session_list_rows(win)
        return win

    return {
        "recents": {
            "sessions": _window(recents_rows, recents_cap),
            "profiles_truncated": recents_truncated,
            "profiles_usage": profile_totals,
        },
        "cron": {"sessions": _window(cron_rows, cron_cap)},
        "messaging": {
            "sessions": _window(messaging_rows, messaging_cap),
            "total": len(messaging_rows),
        },
        "errors": errors,
    }


def _merge_by_id(into: Dict[str, Dict[str, Any]], entries: List[Dict[str, Any]], child_key: str) -> None:
    """Fold ``entries`` into ``into`` by id, recursing through one child list.

    Repos merge their lanes, lanes merge their sessions. Counts add up and the
    newest activity wins; everything else is first-writer, since the entries
    describe the same path either way.
    """
    for entry in entries:
        existing = into.get(entry["id"])
        if existing is None:
            into[entry["id"]] = entry
            continue
        if child_key == "sessions":
            existing["sessions"].extend(entry.get("sessions") or [])
        else:
            children: Dict[str, Dict[str, Any]] = {c["id"]: c for c in existing.get(child_key) or []}
            _merge_by_id(children, entry.get(child_key) or [], "sessions")
            existing[child_key] = list(children.values())
        if "sessionCount" in existing:
            existing["sessionCount"] = (existing.get("sessionCount") or 0) + (entry.get("sessionCount") or 0)


def _merge_profile_tree(
    merged: Dict[str, Dict[str, Any]],
    projects: List[Dict[str, Any]],
    profile: str,
    preview_limit: int,
) -> None:
    """Fold one profile's projects into the shared tree, keyed by folder.

    The same checkout in two profiles is one group, as is ``__no_project__``,
    which every profile has and which would otherwise put a "Home" on screen per
    profile. Keying on the path rather than the id also folds a profile's
    declared project (``p_<hash>``) together with the auto entry another profile
    grows for the same folder. Sessions carry the owning profile instead, which
    is what the row badge and the profile filter read; a group header never
    claims a single owner.
    """
    for project in projects:
        for lane in (repo for r in project.get("repos") or [] for repo in r.get("groups") or []):
            for session in lane.get("sessions") or []:
                session["profile"] = profile
                session["is_default_profile"] = profile == "default"
        for session in project.get("previewSessions") or []:
            session["profile"] = profile
            session["is_default_profile"] = profile == "default"

        key = project.get("path") or project["id"]
        existing = merged.get(key)
        if existing is None:
            merged[key] = project
            continue

        # A declared project carries the label, color and icon the user chose,
        # so it wins the identity when it meets another profile's auto entry.
        if existing.get("isAuto") and not project.get("isAuto"):
            existing, project = project, existing
            merged[key] = existing

        repos: Dict[str, Dict[str, Any]] = {r["id"]: r for r in existing.get("repos") or []}
        _merge_by_id(repos, project.get("repos") or [], "groups")
        existing["repos"] = list(repos.values())
        existing["sessionCount"] = (existing.get("sessionCount") or 0) + (project.get("sessionCount") or 0)
        existing["totalTokens"] = (existing.get("totalTokens") or 0) + (project.get("totalTokens") or 0)
        existing["totalCostUsd"] = (existing.get("totalCostUsd") or 0) + (project.get("totalCostUsd") or 0)
        existing["lastActive"] = max(existing.get("lastActive") or 0, project.get("lastActive") or 0)
        previews = (existing.get("previewSessions") or []) + (project.get("previewSessions") or [])
        previews.sort(key=lambda s: s.get("last_active") or s.get("started_at") or 0, reverse=True)
        existing["previewSessions"] = previews[:preview_limit]


@sessions_router.get("/api/profiles/projects/tree")
def get_profiles_projects_tree(preview_limit: int = 3, session_limit: int = 2000):
    """Project tree for every profile at once, for the all-profiles sidebar.

    ``projects.tree`` over JSON-RPC answers for the backend's own profile, so
    the grouped sidebar had nothing to draw once the user asked for all of
    them. This runs the same authoritative builder once per profile against
    that profile's ``state.db``, scoping the rest of its inputs — projects.db,
    the repo-scan policy, the HERMES_HOME junk filters — through the
    context-local home override the profile-scoped writers already use.

    Projects merge by id across profiles, so a group stands for a checkout
    rather than a checkout-and-owner, and the profile shows up per row where
    the filter can act on it.

    Discovery is off. A repo with zero sessions is the same repo in every
    profile, so folding the disk scan in would multiply empty lanes by the
    profile count — and it is the one part of the builder that writes
    (policy reconciliation), which this read-only fan-out should not do to a
    profile the user is not driving.
    """
    from hermes_cli import profiles as profiles_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server as gateway_server

    try:
        targets: List[Tuple[str, Path]] = [
            (info.name, info.path) for info in profiles_mod.list_profiles()
        ]
    except Exception:
        _log.exception("GET /api/profiles/projects/tree: list_profiles failed")
        targets = []
    if not targets:
        targets.append(("default", profiles_mod.get_profile_dir("default")))

    merged: Dict[str, Dict[str, Any]] = {}
    scoped_session_ids: List[str] = []
    errors: List[Dict[str, str]] = []

    for name, home in targets:
        db_path = Path(home) / "state.db"
        if not db_path.exists():
            continue
        try:
            db = _open_session_db_at_path(db_path, read_only=True)
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
            continue

        token = set_hermes_home_override(str(home))
        try:
            tree, _active_id = gateway_server._build_project_tree(
                db,
                preview_limit=preview_limit,
                hydrate=False,
                session_limit=session_limit,
                include_discovered=False,
            )
            _merge_profile_tree(merged, tree["projects"], name, preview_limit)
            scoped_session_ids.extend(tree["scoped_session_ids"])
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
        finally:
            reset_hermes_home_override(token)
            db.close()

    projects = sorted(merged.values(), key=lambda p: p.get("lastActive") or 0, reverse=True)
    return {
        "projects": projects,
        # Ownership is per profile, so no single project is "the active one"
        # here; the desktop only reads active_id to bias its overview sort.
        "active_id": None,
        "scoped_session_ids": scoped_session_ids,
        "errors": errors,
    }


# `gh pr create` prints the PR url and nothing else, so a tool result whose
# whole output IS a PR url means this session opened that PR. Anything looser —
# a url inside prose, a `gh pr view` payload, an issue link — is a session
# TALKING about a PR, which is not the same claim.
_PR_URL_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/pull/(\d+)/?$")


def _pr_url_from_tool_output(content: str) -> Optional[Tuple[int, str]]:
    """The (number, url) a tool result announces, or None."""
    try:
        output = (json.loads(content) or {}).get("output")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(output, str):
        return None
    match = _PR_URL_RE.match(output.strip())
    return (int(match.group(1)), match.group(0)) if match else None


@sessions_router.post("/api/profiles/sessions/pull-requests")
def post_profiles_sessions_pull_requests(body: SessionPrScanBody):
    """The PR each of these sessions opened, recovered from its own transcript.

    A session records the branch it started on, so the sidebar can join a row to
    its PR — but a session that starts in the main checkout and does its work in
    a worktree has no branch of its own, and its PR is invisible to that join.
    The evidence is in the conversation: ``gh pr create`` ran, and its output is
    a bare PR url. Scanning for exactly that shape recovers the link with no
    inference (see ``_pr_url_from_tool_output``).

    Read-only across every profile, and the caller is expected to ask once per
    session and remember the answer — a session's transcript does not grow a
    second PR.
    """
    from hermes_cli import profiles as profiles_mod

    wanted = list(dict.fromkeys(s for s in (body.ids or []) if s))[:2000]
    if not wanted:
        return {"pull_requests": {}, "scanned": []}

    try:
        targets = [(info.name, info.path) for info in profiles_mod.list_profiles()]
    except Exception:
        _log.exception("POST /api/profiles/sessions/pull-requests: list_profiles failed")
        targets = []
    if not targets:
        targets.append(("default", profiles_mod.get_profile_dir("default")))

    found: Dict[str, Dict[str, Any]] = {}
    for name, home in targets:
        db_path = Path(home) / "state.db"
        if not db_path.exists():
            continue
        try:
            db = _open_session_db_at_path(db_path, read_only=True)
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            continue
        try:
            for pr in db.find_pr_url_messages(wanted):
                parsed = _pr_url_from_tool_output(pr["content"])
                if parsed:
                    number, url = parsed
                    # Ordered oldest-first, so a later `gh pr create` in the
                    # same conversation wins — a reopened/replacement PR is the
                    # one the session ended on.
                    found[pr["session_id"]] = {"number": number, "url": url}
        except Exception as exc:
            _warn_profile_read_error(name, exc)
        finally:
            db.close()

    # Every id we looked at, so the caller can remember "asked, nothing there"
    # and never scan this session again.
    return {"pull_requests": found, "scanned": wanted}


@router.get("/api/profiles")
async def list_profiles_endpoint():
    from hermes_cli import profiles as profiles_mod
    try:
        loop = asyncio.get_running_loop()
        profiles = await loop.run_in_executor(None, profiles_mod.list_profiles)
        return {"profiles": [_profile_to_dict(p) for p in profiles]}
    except Exception:
        _log.exception("GET /api/profiles failed; falling back to profile directory scan")
        return {"profiles": _fallback_profile_dicts(profiles_mod)}


@router.post("/api/profiles")
async def create_profile_endpoint(body: ProfileCreate):
    from hermes_cli import profiles as profiles_mod
    explicit_source = (body.clone_from or "").strip()
    if explicit_source:
        # Duplicating a specific profile: clone its config/skills/SOUL (or full
        # state when clone_all) from the named source rather than "default".
        clone = True
        clone_from = explicit_source
        clone_config = not body.clone_all
    elif body.clone_all:
        # Preserve the dashboard's historical clone-all behavior: a full-copy
        # request with no explicit dropdown source copies from default.
        clone = True
        clone_from = "default"
        clone_config = False
    else:
        clone = body.clone_from_default
        clone_from = "default" if clone else None
        clone_config = clone
    try:
        path = profiles_mod.create_profile(
            name=body.name,
            clone_from=clone_from,
            clone_all=body.clone_all,
            clone_config=clone_config,
            no_skills=body.no_skills,
            description=body.description,
        )
        # Match the CLI's profile-create flow: fresh named profiles get the
        # bundled skills installed. When cloning from default, create_profile()
        # has already copied the source profile's skills, including any
        # user-installed skills. When no_skills=True, create_profile() wrote
        # the opt-out marker and seed_profile_skills() will no-op.
        if not clone:
            profiles_mod.seed_profile_skills(path, quiet=True)

        # Match the CLI's profile-create flow: named profiles should get a
        # wrapper in ~/.local/bin when the alias is safe to create.
        collision = profiles_mod.check_alias_collision(body.name)
        if not collision:
            profiles_mod.create_wrapper_script(body.name)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Optional explicit model assignment for the new profile. Best-effort:
    # the profile already exists, so a model-write hiccup must not 500 the
    # whole create — the user can set the model later from the Models page
    # or `<profile> setup`.
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    model_set = False
    if provider and model:
        try:
            _write_profile_model(path, provider, model)
            model_set = True
        except Exception:
            _log.exception("Setting model for new profile %s failed", body.name)

    # Optional MCP servers. Best-effort, same rationale as model assignment.
    mcp_written = 0
    if body.mcp_servers:
        try:
            mcp_written = _write_profile_mcp_servers(path, body.mcp_servers)
        except Exception:
            _log.exception("Writing MCP servers for new profile %s failed", body.name)

    # Optional "keep" skill selection — replace semantics. When the builder
    # sends an explicit keep list, disable every seeded skill not in it.
    # Best-effort. Skipped when keep_skills is empty (legacy: keep the bundle).
    skills_disabled = 0
    if body.keep_skills:
        try:
            skills_disabled = _disable_unselected_skills(path, body.keep_skills)
        except Exception:
            _log.exception("Applying skill selection for new profile %s failed", body.name)

    # Optional skills-hub installs. Spawned async, scoped to the new profile
    # via `-p <name>` (a fresh subprocess re-binds skills_hub.SKILLS_DIR to the
    # profile's HERMES_HOME at import). Returns PIDs for the UI to poll.
    hub_installs: List[Dict[str, Any]] = []
    for identifier in body.hub_skills:
        ident = (identifier or "").strip()
        if not ident:
            continue
        try:
            proc = _spawn_hermes_action(
                ["-p", body.name, "skills", "install", ident, "--yes"],
                _hub_action_name("install", ident),
            )
            hub_installs.append({"identifier": ident, "pid": proc.pid})
        except Exception:
            _log.exception(
                "Spawning hub-skill install %s for new profile %s failed",
                ident,
                body.name,
            )
            hub_installs.append({"identifier": ident, "pid": None})

    return {
        "ok": True,
        "name": body.name,
        "path": str(path),
        "model_set": model_set,
        "mcp_written": mcp_written,
        "skills_disabled": skills_disabled,
        "hub_installs": hub_installs,
    }


@router.get("/api/profiles/active")
async def get_active_profile_endpoint():
    """Return the sticky active profile and the profile this dashboard
    process is currently running as.

    ``active`` is the sticky default written by ``hermes profile use`` —
    the profile new CLI invocations pick up. ``current`` is the profile
    the running dashboard/gateway is scoped to (derived from HERMES_HOME).
    """
    from hermes_cli import profiles as profiles_mod
    try:
        active = profiles_mod.get_active_profile() or "default"
    except Exception:
        active = "default"
    try:
        current = profiles_mod.get_active_profile_name() or "default"
    except Exception:
        current = "default"
    return {"active": active, "current": current}


@router.post("/api/profiles/active")
async def set_active_profile_endpoint(body: ProfileActiveUpdate):
    """Set the sticky active profile (mirrors ``hermes profile use``).

    Note: this does not retarget the already-running dashboard process —
    it changes which profile subsequent CLI commands and gateways use.
    """
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.set_active_profile(body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles/active failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "active": profiles_mod.normalize_profile_name(body.name)}


@router.get("/api/profiles/{name}/setup-command")
async def get_profile_setup_command(name: str):
    return {"command": _profile_setup_command(name)}


@router.post("/api/profiles/{name}/open-terminal")
async def open_profile_terminal_endpoint(name: str):
    try:
        command = _profile_setup_command(name)

        if sys.platform.startswith("win"):
            subprocess.Popen(["cmd.exe", "/c", "start", "", command])
        elif sys.platform == "darwin":
            escaped = command.replace("\\", "\\\\").replace('"', '\\"')
            applescript = (
                'tell application "Terminal"\n'
                "activate\n"
                f'do script "{escaped}"\n'
                "end tell"
            )
            subprocess.Popen(["osascript", "-e", applescript])
        else:
            terminal_commands = [
                ("x-terminal-emulator", ["x-terminal-emulator", "-e", "sh", "-lc", command]),
                ("gnome-terminal", ["gnome-terminal", "--", "sh", "-lc", command]),
                ("konsole", ["konsole", "-e", "sh", "-lc", command]),
                ("xfce4-terminal", ["xfce4-terminal", "-e", f"sh -lc '{command}'"]),
                ("mate-terminal", ["mate-terminal", "-e", f"sh -lc '{command}'"]),
                ("lxterminal", ["lxterminal", "-e", f"sh -lc '{command}'"]),
                ("tilix", ["tilix", "-e", "sh", "-lc", command]),
                ("alacritty", ["alacritty", "-e", "sh", "-lc", command]),
                ("kitty", ["kitty", "sh", "-lc", command]),
                ("xterm", ["xterm", "-e", "sh", "-lc", command]),
            ]
            for executable, popen_args in terminal_commands:
                if subprocess.call(
                    ["which", executable],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ) == 0:
                    subprocess.Popen(popen_args)
                    break
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No supported terminal emulator found",
                )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /api/profiles/%s/open-terminal failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "command": command}


@router.patch("/api/profiles/{name}")
async def rename_profile_endpoint(name: str, body: ProfileRename):
    from hermes_cli import profiles as profiles_mod
    try:
        path = profiles_mod.rename_profile(name, body.new_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, FileExistsError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("PATCH /api/profiles/%s failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "name": body.new_name, "path": str(path)}


@router.delete("/api/profiles/{name}")
async def delete_profile_endpoint(name: str):
    """Delete a profile. The dashboard collects the user's confirmation in
    its own dialog before this request, so we always pass ``yes=True`` to
    skip the CLI's interactive prompt."""
    from hermes_cli import profiles as profiles_mod
    try:
        path = profiles_mod.delete_profile(name, yes=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("DELETE /api/profiles/%s failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "path": str(path)}


@router.get("/api/profiles/{name}/soul")
async def get_profile_soul(name: str):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"
    if soul_path.exists():
        try:
            return {"content": soul_path.read_text(encoding="utf-8"), "exists": True}
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not read SOUL.md: {e}")
    return {"content": "", "exists": False}


@router.put("/api/profiles/{name}/soul")
async def update_profile_soul(name: str, body: ProfileSoulUpdate):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"
    try:
        from utils import atomic_write_text

        # PUT replaces the whole persona document from the dashboard editor.
        # A bare write_text() truncates SOUL.md before the new body lands, and
        # the paired GET above reports an unreadable file as
        # ``{"content": "", "exists": False}`` -- so an interrupted save shows
        # up as "your persona was never set" and the editor's next Save
        # persists that empty document over it.
        #
        # preserve_mode carries an existing file's permission bits and owner
        # across the replace. create_mode=0o644 covers the first save: named
        # profiles seed SOUL.md at the umask default (hermes_cli.profiles
        # chmods only .env to 0600), and SOUL.md is not a secret. (The default
        # profile's runtime seeder does run it through _secure_file, but that
        # seeder fires on every load_config, so the file already exists there
        # and preserve_mode keeps whatever mode it set.)
        atomic_write_text(
            soul_path, body.content, preserve_mode=True, create_mode=0o644
        )
    except OSError as e:
        _log.exception("PUT /api/profiles/%s/soul failed", name)
        raise HTTPException(status_code=500, detail=f"Could not write SOUL.md: {e}")
    return {"ok": True}


@router.put("/api/profiles/{name}/description")
async def update_profile_description_endpoint(name: str, body: ProfileDescriptionUpdate):
    """Set or clear a profile's role description (kanban routing signal).

    Empty string clears the description. Non-empty stores it as a
    user-authored description (``description_auto: false``) so the
    auto-describer won't overwrite it on a sweep.
    """
    from hermes_cli import profiles as profiles_mod
    profile_dir = _resolve_profile_dir(name)
    text = (body.description or "").strip()
    try:
        profiles_mod.write_profile_meta(
            profile_dir,
            description=text,
            description_auto=False,
        )
    except Exception as e:
        _log.exception("PUT /api/profiles/%s/description failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "description": text, "description_auto": False}


@router.put("/api/profiles/{name}/model")
async def update_profile_model_endpoint(name: str, body: ProfileModelUpdate):
    """Set the main model (``model.default`` + ``model.provider``) for a
    specific profile's config.yaml, without touching the dashboard's own
    active profile. Mirrors ``POST /api/model/set`` (main scope) but scoped
    to the named profile via the HERMES_HOME override.
    """
    profile_dir = _resolve_profile_dir(name)
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model are required")
    try:
        _write_profile_model(profile_dir, provider, model)
    except Exception as e:
        _log.exception("PUT /api/profiles/%s/model failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "provider": provider, "model": model}


@router.post("/api/profiles/{name}/describe-auto")
async def describe_profile_auto_endpoint(name: str, body: ProfileDescribeAuto):
    """Auto-generate a profile's description via the auxiliary LLM
    (``auxiliary.profile_describer``). Mirrors ``hermes profile describe
    <name> --auto``.

    A failed generation (no aux client, LLM error, …) is returned as
    ``ok: false`` with a reason rather than an HTTP error so the UI can
    surface it inline and let the operator fix config and retry.
    """
    _resolve_profile_dir(name)
    try:
        from hermes_cli import profile_describer
        outcome = profile_describer.describe_profile(name, overwrite=bool(body.overwrite))
    except Exception as e:
        _log.exception("POST /api/profiles/%s/describe-auto failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ok": bool(outcome.ok),
        "reason": outcome.reason,
        "description": outcome.description,
        # Only a successful generation is an auto-authored description. A failed
        # sweep leaves any existing description untouched, so don't claim it's
        # auto-generated.
        "description_auto": bool(outcome.ok),
    }


# ── Export / Import ──────────────────────────────────────────────────────────
# Profile sharing for the desktop: wraps hermes_cli.profiles.export_profile /
# import_profile (the same machinery behind `hermes profile export|import`).
# Paths are exchanged, not bytes — the desktop's local and pooled backends
# share the filesystem with the native save/open dialogs that produce them.


@router.post("/api/profiles/{name}/export")
async def export_profile_endpoint(name: str, body: ProfileExport):
    from hermes_cli import profiles as profiles_mod

    output = (body.output or "").strip()
    if not output:
        from hermes_constants import get_hermes_home
        staging = get_hermes_home() / "profile-exports"
        try:
            staging.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not create export directory: {exc}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = str(staging / f"{profiles_mod.normalize_profile_name(name)}-{stamp}.tar.gz")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: profiles_mod.export_profile(name, output, extra_files=body.extra_files or None),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles/%s/export failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "archive": str(result)}


@router.post("/api/profiles/import")
async def import_profile_endpoint(body: ProfileImport):
    from hermes_cli import profiles as profiles_mod

    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")

    loop = asyncio.get_running_loop()
    try:
        profile_dir = await loop.run_in_executor(
            None,
            lambda: profiles_mod.import_profile(archive, name=(body.name or "").strip() or None),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, FileExistsError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles/import failed")
        raise HTTPException(status_code=500, detail=str(e))

    imported = profile_dir.name
    # Match the CLI import flow: create the wrapper alias when it's safe.
    try:
        if not profiles_mod.check_alias_collision(imported):
            profiles_mod.create_wrapper_script(imported)
    except Exception:
        _log.exception("Creating wrapper for imported profile %s failed", imported)

    # Surface the bundled desktop appearance overlay (if the archive carried
    # one) so the desktop can apply theme/interface prefs without re-reading
    # the file over another round-trip.
    desktop_overlay = None
    overlay_path = profile_dir / "desktop.json"
    if overlay_path.is_file():
        try:
            import json as _json
            desktop_overlay = _json.loads(overlay_path.read_text(encoding="utf-8"))
        except Exception:
            _log.exception("Reading desktop.json from imported profile %s failed", imported)

    return {
        "ok": True,
        "name": imported,
        "path": str(profile_dir),
        "desktop": desktop_overlay,
    }


@router.get("/api/profiles/{name}/desktop-overlay")
async def get_profile_desktop_overlay(name: str):
    """The desktop appearance/interface overlay bundled with an imported
    profile (``desktop.json`` at the profile root), or ``exists: false``."""
    overlay_path = _resolve_profile_dir(name) / "desktop.json"
    if not overlay_path.is_file():
        return {"exists": False, "desktop": None}
    try:
        import json as _json
        return {"exists": True, "desktop": _json.loads(overlay_path.read_text(encoding="utf-8"))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read desktop.json: {e}")
