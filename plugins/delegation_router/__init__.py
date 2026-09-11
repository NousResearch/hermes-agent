"""delegation-router — route ``delegate_task`` into the specialist profile pool.

Stock Hermes delegation spawns an in-process subagent: a mini-clone of the parent
model with the parent's tools and no profile context. This plugin replaces the
single model-facing dispatch point (``AIAgent._dispatch_delegate_task``) with a
router that runs the task inside one of the configured specialist profiles — as a
``hermes -p <profile> chat --oneshot`` subprocess — so the child arrives with that
profile's skills, tools, memory, project knowledge and model.

Flow::

    model calls delegate_task(goal=..., routing="coding", project="rapid")
      -> resolve profile            (routing.yaml / explicit name)
      -> run it in that profile     (subprocess, isolated session)
      -> hand the result to the SAME async machinery stock delegation uses
         (tools.async_delegation) so it re-enters the chat as one message

Anything that does not resolve to a profile passes straight through to the stock
implementation, so ``action=list/steer/stop``, schema validation, capacity limits
and every existing behaviour are preserved exactly.

Escape hatch: any process that already carries ``HERMES_DELEGATION_ROUTER_DEPTH``
(a routed child, or a fleet worker) gets stock delegation back, so routing can
never recurse.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("plugins.delegation_router")

PLUGIN_DIR = Path(__file__).resolve().parent
# The user's table wins; the copy beside the plugin is the fallback. A bundled plugin lives in
# the install tree (read-only), so its routing table must be able to live under HERMES_HOME.
def routing_file() -> Path:
    """The routing table actually in force (user home first, then the plugin directory)."""
    user = _root() / "plugins" / "delegation-router" / "routing.yaml"
    return user if user.is_file() else (PLUGIN_DIR / "routing.yaml")
DEPTH_ENV = "HERMES_DELEGATION_ROUTER_DEPTH"
_PATCH_FLAG = "_delegation_router_patched"

# Profiles that are never a routing target even if named: running the command
# centre inside itself would recurse or stall HQ.
_FORBIDDEN_TARGETS = {"default", ""}

_SESSION_LINE = re.compile(r"^session_id:\s*(\S+)\s*$", re.MULTILINE)
# Child startup noise (missing toolset, secrets/banner warnings) is real but must
# not masquerade as the answer — keep it, move it out of the summary.
_NOISE_PREFIXES = ("warning:", " 1password", "1password:", "note:", "hint:")


def _split_noise(text: str) -> tuple[str, List[str]]:
    notes: List[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.lower().startswith(_NOISE_PREFIXES):
            notes.append(stripped)
            i += 1
            continue
        break
    return "\n".join(lines[i:]), notes


# ── paths / environment ─────────────────────────────────────────────────────

def _root() -> Path:
    """Active Hermes home (profile-aware), with an env fallback for bare imports outside Hermes."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()).expanduser()
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def _profiles_root() -> Path:
    return Path.home() / ".hermes" / "profiles"


def _hermes_bin() -> str:
    return os.environ.get("HERMES_BIN") or shutil.which("hermes") or str(Path.home() / ".local" / "bin" / "hermes")


def _runs_dir() -> Path:
    path = _root() / "delegation-router" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _profile_home(name: str) -> Optional[Path]:
    """Profile home when the profile exists on disk, else None (fail closed)."""
    if not name or name in _FORBIDDEN_TARGETS or "/" in name or name.startswith("."):
        return None
    home = _profiles_root() / name
    return home if (home / "config.yaml").is_file() else None


def _child_env() -> Dict[str, str]:
    """Env for a routed child: root config, no parent lane, no board card."""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(_root())
    for key in list(env):
        if key == "HERMES_PROFILE" or key.startswith("HERMES_KANBAN_") or key.startswith("HERMES_SESSION_"):
            env.pop(key, None)
    for key in ("HERMES_UI_SESSION_ID", "HERMES_SINGLE_QUERY_SESSION", "HERMES_SESSION_KEY"):
        env.pop(key, None)
    env[DEPTH_ENV] = str(_current_depth() + 1)
    return env


def _current_depth() -> int:
    try:
        return int(os.environ.get(DEPTH_ENV) or 0)
    except ValueError:
        return 1


def _depth_exhausted(routing: Optional[Dict[str, Any]] = None) -> bool:
    """True when this process may not route further down. ``max_depth`` in routing.yaml is the
    number of routed generations allowed under the command centre (1 = children only)."""
    routing = load_routing() if routing is None else routing
    try:
        max_depth = int((routing or {}).get("max_depth", 1))
    except (TypeError, ValueError):
        max_depth = 1
    return _current_depth() >= max_depth


# ── routing table ───────────────────────────────────────────────────────────

_ROUTING_CACHE: Dict[str, Any] = {"mtime": None, "data": {}}


def load_routing() -> Dict[str, Any]:
    """routing.yaml, cached on mtime. Unreadable/invalid => {} (stock passthrough)."""
    table = routing_file()
    try:
        stat = table.stat()
    except OSError:
        logger.warning("delegation-router: %s is missing; delegate_task stays stock", table)
        return {}
    # Identity is (path, mtime, size): mtime alone collides across two homes written in the
    # same second, which would serve one profile's table to another.
    stamp = (str(table), stat.st_mtime, stat.st_size)
    if _ROUTING_CACHE["mtime"] == stamp:
        return _ROUTING_CACHE["data"]
    try:
        import yaml

        data = yaml.safe_load(table.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # malformed YAML must never break delegation
        logger.warning("delegation-router: routing.yaml unreadable (%s); delegate_task stays stock", exc)
        data = {}
    if not isinstance(data, dict):
        data = {}
    _ROUTING_CACHE.update(mtime=stamp, data=data)
    return data


def bucket_names(routing: Optional[Dict[str, Any]] = None) -> List[str]:
    routing = load_routing() if routing is None else routing
    buckets = routing.get("buckets") or {}
    return sorted(buckets) if isinstance(buckets, dict) else []


def _project_profiles(routing: Dict[str, Any], project: str) -> Dict[str, str]:
    projects = routing.get("projects") or {}
    entry = projects.get(project) if isinstance(projects, dict) else None
    return {str(k): str(v) for k, v in entry.items()} if isinstance(entry, dict) else {}


def _candidates(routing: Dict[str, Any], bucket: str) -> List[str]:
    buckets = routing.get("buckets") or {}
    spec = buckets.get(bucket) if isinstance(buckets, dict) else None
    if isinstance(spec, dict):
        raw = spec.get("candidates") or []
    elif isinstance(spec, list):
        raw = spec
    else:
        raw = []
    return [str(c) for c in raw if isinstance(c, (str, int))]


def _bucket_from_keywords(routing: Dict[str, Any], text: str) -> Optional[str]:
    """Highest-scoring bucket whose keyword list matches the goal text."""
    buckets = routing.get("buckets") or {}
    if not isinstance(buckets, dict):
        return None
    low = (text or "").lower()
    best: Optional[tuple] = None
    for name in sorted(buckets):
        spec = buckets.get(name)
        keywords = spec.get("keywords") if isinstance(spec, dict) else None
        if not isinstance(keywords, list):
            continue
        hits = sum(1 for kw in keywords if isinstance(kw, str) and kw and kw.lower() in low)
        if hits and (best is None or hits > best[0]):
            best = (hits, name)
    return best[1] if best else None


def _bucket_from_keywords_for_goal(routing: Dict[str, Any], goal: str, context: str = "") -> Optional[str]:
    return _bucket_from_keywords(routing, f"{goal or ''}\n{context or ''}")


def _concrete_candidate(
    routing: Dict[str, Any], bucket: str, project: Optional[str], lease_policy: str
) -> Optional[Dict[str, str]]:
    """First existing candidate for a bucket, honouring `project` then lease policy."""
    order: List[str] = []
    if project:
        role_map = _project_profiles(routing, project)
        spec = (routing.get("buckets") or {}).get(bucket)
        role_hint = ""
        if isinstance(spec, dict):
            role_hint = str(spec.get("role") or "")
        # The project map is keyed by role name (dev/brand/growth/support/...).
        # Use it only when this bucket maps to a role the project actually has.
        # Falling back to "first profile in the project" sent design/ads/qa work
        # to the project's dev profile instead of the shared desk.
        key = role_hint or bucket
        if key in role_map:
            order.append(role_map[key])
        elif role_hint:
            # A project role was named explicitly but this project does not define
            # it: fall back to the project's own profiles.
            order.extend(role_map.values())
    order.extend(_candidates(routing, bucket))

    seen: set = set()
    skipped_for_lease: List[str] = []
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        home = _profile_home(name)
        if home is None:
            continue
        leases = _live_leases(name)
        if leases and lease_policy == "skip":
            skipped_for_lease.append(f"{name} (held by {leases[0]})")
            continue
        return {"profile": name, "bucket": bucket, "leases": leases,
                "skipped_for_lease": skipped_for_lease}
    if skipped_for_lease:
        # Every candidate was busy: run the first one anyway and report it.
        first = next((n for n in seen if _profile_home(n)), None)
        if first:
            return {"profile": first, "bucket": bucket, "leases": _live_leases(first),
                    "skipped_for_lease": skipped_for_lease}
    return None


def resolve_target(args: Dict[str, Any], *, routing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Decide where one task goes. Returns ``{"action": "route"|"passthrough"|"error", ...}``."""
    routing = load_routing() if routing is None else routing
    if not routing or not routing.get("enabled", True):
        return {"action": "passthrough", "reason": "delegation-router disabled or no routing.yaml"}
    if _depth_exhausted(routing):
        return {"action": "passthrough", "reason": f"routed-run depth {_current_depth()} reached max_depth"}

    lease_policy = str(routing.get("lease_policy") or "warn").strip().lower()

    explicit = str(args.get("profile") or "").strip()
    if explicit:
        if _profile_home(explicit) is None:
            return {"action": "error", "reason": (
                f"unknown profile {explicit!r}. Available: "
                + ", ".join(sorted(p.name for p in _profiles_root().iterdir() if (p / 'config.yaml').is_file()))
            )}
        return {"action": "route", "profile": explicit, "bucket": "explicit",
                "leases": _live_leases(explicit), "reason": "explicit profile argument"}

    bucket = str(args.get("routing") or "").strip().lower()
    project = str(args.get("project") or "").strip().lower() or None
    context = str(args.get("context") or "")
    goal = str(args.get("goal") or "")
    reason = "bucket argument"

    if not bucket:
        bucket = _bucket_from_keywords_for_goal(routing, goal, context) or ""
        reason = f"keyword match on the goal ({bucket})" if bucket else ""

    if not bucket:
        default = str(routing.get("default_profile") or "").strip()
        if default and _profile_home(default):
            return {"action": "route", "profile": default, "bucket": "default",
                    "leases": _live_leases(default), "reason": "default_profile"}
        return {"action": "passthrough", "reason": "no bucket or profile resolved"}

    if bucket not in {b.lower() for b in bucket_names(routing)}:
        return {"action": "error", "reason": (
            f"unknown routing bucket {bucket!r}. Valid buckets: " + ", ".join(bucket_names(routing))
        )}

    if not project and goal:
        # A project name inside the goal is a strong signal — use it when it helps.
        for key in (routing.get("projects") or {}):
            if re.search(rf"\b{re.escape(str(key))}\b", goal, re.IGNORECASE):
                project = str(key)
                reason += f" + project keyword {project}"
                break

    target = _concrete_candidate(routing, bucket, project, lease_policy)
    if target is None:
        return {"action": "passthrough", "reason": f"bucket {bucket!r} has no available profile"}
    target.update(action="route", reason=reason or f"bucket {bucket}", project=project)
    return target


# ── personas ────────────────────────────────────────────────────────────────
# A persona is a role charter (PERSONA.md) plus a bundle of the skills that role
# needs. The router loads it into every routed run; a profile can also pin one for
# its own sessions with `persona: <name>` in its config.yaml.

def _personas_dir() -> Path:
    return _root() / "personas"


def load_persona(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """{name, charter, members, buckets, profiles} for a persona, or None."""
    if not name or "/" in str(name):
        return None
    home = _personas_dir() / str(name)
    charter_path = home / "PERSONA.md"
    if not charter_path.is_file():
        return None
    try:
        charter = charter_path.read_text(encoding="utf-8").strip()
        manifest: Dict[str, Any] = {}
        if (home / "persona.json").is_file():
            manifest = json.loads((home / "persona.json").read_text(encoding="utf-8")) or {}
        members: List[str] = []
        if (home / "bundle.json").is_file():
            members = [str(m) for m in (json.loads((home / "bundle.json").read_text(encoding="utf-8")).get("members") or [])]
    except Exception as exc:
        logger.warning("delegation-router: persona %s unreadable (%s)", name, exc)
        return None
    return {
        "name": str(name), "charter": charter, "members": members,
        "buckets": manifest.get("buckets") or [], "profiles": manifest.get("profiles") or [],
        "job": manifest.get("job") or "",
    }


def persona_for_bucket(bucket: str, routing: Optional[Dict[str, Any]] = None) -> Optional[str]:
    routing = load_routing() if routing is None else routing
    spec = (routing.get("buckets") or {}).get(bucket)
    if isinstance(spec, dict) and spec.get("persona"):
        return str(spec["persona"])
    return None


def profile_persona(profile: Optional[str]) -> Optional[str]:
    """A profile's pinned persona: `persona: <name>` in its config.yaml."""
    home = _root() if (not profile or profile == "default") else (_profiles_root() / str(profile))
    cfg = home / "config.yaml"
    if not cfg.is_file():
        return None
    try:
        import yaml
        value = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("persona")
    except Exception:
        return None
    return str(value).strip() or None if value else None


_MEMBER_INDEX: Dict[str, set] = {}


def _skill_index_for(profile: str) -> Dict[str, int]:
    """name -> how many roots of this profile hold it (cached per profile).

    Exactly the profile's own skills dir plus its configured external dirs — no globbing
    of other `shared-skills-*` dirs, because an unconfigured dir is invisible to it.
    A count above 1 means the loader refuses that name outright ("Refusing to guess"),
    so it must not be preloaded.
    """
    if profile in _MEMBER_INDEX:
        return _MEMBER_INDEX[profile]
    home = _profiles_root() / profile
    roots = [home / "skills"]
    try:
        import yaml
        cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        for d in (cfg.get("skills") or {}).get("external_dirs") or []:
            roots.append(Path(str(d)).expanduser())
    except Exception:
        pass
    index: Dict[str, int] = {}
    for root in roots:
        if not root.is_dir():
            continue
        seen_here: set = set()
        for path in root.rglob("SKILL.md"):
            rel = path.parent.relative_to(root).parts
            if any(part.startswith(".") for part in rel):
                continue
            name = path.parent.name
            if name in seen_here:
                continue
            seen_here.add(name)
            index[name] = index.get(name, 0) + 1
    _MEMBER_INDEX[profile] = index
    return index


def _skill_names_for(profile: str) -> set:
    """Names this profile can load at all (kept for callers that only need a set)."""
    return set(_skill_index_for(profile))


def resolvable_members(profile: str, members: List[str], limit: int = 25) -> List[str]:
    """Members that load cleanly for this profile, in bundle order (deduped, capped)."""
    return resolve_members(profile, members, limit=limit)["preload"]


def resolve_members(profile: str, members: List[str], limit: int = 25) -> Dict[str, List[str]]:
    """Split a persona bundle into preloadable / ambiguous / unavailable skill names.

    Ambiguous names are left out deliberately: two visible copies make the skill loader
    refuse the bare name, so passing them to `--skills` would fail silently.
    """
    index = _skill_index_for(profile)
    out: Dict[str, List[str]] = {"preload": [], "ambiguous": [], "unavailable": []}
    for member in members:
        count = index.get(member, 0)
        if count == 0:
            out["unavailable"].append(member)
        elif count > 1:
            out["ambiguous"].append(member)
        elif member not in out["preload"] and len(out["preload"]) < limit:
            out["preload"].append(member)
    return out


# ── lease probing ───────────────────────────────────────────────────────────

def _live_leases(profile: str) -> List[str]:
    """Live turn leases in a profile (an active Discord/TUI session)."""
    db = _profiles_root() / profile / "state.db"
    if not db.is_file():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT conversation_id, holder FROM session_turn_leases WHERE expires_at > ?",
                (time.time(),),
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return []
    return [f"{conv} (holder={holder})" for conv, holder in rows]


# ── running one profile ─────────────────────────────────────────────────────

_PROMPT = """[ROUTED DELEGATION — {run_id}]
You are running as the `{profile}` profile, which owns the {bucket} workstream.
This task was routed here by the delegation-router instead of being given to a
generic subagent, so use THIS profile's own skills, tools, memory and project
knowledge. Work in {cwd}.
{charter_block}
Finish the task and verify it with real evidence (command output, file paths,
URLs) before you answer. Do not hand routine work back to HQ.
Answer with the finished result only — nothing after it is read.

TASK:
{goal}
{context_block}"""


def _build_prompt(profile: str, bucket: str, run_id: str, goal: str, context: str, cwd: str,
                  charter: str = "", preloaded: Optional[List[str]] = None) -> str:
    context_block = f"\nCONTEXT FROM THE DELEGATING SESSION:\n{context.strip()}\n" if context and context.strip() else ""
    charter_block = ""
    if charter:
        preload_note = ""
        if preloaded:
            preload_note = ("\nThese member skills are preloaded for this run: "
                            + ", ".join(f"`{m}`" for m in preloaded) + ".")
        charter_block = (
            f"\nROLE CHARTER you are acting under (binding for this run){preload_note}\n"
            "---8<--- CHARTER START ---8<---\n"
            f"{charter[:6000]}\n"
            "---8<--- CHARTER END ---8<---\n"
        )
    return _PROMPT.format(profile=profile, bucket=bucket, run_id=run_id, goal=(goal or "").strip(),
                          context_block=context_block, cwd=cwd, charter_block=charter_block)


# ── visibility: register routed runs where /agents, action=list and the TUI look ──
#
# Stock children are AIAgent objects tracked in tools.delegate_tool_registry. A routed
# run is a subprocess, so this proxy carries what that registry reads: an id, a stop
# hook (interrupt), a steer hook, a session id and the lineage weakref to the parent.

class _RoutedChildProxy:
    """Stand-in for the child AIAgent that the subagent registry and TUI overlay inspect."""

    def __init__(self, *, subagent_id: str, parent_agent: Any, delegation_id: str, profile: str,
                 goal: str, log_path: Path):
        import weakref
        self._subagent_id = subagent_id
        self._delegation_id = delegation_id
        self._parent_subagent_id = getattr(parent_agent, "_subagent_id", None)
        self._parent_session_id = str(getattr(parent_agent, "session_id", "") or "")
        self._delegate_parent_ref = weakref.ref(parent_agent) if parent_agent is not None else None
        self._delegate_depth = int(getattr(parent_agent, "_delegate_depth", 0) or 0) + 1
        self._live_transcript_path = str(log_path)
        self.model = f"profile:{profile}"
        self.session_id = None            # filled in once the child prints its session id
        self.routed_profile = profile
        self.goal = goal
        self._proc: Optional[subprocess.Popen] = None
        self._interrupted = False
        self._steer_file: Optional[Path] = None
        self.api_call_count = 0
        self._steers: List[str] = []

    # --- what interrupt_subagent / request_hard_interrupt call ---
    def hard_interrupt(self, message: Optional[str] = None, **_: Any) -> bool:
        self._interrupted = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
        return True

    interrupt = hard_interrupt

    # --- what steer_subagent calls; a subprocess cannot take mid-turn text, so record it ---
    def steer(self, text: str) -> bool:
        self._steers.append(text)
        if self._steer_file is not None:
            try:
                with self._steer_file.open("a", encoding="utf-8") as fh:
                    fh.write(text.rstrip() + "\n")
            except OSError:
                return False
        return True

    def get_activity_summary(self) -> Dict[str, Any]:
        proc = self._proc
        alive = proc is not None and proc.poll() is None
        return {"api_call_count": self.api_call_count, "current_tool": f"hermes -p {self.routed_profile}" if alive else None,
                "last_activity_ts": time.time() if alive else None}


def _register_routed_child(proxy: _RoutedChildProxy, parent_agent: Any) -> None:
    try:
        from tools.delegate_tool_registry import _register_subagent
    except Exception as exc:
        logger.debug("delegation-router: subagent registry unavailable (%s)", exc)
        return
    _register_subagent({
        "subagent_id": proxy._subagent_id,
        "parent_id": proxy._parent_subagent_id,
        "depth": max(0, proxy._delegate_depth - 1),
        "goal": proxy.goal,
        "delegation_id": proxy._delegation_id,
        "model": proxy.model,
        "started_at": time.time(), "status": "running", "tool_count": 0, "agent": proxy,
        "owner_agent_session_id": proxy._parent_session_id or None,
        "owner_session_id": None, "owner_transport": None, "owner_session_record": None,
        "routed_profile": proxy.routed_profile,
    })


def _unregister_routed_child(proxy: _RoutedChildProxy) -> None:
    try:
        from tools.delegate_tool_registry import _unregister_subagent
        _unregister_subagent(proxy._subagent_id, agent=proxy)
    except Exception:
        logger.debug("delegation-router: unregister failed", exc_info=True)


def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the child's whole process group."""
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_profile_task(
    *,
    profile: str,
    bucket: str,
    goal: str,
    context: str = "",
    project: Optional[str] = None,
    persona: Optional[str] = None,
    routing: Optional[Dict[str, Any]] = None,
    task_index: int = 0,
    deadline: Optional[float] = None,
    cancel: Optional[threading.Event] = None,
    register_proc=None,
    parent_agent: Any = None,
    delegation_id: Optional[str] = None,
    subagent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one goal inside one profile. Returns a stock-shaped result entry."""
    routing = load_routing() if routing is None else routing
    started = time.monotonic()
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{profile}_{uuid.uuid4().hex[:6]}"
    subagent_id = subagent_id or f"sa-{task_index}-{uuid.uuid4().hex[:8]}"
    cwd = str(Path.cwd())

    persona_name = persona or persona_for_bucket(bucket, routing) or profile_persona(profile)
    persona_obj = load_persona(persona_name)
    member_split = resolve_members(profile, persona_obj["members"]) if persona_obj else {
        "preload": [], "ambiguous": [], "unavailable": []}
    preloaded = member_split["preload"]
    prompt_text = _build_prompt(
        profile, bucket, run_id, goal, context, cwd,
        charter=persona_obj["charter"] if persona_obj else "",
        preloaded=preloaded,
    )

    runs = _runs_dir()
    prompt_path = runs / f"{run_id}.prompt.txt"
    log_path = runs / f"{run_id}.log"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    home = _profile_home(profile)
    entry: Dict[str, Any] = {
        "task_index": task_index, "goal": goal, "status": "error", "summary": None,
        "error": None, "duration_seconds": 0.0, "routed_profile": profile,
        "routing_bucket": bucket, "routing_project": project, "run_id": run_id,
        "subagent_id": subagent_id, "delegation_id": delegation_id,
        "persona": persona_obj["name"] if persona_obj else None,
        "skills_preloaded": preloaded or None,
        "skills_ambiguous_not_preloaded": member_split["ambiguous"] or None,
        "skills_unavailable": member_split["unavailable"] or None,
        "session_id": None, "exit_code": None,
        "live_transcript": str(log_path), "prompt_file": str(prompt_path),
    }
    if home is None:
        entry["error"] = f"profile {profile!r} does not exist on disk"
        return entry

    leases = _live_leases(profile)
    if leases:
        entry["concurrent_leases"] = leases
        entry["lease_note"] = (
            f"profile {profile} had a live session turn lease ({leases[0]}); this run used its own "
            "separate CLI session"
        )

    max_turns = int(routing.get("max_turns") or 800)
    budget = int(routing.get("max_runtime_seconds") or 3600)
    cmd = [
        _hermes_bin(), "-p", profile, "chat",
        "--query-file", str(prompt_path), "--oneshot", "--quiet",
        "--max-turns", str(max_turns), "--run-budget", str(budget),
        "--no-restore-cwd",
    ]
    if os.environ.get("HERMES_DELEGATION_ROUTER_ACCEPT_HOOKS") == "1":
        cmd.append("--accept-hooks")
    if preloaded:
        cmd += ["-s", ",".join(preloaded)]

    proc: Optional[subprocess.Popen] = None
    timed_out = False
    cancelled = False
    proxy = _RoutedChildProxy(subagent_id=subagent_id, parent_agent=parent_agent,
                              delegation_id=delegation_id or "", profile=profile, goal=goal, log_path=log_path)
    proxy._steer_file = runs / f"{run_id}.steer.txt"
    _register_routed_child(proxy, parent_agent)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=_child_env(), stdout=log, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            proxy._proc = proc
            if register_proc is not None:
                register_proc(proc)
            wait_for = None
            if deadline is not None:
                wait_for = max(5.0, deadline - time.monotonic())
            try:
                entry["exit_code"] = proc.wait(timeout=wait_for)
            except subprocess.TimeoutExpired:
                timed_out = True
                entry["error"] = f"routed run exceeded its {budget}s budget"
                _kill_tree(proc)
                entry["exit_code"] = proc.poll()
            if not timed_out and ((cancel is not None and cancel.is_set()) or proxy._interrupted):
                cancelled = True
                entry["error"] = "routed run stopped by the delegating session"
                _kill_tree(proc)
    except Exception as exc:  # a failed spawn must never lose the task silently
        entry["error"] = f"could not run profile {profile}: {type(exc).__name__}: {exc}"
        _unregister_routed_child(proxy)
        return entry
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
    _unregister_routed_child(proxy)
    if proxy._steers:
        entry["steers_received_not_deliverable"] = proxy._steers
        entry.setdefault("child_notes", []).append(
            f"{len(proxy._steers)} steer message(s) were queued but a routed profile run cannot take mid-turn "
            f"text; they are saved at {proxy._steer_file} — re-dispatch with them in context if still needed."
        )

    entry["duration_seconds"] = round(time.monotonic() - started, 2)
    raw = ""
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    match = _SESSION_LINE.search(raw)
    if match:
        entry["session_id"] = match.group(1)
        body = raw[:match.start()] + raw[match.end():]
    else:
        body = raw
    body, notes = _split_noise(body)
    summary = body.strip()
    if notes:
        entry["child_notes"] = notes
    entry["summary"] = summary[:20000] if summary else None

    if cancelled:
        entry["status"] = "interrupted"
    elif timed_out:
        entry["status"] = "timeout"
    elif entry["exit_code"] == 0 and summary:
        entry["status"] = "completed"
        entry.pop("error", None)
    else:
        entry["status"] = "error"
        if not entry.get("error"):
            tail = summary[-800:] if summary else "(no output)"
            entry["error"] = f"profile {profile} exited {entry['exit_code']}: {tail}"
    entry["resume_hint"] = f"hermes -p {profile} --resume {entry['session_id']}" if entry.get("session_id") else None
    if member_split["ambiguous"]:
        # Two visible copies of the same skill name make the loader refuse it. Say so
        # instead of letting a charter name a playbook the child cannot open.
        notes = list(entry.get("child_notes") or [])
        notes.append(
            "Not preloaded (duplicate name in two skill dirs — the loader refuses an ambiguous "
            "skill): " + ", ".join(member_split["ambiguous"])
        )
        entry["child_notes"] = notes
    _audit(entry)
    return entry


def _audit(entry: Dict[str, Any]) -> None:
    try:
        index = _runs_dir().parent / "runs.jsonl"
        with index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**entry, "at": time.time()}, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("delegation-router: audit write failed", exc_info=True)


# ── the routed delegation core ──────────────────────────────────────────────

def _normalise_tasks(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        return [t for t in tasks if isinstance(t, dict)]
    goal = args.get("goal")
    if goal:
        return [{"goal": goal, "context": args.get("context"), "profile": args.get("profile"),
                 "routing": args.get("routing"), "project": args.get("project")}]
    return []


def _sync_note(reason: str) -> str:
    if reason == "no_async":
        return ("This session cannot receive a detached result later (one-shot CLI, cron job or Kanban "
                "worker), so the routed task(s) ran synchronously and the results are above.")
    return ("The background delegation pool was at capacity (delegation.max_concurrent_children), so the "
            "routed task(s) ran synchronously and the results are above.")


def _run_tasks(tasks: List[Dict[str, Any]], routing: Dict[str, Any], cancel: threading.Event,
               register_proc, max_workers: int, *, parent_agent: Any = None,
               delegation_id: Optional[str] = None, subagent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run every task in its routed profile (in parallel) and return the combined dict.

    ``subagent_ids`` (pre-allocated, one per task) lets the dispatch handle name the ids the
    model can later ``stop``; ``run_profile_task`` registers each under that id."""
    started = time.monotonic()
    targets: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks):
        args = {**task, "context": task.get("context") or ""}
        decision = resolve_target(args, routing=routing)
        targets.append(decision)

    def _one(i: int, task: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        if decision.get("action") == "error":
            return {"task_index": i, "goal": task.get("goal"), "status": "error",
                    "error": decision.get("reason"), "summary": None, "duration_seconds": 0.0,
                    "routed_profile": None}
        if decision.get("action") != "route":
            # Belt and braces: this task should never have been routed.
            return {"task_index": i, "goal": task.get("goal"), "status": "error",
                    "error": f"router could not resolve a profile ({decision.get('reason')})",
                    "summary": None, "duration_seconds": 0.0, "routed_profile": None}
        try:
            return run_profile_task(
                profile=decision["profile"], bucket=decision.get("bucket") or "unknown",
                goal=str(task.get("goal") or ""), context=str(task.get("context") or ""),
                project=decision.get("project"), routing=routing, task_index=i,
                cancel=cancel, register_proc=register_proc, parent_agent=parent_agent,
                delegation_id=delegation_id,
                subagent_id=(subagent_ids[i] if subagent_ids and i < len(subagent_ids) else None),
            )
        except Exception as exc:  # never lose a task
            return {"task_index": i, "goal": task.get("goal"), "status": "error",
                    "error": f"{type(exc).__name__}: {exc}", "summary": None,
                    "duration_seconds": 0.0, "routed_profile": decision.get("profile")}

    if len(targets) == 1:
        results = [_one(0, tasks[0], targets[0])]
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(targets)))) as pool:
            futures = [pool.submit(_one, i, t, d) for i, (t, d) in enumerate(zip(tasks, targets))]
            results = [f.result() for f in futures]
    results.sort(key=lambda r: r.get("task_index", 0))
    combined = {"results": results, "total_duration_seconds": round(time.monotonic() - started, 2)}
    profiles = [r.get("routed_profile") for r in results if r.get("routed_profile")]
    if profiles:
        combined["routed_profiles"] = profiles
    return combined


def _dispatch_payload(args: Dict[str, Any]) -> str:
    """Router entry point shared by the model path and the registry path."""
    routing = load_routing()
    tasks = _normalise_tasks(args)
    if not tasks:
        return json.dumps({"error": "delegate_task needs tasks:[{goal}] or a goal."})
    decisions = [resolve_target({**t, "context": t.get("context") or ""}, routing=routing) for t in tasks]
    unresolvable = [(i, d) for i, d in enumerate(decisions) if d.get("action") != "route"]
    if unresolvable:
        i, d = unresolvable[0]
        if d.get("action") == "error":
            return json.dumps({
                "error": d.get("reason"),
                "hint": "Pass a valid `routing` bucket or an existing `profile` name.",
                "valid_buckets": bucket_names(routing),
            })
        return json.dumps({
            "status": "passthrough", "reason": d.get("reason"),
            "note": "No specialist profile matched; stock delegate_task handles it.",
        })

    cancel = threading.Event()
    procs: List[subprocess.Popen] = []
    lock = threading.Lock()

    def register_proc(proc: subprocess.Popen) -> None:
        with lock:
            procs.append(proc)

    def interrupt() -> None:
        # Called by the stock async machinery (parent interrupt / CLI shutdown, via
        # interrupt_all or interrupt_for_session) — a routed child dies with its session,
        # exactly like a stock background subagent. Name the caller so a lost batch is
        # explainable from the log instead of looking like a crash.
        if logger.isEnabledFor(logging.INFO):
            import traceback as _tb
            frames = [f for f in _tb.format_stack() if "/hermes-agent/" in f]
            caller = (frames[-1].strip().splitlines()[0] if frames else "unknown")
            logger.info("delegation-router: %s interrupted (%s)", delegation_id, caller)
        cancel.set()
        with lock:
            live = list(procs)
        for proc in live:
            _kill_tree(proc)

    try:
        from tools.delegate_tool import _get_max_async_children, _get_max_concurrent_children
        max_children = max(1, int(_get_max_concurrent_children()))
        max_async = max(1, int(_get_max_async_children()))
    except Exception:
        max_children, max_async = 3, 3

    goals = [str(t.get("goal") or "") for t in tasks]
    parent_agent = args.get("_parent_agent")
    parent_session_id = getattr(parent_agent, "session_id", None) if parent_agent is not None else None

    # Mirror stock dispatch: detached when this session can receive a later message,
    # inline when it cannot (one-shot CLI, cron, Kanban worker).
    try:
        from tools.delegate_tool_dispatch import (
            _capture_origin,
            _resolve_async_session_key,
            _resolve_async_wake_sid,
        )
        origin_wake_sid, origin_ui_session_id, _t, _r, history_delivery = _capture_origin()
        wake_sid = _resolve_async_wake_sid(origin_wake_sid, history_delivery)
        session_key, origin_ui = _resolve_async_session_key(parent_agent, origin_ui_session_id)
    except Exception as exc:
        logger.debug("delegation-router: async plumbing unavailable (%s); running inline", exc)
        wake_sid = None

    if wake_sid is None:
        sync_id = f"deleg_{uuid.uuid4().hex[:8]}"
        combined = _run_tasks(tasks, routing, cancel, register_proc, max_children,
                              parent_agent=parent_agent, delegation_id=sync_id)
        combined["mode"] = "sync"
        combined["delegation_id"] = sync_id
        combined["note"] = _sync_note("no_async")
        return json.dumps(combined, ensure_ascii=False)

    delegation_id = f"deleg_{uuid.uuid4().hex[:8]}"
    subagent_ids = [f"sa-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(tasks))]

    def runner() -> Dict[str, Any]:
        return _run_tasks(tasks, routing, cancel, register_proc, max_children,
                          parent_agent=parent_agent, delegation_id=delegation_id, subagent_ids=subagent_ids)

    from tools.async_delegation import dispatch_async_delegation_batch

    handle = dispatch_async_delegation_batch(
        goals=goals, context=args.get("context"), toolsets=None,
        role="routed-profile:" + ",".join(sorted({d["profile"] for d in decisions})),
        model="profile-session", session_key=session_key, origin_ui_session_id=origin_ui,
        origin_session_id=wake_sid, parent_session_id=parent_session_id,
        runner=runner, interrupt_fn=interrupt, delegation_id=delegation_id,
        max_async_children=max_async, progress_fn=None,
    )
    if handle.get("status") != "dispatched":
        combined = _run_tasks(tasks, routing, cancel, register_proc, max_children,
                              parent_agent=parent_agent, delegation_id=delegation_id)
        combined["mode"] = "sync"
        combined["delegation_id"] = delegation_id
        combined["note"] = _sync_note("at_capacity") + f" (router: {handle.get('error')})"
        return json.dumps(combined, ensure_ascii=False)

    plan = [{"task_index": i, "profile": d.get("profile"), "bucket": d.get("bucket"),
             "subagent_id": subagent_ids[i]} for i, d in enumerate(decisions)]
    payload = {
        "status": "dispatched", "mode": "background-profiles", "count": len(tasks),
        "delegation_id": delegation_id, "goals": goals, "routed_plan": plan,
        "subagent_ids": subagent_ids,
        "routed_profiles": sorted({d["profile"] for d in decisions}),
        "note": (
            f"Routed into {len({d['profile'] for d in decisions})} specialist profile(s) running as their own "
            "Hermes sessions (their skills, tools and memory — not clones of this model). The consolidated "
            "result re-enters this conversation as one new message when the last one finishes. Read any run "
            f"transcript live under {_runs_dir()}. End your turn after anything that does not depend on it; "
            "do not poll."
        ),
        "control_hint": (
            "delegate_task(action='list') shows these as live subagents; action='stop' with a subagent_id "
            "ends that profile run early. action='steer' is recorded but cannot reach a running profile "
            "session mid-turn — re-dispatch with the correction in context instead."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _stock_call(args: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
    """Stock delegate_task, called exactly as the built-in registry lambda calls it."""
    from tools.delegate_tool import _strip_model_hidden_task_fields, delegate_task as _delegate_task

    try:
        from tools.delegate_tool import _model_background_value
        background = _model_background_value(args, kwargs.get("parent_agent"))
    except Exception:
        background = not getattr(kwargs.get("parent_agent"), "_delegate_depth", 0) > 0
    return _delegate_task(
        goal=args.get("goal"), context=args.get("context"),
        tasks=_strip_model_hidden_task_fields(args.get("tasks")),
        max_iterations=args.get("max_iterations"), role=args.get("role"),
        background=background, output_schema=args.get("output_schema"),
        action=args.get("action"), subagent_id=args.get("subagent_id"),
        message=args.get("message"), parent_agent=kwargs.get("parent_agent"),
    )


def _registry_dispatch(args: Dict[str, Any], **kwargs: Any) -> str:
    """Registry entry point: route when possible, else run stock delegation."""
    routing = load_routing()
    action = str(args.get("action") or "spawn").strip().lower()
    tasks = _normalise_tasks(args)
    if routing and routing.get("enabled", True) and tasks and action not in ("list", "steer", "stop") \
            and not _depth_exhausted(routing):
        decisions = [resolve_target({**t, "context": t.get("context") or ""}, routing=routing) for t in tasks]
        if all(d.get("action") == "route" for d in decisions):
            try:
                return _dispatch_payload({**args, "_parent_agent": kwargs.get("parent_agent")})
            except Exception:
                logger.exception("delegation-router: routing failed; running stock delegation")
    return _stock_call(args, kwargs)


# ── the model-facing dispatch patch ─────────────────────────────────────────

def _resolve_and_dispatch(agent, function_args: Dict[str, Any], original) -> str:
    """Route when a profile resolves; otherwise behave exactly like stock."""
    args = dict(function_args or {})
    if str(args.get("action") or "spawn").strip().lower() in ("list", "steer", "stop"):
        return original(agent, args)
    routing = load_routing()
    if not routing or not routing.get("enabled", True):
        return original(agent, args)
    if _depth_exhausted(routing):
        return original(agent, args)
    tasks = _normalise_tasks(args)
    if not tasks:
        return original(agent, args)
    decisions = [resolve_target({**t, "context": t.get("context") or ""}, routing=routing) for t in tasks]
    if any(d.get("action") != "route" for d in decisions):
        # Any unresolvable task: stock delegation owns the whole call, except for a
        # hard error (a bad bucket/profile name), which the model must be told about.
        hard = [d for d in decisions if d.get("action") == "error"]
        if hard and all(d.get("action") == "error" for d in decisions):
            return json.dumps({
                "error": hard[0].get("reason"),
                "hint": "Pass a valid `routing` bucket or an existing `profile` name.",
                "valid_buckets": bucket_names(routing),
            })
        return original(agent, args)
    args["_parent_agent"] = agent
    try:
        return _dispatch_payload(args)
    except Exception as exc:
        logger.exception("delegation-router: routing failed; falling back to stock delegate_task")
        args.pop("_parent_agent", None)
        return original(agent, args)


def _core_supports_delegate_override() -> bool:
    """True when core asks the registry who owns ``delegate_task`` (this fork's hook).

    With the hook present the registry handler below IS the dispatch entry point — no
    monkey-patch. Without it (stock Hermes) ``_install_patch`` keeps this plugin working,
    so one file serves both trees."""
    try:
        from tools.registry import ToolRegistry
        return callable(getattr(ToolRegistry, "plugin_handler", None))
    except Exception:
        return False


def _install_patch() -> bool:
    """Replace AIAgent._dispatch_delegate_task (the single model-facing dispatch site)."""
    try:
        from run_agent import AIAgent
    except Exception as exc:
        logger.warning("delegation-router: cannot import AIAgent (%s); delegate_task stays stock", exc)
        return False
    current = getattr(AIAgent, "_dispatch_delegate_task", None)
    if current is None:
        logger.warning(
            "delegation-router: AIAgent._dispatch_delegate_task is gone in this Hermes build "
            "(v0.21.1+ moved it?). delegate_task stays stock — the schema hint is still applied."
        )
        return False
    if getattr(current, _PATCH_FLAG, False):
        return True
    original = current

    def _patched(self, function_args):  # noqa: ANN001 - mirrors the stock signature
        return _resolve_and_dispatch(self, function_args, lambda ag, a: original(ag, a))

    setattr(_patched, _PATCH_FLAG, True)
    _patched.__wrapped__ = original
    AIAgent._dispatch_delegate_task = _patched
    logger.info("delegation-router: delegate_task now routes into the specialist profile pool")
    return True


# ── schema (what the model sees) ────────────────────────────────────────────

def _pool_hint() -> str:
    routing = load_routing()
    parts = []
    for bucket in bucket_names(routing):
        candidates = _candidates(routing, bucket)
        if candidates:
            parts.append(f"{bucket}->{candidates[0]}")
    projects = ", ".join(sorted((routing.get("projects") or {}).keys()))
    hint = (
        "\n\nPROFILE ROUTING (delegation-router): this delegate_task does not spawn a clone of you. "
        "Name the workstream and the task runs inside a specialist profile that owns it — with that "
        "profile's skills, tools and memory. Pass `routing` (bucket) and, when you know the business, "
        f"`project` ({projects or 'none configured'}), or pass `profile` to name one directly.\n"
        f"Buckets: {'; '.join(parts) if parts else '(none configured)'}.\n"
        "Each bucket runs under a ROLE CHARTER (persona): the role's operating contract, authority "
        "boundary, evidence rules and its member skills, preloaded for that run.\n"
        "Unrouted calls (no bucket/profile, or an action=list/steer/stop) fall back to stock subagents."
    )
    return hint


def _route_params() -> Dict[str, Any]:
    return {
        "profile": {
            "type": "string",
            "description": ("Run this task inside a named Hermes profile (its own skills, tools, memory, "
                            "model). Takes precedence over routing/project."),
        },
        "routing": {
            "type": "string",
            "description": ("Workstream bucket: which specialist profile owns this kind of work. "
                            "See the bucket list in the tool description."),
        },
        "project": {
            "type": "string",
            "description": "Business the task belongs to, so a project-scoped profile is chosen (e.g. acme, beta — see projects in routing.yaml).",
        },
    }


def _build_schema() -> Dict[str, Any]:
    try:
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        base = json.loads(json.dumps(DELEGATE_TASK_SCHEMA))  # deep copy, never mutate stock
    except Exception:
        base = {
            "name": "delegate_task",
            "description": "Spawn subagents. (delegation-router: stock schema unavailable.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "minItems": 1, "items": {
                        "type": "object", "properties": {"goal": {"type": "string"}, "context": {"type": "string"}},
                        "required": ["goal"]}},
                },
                "required": [],
            },
        }
    params = base.setdefault("parameters", {}).setdefault("properties", {})
    params.update(_route_params())
    tasks = params.get("tasks")
    if isinstance(tasks, dict) and isinstance(tasks.get("items"), dict):
        item_props = tasks["items"].setdefault("properties", {})
        item_props.update(_route_params())
    base["description"] = (base.get("description") or "") + _pool_hint()
    return base


# ── status tool ─────────────────────────────────────────────────────────────

def delegation_router_status(limit: int = 10) -> str:
    """Pool + recent routed runs. Read-only; safe for the model to call."""
    routing = load_routing()
    pool = {}
    for bucket in bucket_names(routing):
        pool[bucket] = [{"profile": c, "exists": _profile_home(c) is not None,
                         "leases": _live_leases(c)} for c in _candidates(routing, bucket)]
    recent = []
    try:
        index = _runs_dir().parent / "runs.jsonl"
        lines = index.read_text(encoding="utf-8").splitlines()[-max(1, int(limit)):]
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            recent.append({k: entry.get(k) for k in
                           ("run_id", "goal", "routed_profile", "routing_bucket", "persona",
                            "status", "duration_seconds", "session_id", "live_transcript", "error")})
    except OSError:
        pass
    personas = []
    for role_dir in sorted(p for p in _personas_dir().iterdir() if p.is_dir()) if _personas_dir().is_dir() else []:
        persona = load_persona(role_dir.name)
        if persona:
            personas.append({"name": persona["name"], "job": persona.get("job"),
                             "buckets": persona.get("buckets"), "profiles": persona.get("profiles"),
                             "members": len(persona["members"])})
    return json.dumps({
        "enabled": bool(routing and routing.get("enabled", True)),
        "routing_file": str(routing_file()),
        "default_profile": routing.get("default_profile") or None,
        "lease_policy": routing.get("lease_policy"),
        "buckets": pool,
        "personas": personas,
        "host_persona": profile_persona(os.environ.get("HERMES_PROFILE") or "default"),
        "recent_runs": recent,
    }, indent=2, ensure_ascii=False)


# ── plugin entry point ──────────────────────────────────────────────────────

def _register_host_persona(ctx) -> Optional[str]:
    """Pin a persona to this profile's OWN sessions via `persona: <name>` in its config.yaml."""
    try:
        profile = ctx.profile_name or "default"
    except Exception:
        profile = "default"
    name = profile_persona(profile)
    persona = load_persona(name)
    if not persona:
        return None
    registrar = getattr(ctx, "register_system_prompt_section", None)
    if registrar is None:
        logger.warning("delegation-router: this build cannot pin a persona section; router still active")
        return None

    def _section(session_info=None):  # noqa: ANN001 - contract takes a read-only mapping
        fresh = load_persona(name) or persona
        return fresh["charter"][:3900]

    try:
        registrar(f"personas.{persona['name']}", _section, position="after_memory", max_chars=4000)
        logger.info("delegation-router: persona %s pinned to profile %s", persona["name"], profile)
        return persona["name"]
    except Exception as exc:
        logger.warning("delegation-router: could not pin persona %s (%s)", name, exc)
        return None


def register(ctx) -> None:
    routing = load_routing()
    if not routing:
        logger.warning("delegation-router: no usable routing.yaml; plugin inert (stock delegate_task)")
        return
    # This fork's core asks the registry who owns delegate_task, so the handler registered
    # below is the dispatch entry point; only patch on a build without that hook.
    core_hook = _core_supports_delegate_override()
    patched = True if core_hook else _install_patch()
    pinned = _register_host_persona(ctx)
    ctx.register_tool(
        name="delegate_task",
        toolset="delegation",
        schema=_build_schema(),
        handler=_registry_dispatch,
        description="Delegate work: routes into the specialist profile pool when a bucket/profile resolves.",
        emoji="🔀",
        override=True,
    )
    ctx.register_tool(
        name="delegation_router_status",
        toolset="delegation",
        schema={
            "name": "delegation_router_status",
            "description": ("Show the delegation-router pool (bucket -> specialist profiles, with live "
                            "session leases) and the most recent routed runs."),
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "description": "How many recent runs to list (default 10)."}},
                "required": []},
        },
        handler=lambda args, **kw: delegation_router_status(limit=args.get("limit") or 10),
        description="Inspect the delegation-router pool and recent routed runs",
        emoji="🧭",
    )
    logger.info(
        "delegation-router loaded (dispatch=%s, %d buckets, %d personas, host persona=%s)",
        "core-hook" if core_hook else ("monkey-patch" if patched else "none"), len(bucket_names(routing)),
        len([p for p in _personas_dir().iterdir() if (p / "PERSONA.md").is_file()]) if _personas_dir().is_dir() else 0,
        pinned or "none",
    )
