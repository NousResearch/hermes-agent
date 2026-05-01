"""Curator — background skill maintenance orchestrator.

The curator is an auxiliary-model task that periodically reviews agent-created
skills and maintains the collection. It runs inactivity-triggered (no cron
daemon): when the agent is idle and the last curator run was longer than
``interval_hours`` ago, ``maybe_run_curator()`` spawns a forked AIAgent to do
the review.

Responsibilities:
  - Auto-transition lifecycle states based on derived skill activity timestamps
  - Spawn a background review agent that can pin / archive / consolidate /
    patch agent-created skills via skill_manage
  - Persist curator state (last_run_at, paused, etc.) in .curator_state

Strict invariants:
  - Only touches agent-created skills (see tools/skill_usage.is_agent_created)
  - Never auto-deletes — only archives. Archive is recoverable.
  - Pinned skills bypass all auto-transitions
  - Uses the auxiliary client; never touches the main session's prompt cache
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from hermes_constants import get_hermes_home
from tools import skill_usage

logger = logging.getLogger(__name__)


DEFAULT_INTERVAL_HOURS = 24 * 7  # 7 days
DEFAULT_MIN_IDLE_HOURS = 2
DEFAULT_STALE_AFTER_DAYS = 30
DEFAULT_ARCHIVE_AFTER_DAYS = 90


# ---------------------------------------------------------------------------
# .curator_state — persistent scheduler + status
# ---------------------------------------------------------------------------

def _state_file() -> Path:
    return get_hermes_home() / "skills" / ".curator_state"


def _default_state() -> Dict[str, Any]:
    return {
        "last_run_at": None,
        "last_run_duration_seconds": None,
        "last_run_summary": None,
        "last_report_path": None,
        "paused": False,
        "run_count": 0,
    }


def load_state() -> Dict[str, Any]:
    path = _state_file()
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            base = _default_state()
            base.update({k: v for k, v in data.items() if k in base or k.startswith("_")})
            return base
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read curator state: %s", e)
    return _default_state()


def save_state(data: Dict[str, Any]) -> None:
    path = _state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".curator_state_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to save curator state: %s", e, exc_info=True)


def set_paused(paused: bool) -> None:
    state = load_state()
    state["paused"] = bool(paused)
    save_state(state)


def is_paused() -> bool:
    return bool(load_state().get("paused"))


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    """Read curator.* config from ~/.hermes/config.yaml. Tolerates missing file."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as e:
        logger.debug("Failed to load config for curator: %s", e)
        return {}
    if not isinstance(cfg, dict):
        return {}
    cur = cfg.get("curator") or {}
    if not isinstance(cur, dict):
        return {}
    return cur


def is_enabled() -> bool:
    """Default ON when no config says otherwise."""
    cfg = _load_config()
    return bool(cfg.get("enabled", True))


def get_interval_hours() -> int:
    cfg = _load_config()
    try:
        return int(cfg.get("interval_hours", DEFAULT_INTERVAL_HOURS))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_HOURS


def get_min_idle_hours() -> float:
    cfg = _load_config()
    try:
        return float(cfg.get("min_idle_hours", DEFAULT_MIN_IDLE_HOURS))
    except (TypeError, ValueError):
        return DEFAULT_MIN_IDLE_HOURS


def get_stale_after_days() -> int:
    cfg = _load_config()
    try:
        return int(cfg.get("stale_after_days", DEFAULT_STALE_AFTER_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_STALE_AFTER_DAYS


def get_archive_after_days() -> int:
    cfg = _load_config()
    try:
        return int(cfg.get("archive_after_days", DEFAULT_ARCHIVE_AFTER_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_ARCHIVE_AFTER_DAYS


# ---------------------------------------------------------------------------
# Idle / interval check
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def should_run_now(now: Optional[datetime] = None) -> bool:
    """Return True if the curator should run immediately.

    Gates:
      - curator.enabled == True
      - not paused
      - last_run_at present AND older than interval_hours

    First-run behavior: when there is no ``last_run_at`` (fresh install, or
    install that predates the curator), we DO NOT run immediately. The
    curator is designed to run after at least ``interval_hours`` (7 days by
    default) of skill activity, not on the first background tick after
    ``hermes update``. On first observation we seed ``last_run_at`` to "now"
    and defer the first real pass by one full interval. Users who want to
    run it sooner can always invoke ``hermes curator run`` (with or without
    ``--dry-run``) explicitly — that path bypasses this gate.

    The idle check (min_idle_hours) is applied at the call site where we know
    whether an agent is actively running — here we only enforce the static
    gates.
    """
    if not is_enabled():
        return False
    if is_paused():
        return False

    state = load_state()
    last = _parse_iso(state.get("last_run_at"))
    if last is None:
        # Never run before. Seed state so we wait a full interval before the
        # first real pass. Report-only; do not auto-mutate the library the
        # very first time a gateway ticks after an update.
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            state["last_run_at"] = now.isoformat()
            state["last_run_summary"] = (
                "deferred first run — curator seeded, will run after one "
                "interval; use `hermes curator run --dry-run` to preview now"
            )
            save_state(state)
        except Exception as e:  # pragma: no cover — best-effort persistence
            logger.debug("Failed to seed curator last_run_at: %s", e)
        return False

    if now is None:
        now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    interval = timedelta(hours=get_interval_hours())
    return (now - last) >= interval


# ---------------------------------------------------------------------------
# Automatic state transitions (pure function, no LLM)
# ---------------------------------------------------------------------------

def apply_automatic_transitions(now: Optional[datetime] = None) -> Dict[str, int]:
    """Walk every agent-created skill and move active/stale/archived based on
    the latest real activity timestamp. Pinned skills are never touched.
    Returns a counter dict describing what changed."""
    from tools import skill_usage as _u

    if now is None:
        now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(days=get_stale_after_days())
    archive_cutoff = now - timedelta(days=get_archive_after_days())

    counts = {"marked_stale": 0, "archived": 0, "reactivated": 0, "checked": 0}

    for row in _u.agent_created_report():
        counts["checked"] += 1
        name = row["name"]
        if row.get("pinned"):
            continue

        last_activity = _parse_iso(row.get("last_activity_at"))
        # If never active, treat created_at as the anchor so new skills don't
        # immediately archive themselves.
        anchor = last_activity or _parse_iso(row.get("created_at")) or now
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)

        current = row.get("state", _u.STATE_ACTIVE)

        if anchor <= archive_cutoff and current != _u.STATE_ARCHIVED:
            ok, _msg = _u.archive_skill(name)
            if ok:
                counts["archived"] += 1
        elif anchor <= stale_cutoff and current == _u.STATE_ACTIVE:
            _u.set_state(name, _u.STATE_STALE)
            counts["marked_stale"] += 1
        elif anchor > stale_cutoff and current == _u.STATE_STALE:
            # Skill got used again after being marked stale — reactivate.
            _u.set_state(name, _u.STATE_ACTIVE)
            counts["reactivated"] += 1

    return counts


# ---------------------------------------------------------------------------
# Review prompt for the forked agent
# ---------------------------------------------------------------------------

CURATOR_DRY_RUN_BANNER = (
    "═══════════════════════════════════════════════════════════════\n"
    "DRY-RUN — REPORT ONLY. DO NOT MUTATE THE SKILL LIBRARY.\n"
    "═══════════════════════════════════════════════════════════════\n"
    "\n"
    "This is a PREVIEW pass. Follow every instruction below EXCEPT:\n"
    "\n"
    "  • DO NOT call skill_manage with action=patch, create, delete, "
    "write_file, or remove_file.\n"
    "  • DO NOT call terminal to mv skill directories into .archive/.\n"
    "  • DO NOT call terminal to mv, cp, rm, or rewrite any file under "
    "~/.hermes/skills/.\n"
    "  • skills_list and skill_view are FINE — read as much as you need.\n"
    "\n"
    "Your output IS the deliverable. Produce the exact same "
    "human-readable summary and structured YAML block you would "
    "produce on a live run — but describe the actions you WOULD take, "
    "not actions you took. A downstream reviewer will read the report "
    "and decide whether to approve a live run with "
    "`hermes curator run` (no flag).\n"
    "\n"
    "If you accidentally take a mutating action, say so explicitly in "
    "the summary so the reviewer can revert it.\n"
    "═══════════════════════════════════════════════════════════════"
)


CURATOR_REVIEW_PROMPT = (
    "You are running as Hermes' background skill CURATOR. This is an "
    "UMBRELLA-BUILDING consolidation pass, not a passive audit and not a "
    "duplicate-finder.\n\n"
    "The goal of the skill collection is a LIBRARY OF CLASS-LEVEL "
    "INSTRUCTIONS AND EXPERIENTIAL KNOWLEDGE. A collection of hundreds of "
    "narrow skills where each one captures one session's specific bug is "
    "a FAILURE of the library — not a feature. An agent searching skills "
    "matches on descriptions, not on exact names; one broad umbrella "
    "skill with labeled subsections beats five narrow siblings for "
    "discoverability, not the other way around.\n\n"
    "The right target shape is CLASS-LEVEL skills with rich SKILL.md "
    "bodies + `references/`, `templates/`, and `scripts/` subfiles for "
    "session-specific detail — not one-session-one-skill micro-entries.\n\n"
    "Hard rules — do not violate:\n"
    "1. DO NOT touch bundled or hub-installed skills. The candidate list "
    "below is already filtered to agent-created skills only.\n"
    "2. DO NOT delete any skill. Archiving (moving the skill's directory "
    "into ~/.hermes/skills/.archive/) is the maximum destructive action. "
    "Archives are recoverable; deletion is not.\n"
    "3. DO NOT touch skills shown as pinned=yes. Skip them entirely.\n"
    "4. DO NOT use usage counters as a reason to skip consolidation. The "
    "counters are new and often mostly zero. Judge overlap on CONTENT, "
    "not on use_count. 'use=0' is not evidence a skill is valuable; it's "
    "absence of evidence either way.\n"
    "5. DO NOT reject consolidation on the grounds that 'each skill has "
    "a distinct trigger'. Pairwise distinctness is the wrong bar. The "
    "right bar is: 'would a human maintainer write this as N separate "
    "skills, or as one skill with N labeled subsections?' When the "
    "answer is the latter, merge.\n\n"
    "How to work — not optional:\n"
    "1. Scan the full candidate list. Identify PREFIX CLUSTERS (skills "
    "sharing a first word or domain keyword). Examples you are likely "
    "to find: hermes-config-*, hermes-dashboard-*, gateway-*, codex-*, "
    "ollama-*, anthropic-*, gemini-*, mcp-*, salvage-*, pr-*, "
    "competitor-*, python-*, security-*, etc. Expect 10-25 clusters.\n"
    "2. For each cluster with 2+ members, do NOT ask 'are these pairs "
    "overlapping?' — ask 'what is the UMBRELLA CLASS these skills all "
    "serve? Would a maintainer name that class and write one skill for "
    "it?' If yes, pick (or create) the umbrella and absorb the siblings "
    "into it.\n"
    "3. Three ways to consolidate — use the right one per cluster:\n"
    "   a. MERGE INTO EXISTING UMBRELLA — one skill in the cluster is "
    "already broad enough to be the umbrella (example: `pr-triage-"
    "salvage` for the PR review cluster). Patch it to add a labeled "
    "section for each sibling's unique insight, then archive the "
    "siblings.\n"
    "   b. CREATE A NEW UMBRELLA SKILL.md — no existing member is broad "
    "enough. Use skill_manage action=create to write a new class-level "
    "skill whose SKILL.md covers the shared workflow and has short "
    "labeled subsections. Archive the now-absorbed narrow siblings.\n"
    "   c. DEMOTE TO REFERENCES/TEMPLATES/SCRIPTS — a sibling has "
    "narrow-but-valuable session-specific content. Move it into the "
    "umbrella's appropriate support directory:\n"
    "      • `references/<topic>.md` for session-specific detail OR "
    "condensed knowledge banks (quoted research, API docs excerpts, "
    "domain notes, provider quirks, reproduction recipes)\n"
    "      • `templates/<name>.<ext>` for starter files meant to be "
    "copied and modified\n"
    "      • `scripts/<name>.<ext>` for statically re-runnable actions "
    "(verification scripts, fixture generators, probes)\n"
    "      Then archive the old sibling. Use `terminal` with `mkdir -p "
    "~/.hermes/skills/<umbrella>/references/ && mv ... <umbrella>/"
    "references/<topic>.md` (or templates/ / scripts/).\n"
    "4. Also flag skills whose NAME is too narrow (contains a PR number, "
    "a feature codename, a specific error string, an 'audit' / "
    "'diagnosis' / 'salvage' session artifact). These almost always "
    "belong as a subsection or support file under a class-level umbrella.\n"
    "5. Iterate. After one consolidation round, scan the remaining set "
    "and look for the NEXT umbrella opportunity. Don't stop after 3 "
    "merges.\n\n"
    "Your toolset:\n"
    "  - skills_list, skill_view        — read the current landscape\n"
    "  - skill_manage action=patch      — add sections to the umbrella\n"
    "  - skill_manage action=create     — create a new umbrella SKILL.md\n"
    "  - skill_manage action=write_file — add a references/, templates/, "
    "or scripts/ file under an existing skill (the skill must already "
    "exist)\n"
    "  - terminal                       — mv a sibling into the archive "
    "OR move its content into a support subfile\n\n"
    "'keep' is a legitimate decision ONLY when the skill is already a "
    "class-level umbrella and none of the proposed merges would improve "
    "discoverability. 'This is narrow but distinct from its siblings' "
    "is NOT a reason to keep — it's a reason to move it under an "
    "umbrella as a subsection or support file.\n\n"
    "Expected output: real umbrella-ification. Process every obvious "
    "cluster. If you end the pass with fewer than 10 archives, you "
    "stopped too early — go back and look at the clusters you left "
    "alone.\n\n"
    "When done, write a human summary AND a structured machine-readable "
    "block so downstream tooling can distinguish consolidation from "
    "pruning. Format EXACTLY:\n\n"
    "## Structured summary (required)\n"
    "```yaml\n"
    "consolidations:\n"
    "  - from: <old-skill-name>\n"
    "    into: <umbrella-skill-name>\n"
    "    reason: <one short sentence — why merged, not just 'similar'>\n"
    "prunings:\n"
    "  - name: <skill-name>\n"
    "    reason: <one short sentence — why archived with no merge target>\n"
    "```\n\n"
    "Every skill you moved to .archive/ MUST appear in exactly one of the "
    "two lists. If you consolidated X into umbrella Y (patched Y, wrote "
    "a references file to Y, or created Y with X's content absorbed), X "
    "goes under `consolidations` with `into: Y`. If you archived X with "
    "no absorption — truly stale, irrelevant, or obsolete — X goes under "
    "`prunings`. Leave a list empty (`consolidations: []`) if none. Do "
    "not omit the block. The block comes AFTER your human-readable "
    "summary of clusters processed, patches made, and decisions left alone."
)


# ---------------------------------------------------------------------------
# Per-run reports — {YYYYMMDD-HHMMSS}/run.json + REPORT.md under logs/curator/
# ---------------------------------------------------------------------------

def _reports_root() -> Path:
    """Directory where curator run reports are written.

    Lives under the profile-aware logs dir (``~/.hermes/logs/curator/``)
    alongside ``agent.log`` and ``gateway.log`` so it's found by anyone
    looking for operational telemetry, not mixed in with the user's
    authored skill data in ``~/.hermes/skills/``.

    ``ensure_hermes_home()`` pre-creates this dir on every CLI launch and
    the v22→v23 migration backfills it for existing profiles, but we
    still mkdir here as a belt-and-suspenders so the curator works even
    from an odd entry path (e.g. gateway-only install, bare library use)
    that bypasses both.
    """
    root = get_hermes_home() / "logs" / "curator"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("Curator reports dir create failed: %s", e)
    return root


def _classify_removed_skills(
    removed: List[str],
    added: List[str],
    after_names: Set[str],
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split ``removed`` into consolidated vs pruned.

    A removed skill is "consolidated" when the curator absorbed its content
    into another skill (an umbrella) during this run — the content still
    lives, just under a different name. A removed skill is "pruned" when the
    curator archived it for staleness/irrelevance without preserving its
    content elsewhere.

    Heuristic: scan this run's ``skill_manage`` tool calls and look for
    ``write_file``/``patch``/``create``/``edit`` actions whose target skill
    (the ``name`` argument) is NOT the removed skill and whose
    ``file_path`` / ``file_content`` / ``content`` arguments reference the
    removed skill's name. That's the textbook "absorbed into umbrella"
    signal. Ties are broken by first-match (earliest tool call wins).

    Returns ``{"consolidated": [{"name", "into", "evidence"}, ...],
               "pruned":       [{"name"}, ...]}``.
    """
    consolidated: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []

    # Pre-parse tool calls: we only care about skill_manage.
    parsed_calls: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        if tc.get("name") != "skill_manage":
            continue
        raw = tc.get("arguments") or ""
        # Arguments can be a JSON string (standard) or a dict (defensive).
        args: Dict[str, Any] = {}
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str):
            try:
                args = json.loads(raw)
            except Exception:
                # Truncated or malformed — fall back to substring match on
                # the raw string so we still catch the common case.
                args = {"_raw": raw}
        if not isinstance(args, dict):
            continue
        parsed_calls.append(args)

    # Build a set of "destination" skill names: anything still present after
    # the run plus anything newly added this run. A removed skill being
    # referenced from one of these is the consolidation signal.
    destinations = set(after_names) | set(added or [])

    for name in removed:
        if not name:
            continue
        into: Optional[str] = None
        evidence: Optional[str] = None

        # Normalise name variants we'll search for in path/content strings.
        needles = {name, name.replace("-", "_"), name.replace("_", "-")}

        for args in parsed_calls:
            target = args.get("name")
            if not isinstance(target, str) or not target:
                continue
            # A call that operates on the removed skill itself isn't
            # consolidation evidence.
            if target == name:
                continue
            # The target must be a surviving or newly-created skill —
            # otherwise we're pointing to a skill that doesn't exist.
            if target not in destinations:
                continue

            # Look for the removed skill's name in file_path / content / raw.
            haystacks: List[str] = []
            for key in ("file_path", "file_content", "content", "new_string", "_raw"):
                v = args.get(key)
                if isinstance(v, str):
                    haystacks.append(v)
            hit = False
            for hay in haystacks:
                for needle in needles:
                    if needle and needle in hay:
                        hit = True
                        evidence = (
                            f"skill_manage action={args.get('action', '?')} "
                            f"on '{target}' referenced '{name}' "
                            f"in {hay[:80]}"
                        )
                        break
                if hit:
                    break
            if hit:
                into = target
                break

        if into:
            consolidated.append({"name": name, "into": into, "evidence": evidence})
        else:
            pruned.append({"name": name})

    return {"consolidated": consolidated, "pruned": pruned}


def _parse_structured_summary(
    llm_final: str,
) -> Dict[str, List[Dict[str, str]]]:
    """Extract the structured YAML block from the curator's final response.

    The curator prompt requires a fenced ```yaml block under
    ``## Structured summary (required)`` with ``consolidations:`` and
    ``prunings:`` lists. This parses it tolerantly:

    - Missing block → returns empty lists (we'll fall back to heuristic).
    - Malformed YAML → returns empty lists and we rely on heuristic.
    - Partial block (e.g. only consolidations) → returns what we could parse.

    Returns ``{"consolidations": [{"from", "into", "reason"}, ...],
               "prunings":       [{"name", "reason"}, ...]}``.
    """
    empty = {"consolidations": [], "prunings": []}
    if not llm_final or not isinstance(llm_final, str):
        return empty

    # Find the YAML fenced block. We look for ```yaml ... ``` specifically
    # rather than any fenced block so we don't accidentally pick up a code
    # sample the model quoted elsewhere.
    import re
    match = re.search(
        r"```ya?ml\s*\n(.*?)\n```",
        llm_final,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return empty

    body = match.group(1)

    # Prefer PyYAML when available — every hermes install already has it
    # (config.yaml loader). Fall back to a hand parser for paranoia.
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(body)
    except Exception:
        return empty

    if not isinstance(data, dict):
        return empty

    out: Dict[str, List[Dict[str, str]]] = {"consolidations": [], "prunings": []}
    cons_raw = data.get("consolidations") or []
    prun_raw = data.get("prunings") or []

    if isinstance(cons_raw, list):
        for entry in cons_raw:
            if not isinstance(entry, dict):
                continue
            frm = entry.get("from")
            into = entry.get("into")
            if not (isinstance(frm, str) and frm.strip()
                    and isinstance(into, str) and into.strip()):
                continue
            reason = entry.get("reason")
            out["consolidations"].append({
                "from": frm.strip(),
                "into": into.strip(),
                "reason": (reason or "").strip() if isinstance(reason, str) else "",
            })

    if isinstance(prun_raw, list):
        for entry in prun_raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not (isinstance(name, str) and name.strip()):
                continue
            reason = entry.get("reason")
            out["prunings"].append({
                "name": name.strip(),
                "reason": (reason or "").strip() if isinstance(reason, str) else "",
            })

    return out


def _reconcile_classification(
    removed: List[str],
    heuristic: Dict[str, List[Dict[str, Any]]],
    model_block: Dict[str, List[Dict[str, str]]],
    destinations: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge heuristic (tool-call evidence) with the model's structured block.

    Rules:
    - Model-declared consolidation wins when its ``into`` target exists
      in ``destinations`` (survived or newly-created). This gives the
      model authority over intent + rationale.
    - Model-declared consolidation whose ``into`` target does NOT exist is
      downgraded: the model hallucinated an umbrella. We prefer the
      heuristic's finding for that skill, or fall back to pruned.
    - Heuristic-only finding (model didn't mention it, tool calls confirm)
      is preserved as a consolidation, marked ``source="tool-call audit"``.
    - Model-declared pruning is accepted unless the heuristic has
      tool-call evidence that contradicts it (rare — the heuristic would
      have flagged consolidation). In that case we log both.

    Every removed skill is placed in exactly one bucket.
    """
    heur_cons = {e["name"]: e for e in heuristic.get("consolidated", [])}
    heur_pruned = {e["name"] for e in heuristic.get("pruned", [])}

    model_cons = {e["from"]: e for e in model_block.get("consolidations", [])}
    model_pruned = {e["name"]: e for e in model_block.get("prunings", [])}

    consolidated: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []

    for name in removed:
        mc = model_cons.get(name)
        mp = model_pruned.get(name)
        hc = heur_cons.get(name)

        # Model says consolidated — trust it if the destination is real.
        if mc and mc.get("into") in destinations:
            entry: Dict[str, Any] = {
                "name": name,
                "into": mc["into"],
                "source": "model" + ("+audit" if hc else ""),
                "reason": mc.get("reason") or "",
            }
            if hc and hc.get("evidence"):
                entry["evidence"] = hc["evidence"]
            consolidated.append(entry)
            continue

        # Model says consolidated but the umbrella doesn't exist —
        # hallucination. Fall back to heuristic or prune.
        if mc and mc.get("into") not in destinations:
            if hc:
                consolidated.append({
                    "name": name,
                    "into": hc["into"],
                    "source": "tool-call audit (model named missing umbrella)",
                    "reason": "",
                    "evidence": hc.get("evidence", ""),
                    "model_claimed_into": mc["into"],
                })
            else:
                pruned.append({
                    "name": name,
                    "source": "fallback (model named missing umbrella, no tool-call evidence)",
                    "reason": "",
                })
            continue

        # Heuristic found consolidation the model didn't mention.
        if hc:
            consolidated.append({
                "name": name,
                "into": hc["into"],
                "source": "tool-call audit (model omitted from structured block)",
                "reason": "",
                "evidence": hc.get("evidence", ""),
            })
            continue

        # Model says pruned (or no mention + no heuristic evidence).
        reason = mp.get("reason", "") if mp else ""
        pruned.append({
            "name": name,
            "source": "model" if mp else "no-evidence fallback",
            "reason": reason,
        })

    return {"consolidated": consolidated, "pruned": pruned}


def _write_run_report(
    *,
    started_at: datetime,
    elapsed_seconds: float,
    auto_counts: Dict[str, int],
    auto_summary: str,
    before_report: List[Dict[str, Any]],
    before_names: Set[str],
    after_report: List[Dict[str, Any]],
    llm_meta: Dict[str, Any],
) -> Optional[Path]:
    """Write run.json + REPORT.md under logs/curator/{YYYYMMDD-HHMMSS}/.

    Returns the report directory path on success, None if the write
    couldn't happen (caller logs and continues — reporting is best-effort).
    """
    root = _reports_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug("Curator report dir create failed: %s", e)
        return None

    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    # If we crash-reran within the same second, append a disambiguator
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}-{suffix}"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except Exception as e:
        logger.debug("Curator run dir create failed: %s", e)
        return None

    # Diff before/after
    after_by_name = {r.get("name"): r for r in after_report if isinstance(r, dict)}
    after_names = set(after_by_name.keys())
    removed = sorted(before_names - after_names)   # archived during this run
    added = sorted(after_names - before_names)     # new skills this run
    before_by_name = {r.get("name"): r for r in before_report if isinstance(r, dict)}

    # State transitions between the two snapshots (e.g. active -> stale)
    transitions: List[Dict[str, str]] = []
    for name in sorted(after_names & before_names):
        s_before = (before_by_name.get(name) or {}).get("state")
        s_after = (after_by_name.get(name) or {}).get("state")
        if s_before and s_after and s_before != s_after:
            transitions.append({"name": name, "from": s_before, "to": s_after})

    # Classify LLM tool calls
    tc_counts: Dict[str, int] = {}
    for tc in llm_meta.get("tool_calls", []) or []:
        name = tc.get("name", "unknown")
        tc_counts[name] = tc_counts.get(name, 0) + 1

    # Split "removed" into consolidated (absorbed into umbrella) vs pruned
    # (archived for staleness, content not preserved elsewhere). The old
    # "Skills archived" section lumped both together, which misled users
    # into thinking consolidated skills had been pruned.
    #
    # Classification strategy:
    # 1. Parse the curator's structured YAML block from its final response.
    #    The curator is now prompted to emit consolidations/prunings lists
    #    with short rationale. The model has intent visibility the tool
    #    calls don't.
    # 2. Run the tool-call heuristic as a ground-truth audit.
    # 3. Reconcile: model gets authority over intent + rationale, heuristic
    #    catches hallucination (umbrella doesn't exist) and omission
    #    (model forgot to list an actual consolidation).
    heuristic = _classify_removed_skills(
        removed=removed,
        added=added,
        after_names=after_names,
        tool_calls=llm_meta.get("tool_calls", []) or [],
    )
    model_block = _parse_structured_summary(llm_meta.get("final", "") or "")
    destinations = set(after_names) | set(added or [])
    classification = _reconcile_classification(
        removed=removed,
        heuristic=heuristic,
        model_block=model_block,
        destinations=destinations,
    )
    consolidated = classification["consolidated"]
    pruned = classification["pruned"]

    # Rewrite cron job skill references. When the curator consolidates
    # skill X into umbrella Y, any cron job that lists X fails to load
    # it at run time — the scheduler skips it and the job runs without
    # the instructions it was scheduled to follow. Rewriting the
    # references in-place keeps scheduled jobs working across
    # consolidation passes. Best-effort: never let a cron-module issue
    # break the curator.
    cron_rewrites: Dict[str, Any] = {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}
    try:
        consolidated_map = {
            e["name"]: e["into"]
            for e in consolidated
            if isinstance(e, dict) and e.get("name") and e.get("into")
        }
        pruned_names = [
            e["name"] for e in pruned
            if isinstance(e, dict) and e.get("name")
        ]
        if consolidated_map or pruned_names:
            from cron.jobs import rewrite_skill_refs as _rewrite_cron_refs
            cron_rewrites = _rewrite_cron_refs(
                consolidated=consolidated_map,
                pruned=pruned_names,
            )
    except Exception as e:
        logger.debug("Curator cron skill rewrite failed: %s", e, exc_info=True)
        cron_rewrites = {
            "rewrites": [],
            "jobs_updated": 0,
            "jobs_scanned": 0,
            "error": str(e),
        }

    payload = {
        "started_at": started_at.isoformat(),
        "duration_seconds": round(elapsed_seconds, 2),
        "model": llm_meta.get("model", ""),
        "provider": llm_meta.get("provider", ""),
        "auto_transitions": auto_counts,
        "counts": {
            "before": len(before_names),
            "after": len(after_names),
            "delta": len(after_names) - len(before_names),
            "archived_this_run": len(removed),
            "added_this_run": len(added),
            "consolidated_this_run": len(consolidated),
            "pruned_this_run": len(pruned),
            "state_transitions": len(transitions),
            "cron_jobs_rewritten": int(cron_rewrites.get("jobs_updated", 0)),
            "tool_calls_total": sum(tc_counts.values()),
        },
        "tool_call_counts": tc_counts,
        "archived": removed,
        "consolidated": consolidated,
        "pruned": pruned,
        "pruned_names": [p["name"] for p in pruned],
        "added": added,
        "state_transitions": transitions,
        "cron_rewrites": cron_rewrites,
        "llm_final": llm_meta.get("final", ""),
        "llm_summary": llm_meta.get("summary", ""),
        "llm_error": llm_meta.get("error"),
        "tool_calls": llm_meta.get("tool_calls", []),
    }

    # run.json — machine-readable, full fidelity
    try:
        (run_dir / "run.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("Curator run.json write failed: %s", e)

    # REPORT.md — human-readable
    try:
        md = _render_report_markdown(payload)
        (run_dir / "REPORT.md").write_text(md, encoding="utf-8")
    except Exception as e:
        logger.debug("Curator REPORT.md write failed: %s", e)

    # cron_rewrites.json — only when at least one job was touched, to
    # keep run dirs uncluttered for the common no-op case.
    try:
        if int(cron_rewrites.get("jobs_updated", 0)) > 0:
            (run_dir / "cron_rewrites.json").write_text(
                json.dumps(cron_rewrites, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except Exception as e:
        logger.debug("Curator cron_rewrites.json write failed: %s", e)

    return run_dir


def _render_report_markdown(p: Dict[str, Any]) -> str:
    """Render the human-readable report."""
    lines: List[str] = []
    started = p.get("started_at", "")
    duration = p.get("duration_seconds", 0) or 0
    mins, secs = divmod(int(duration), 60)
    dur_label = f"{mins}m {secs}s" if mins else f"{secs}s"

    lines.append(f"# Curator run — {started}\n")
    model = p.get("model") or "(not resolved)"
    prov = p.get("provider") or "(not resolved)"
    counts = p.get("counts") or {}
    lines.append(
        f"Model: `{model}` via `{prov}`  ·  Duration: {dur_label}  ·  "
        f"Agent-created skills: {counts.get('before', 0)} → {counts.get('after', 0)} "
        f"({counts.get('delta', 0):+d})\n"
    )

    error = p.get("llm_error")
    if error:
        lines.append(f"> ⚠ LLM pass error: `{error}`\n")

    # Auto-transitions (pure, no LLM)
    auto = p.get("auto_transitions") or {}
    lines.append("## Auto-transitions (pure, no LLM)\n")
    lines.append(f"- checked: {auto.get('checked', 0)}")
    lines.append(f"- marked stale: {auto.get('marked_stale', 0)}")
    lines.append(f"- archived (no LLM, pure time-based staleness): {auto.get('archived', 0)}")
    lines.append(f"- reactivated: {auto.get('reactivated', 0)}")
    lines.append("")

    # LLM pass numbers
    tc_counts = p.get("tool_call_counts") or {}
    lines.append("## LLM consolidation pass\n")
    lines.append(f"- tool calls: **{counts.get('tool_calls_total', 0)}** "
                 f"(by name: {', '.join(f'{k}={v}' for k, v in sorted(tc_counts.items())) or 'none'})")
    lines.append(f"- consolidated into umbrellas: **{counts.get('consolidated_this_run', 0)}**")
    lines.append(f"- pruned (archived for staleness): **{counts.get('pruned_this_run', 0)}**")
    lines.append(f"- new skills this run: **{counts.get('added_this_run', 0)}**")
    lines.append(f"- state transitions (active ↔ stale ↔ archived): "
                 f"**{counts.get('state_transitions', 0)}**")
    lines.append("")

    # Consolidated list — content absorbed into an umbrella. The directory
    # on disk still lives under ~/.hermes/skills/.archive/ (every removal is
    # recoverable by design), but the "live" content for these skills
    # continues to exist inside the destination umbrella.
    consolidated = p.get("consolidated") or []
    if consolidated:
        lines.append(f"### Consolidated into umbrella skills ({len(consolidated)})\n")
        lines.append(
            "_These skills were **absorbed into another skill** during this run — "
            "their content still lives, just under a different name. "
            "The original directory was moved to `~/.hermes/skills/.archive/` for "
            "safety and can be restored via `hermes curator restore <name>` if the "
            "consolidation was wrong._\n"
        )
        SHOW = 50
        for entry in consolidated[:SHOW]:
            name = entry.get("name", "?")
            into = entry.get("into", "?")
            reason = (entry.get("reason") or "").strip()
            source = entry.get("source", "")
            line = f"- `{name}` → merged into `{into}`"
            if reason:
                line += f" — {reason}"
            if source and source.startswith("tool-call audit"):
                # The model didn't enumerate this one — surface that to the
                # user so they know why the row has no rationale.
                line += f"  _(detected via {source})_"
            lines.append(line)
            if entry.get("model_claimed_into"):
                lines.append(
                    f"  ⚠ The curator's summary named `{entry['model_claimed_into']}` "
                    "as the umbrella but that skill doesn't exist post-run; "
                    "showing the tool-call audit's finding instead."
                )
        if len(consolidated) > SHOW:
            lines.append(f"- … and {len(consolidated) - SHOW} more (see `run.json`)")
        lines.append("")

    # Pruned list — archived without consolidation. These are the
    # "stale skill pruned" cases the UI should mark clearly.
    pruned = p.get("pruned") or []
    if pruned:
        lines.append(f"### Pruned — archived for staleness ({len(pruned)})\n")
        lines.append(
            "_These skills were archived without being merged into an umbrella "
            "(e.g. stale, unused, or judged irrelevant). "
            "Directories live under `~/.hermes/skills/.archive/`. "
            "Restore any via `hermes curator restore <name>`._\n"
        )
        SHOW = 50
        for entry in pruned[:SHOW]:
            # Entries are dicts with {name, source, reason} when written via
            # the reconciler, or bare strings when an older format slipped
            # through. Handle both.
            if isinstance(entry, dict):
                name = entry.get("name", "?")
                reason = (entry.get("reason") or "").strip()
                line = f"- `{name}`"
                if reason:
                    line += f" — {reason}"
                lines.append(line)
            else:
                lines.append(f"- `{entry}`")
        if len(pruned) > SHOW:
            lines.append(f"- … and {len(pruned) - SHOW} more (see `run.json`)")
        lines.append("")

    # Added list
    added = p.get("added") or []
    if added:
        lines.append(f"### New skills this run ({len(added)})\n")
        lines.append("_Usually these are new class-level umbrellas created via `skill_manage action=create`._\n")
        for n in added:
            lines.append(f"- `{n}`")
        lines.append("")

    # State transitions
    trans = p.get("state_transitions") or []
    if trans:
        lines.append(f"### State transitions ({len(trans)})\n")
        for t in trans:
            lines.append(f"- `{t.get('name')}`: {t.get('from')} → {t.get('to')}")
        lines.append("")

    # Cron job rewrites — show which scheduled jobs had their skill
    # references updated so users can audit that the auto-rewrite did
    # the right thing. Only present when at least one job changed.
    cron_rw = p.get("cron_rewrites") or {}
    cron_rewrites_list = cron_rw.get("rewrites") or []
    if cron_rewrites_list:
        lines.append(f"### Cron job skill references rewritten ({len(cron_rewrites_list)})\n")
        lines.append(
            "_Cron jobs that referenced a consolidated or pruned skill were "
            "updated in-place so they keep loading the right instructions "
            "on their next run. See `cron_rewrites.json` for the full record._\n"
        )
        SHOW = 25
        for entry in cron_rewrites_list[:SHOW]:
            job_name = entry.get("job_name") or entry.get("job_id") or "?"
            before = entry.get("before") or []
            after = entry.get("after") or []
            mapped = entry.get("mapped") or {}
            dropped = entry.get("dropped") or []
            lines.append(
                f"- `{job_name}`: `{', '.join(before)}` → `{', '.join(after) or '(none)'}`"
            )
            for old, new in mapped.items():
                lines.append(f"    - `{old}` → `{new}` (consolidated)")
            for name in dropped:
                lines.append(f"    - `{name}` dropped (pruned)")
        if len(cron_rewrites_list) > SHOW:
            lines.append(
                f"- … and {len(cron_rewrites_list) - SHOW} more "
                "(see `cron_rewrites.json`)"
            )
        lines.append("")

    # Full LLM final response
    final = (p.get("llm_final") or "").strip()
    if final:
        lines.append("## LLM final summary\n")
        lines.append(final)
        lines.append("")
    elif not error:
        llm_sum = p.get("llm_summary") or ""
        if llm_sum:
            lines.append("## LLM summary\n")
            lines.append(llm_sum)
            lines.append("")

    # Recovery footer
    lines.append("## Recovery\n")
    lines.append("- Restore an archived skill: `hermes curator restore <name>`")
    lines.append("- All archives live under `~/.hermes/skills/.archive/` and are recoverable by `mv`")
    lines.append("- See `run.json` in this directory for the full machine-readable record.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator — spawn a forked AIAgent for the LLM review pass
# ---------------------------------------------------------------------------

def _render_candidate_list() -> str:
    """Human/agent-readable list of agent-created skills with usage stats."""
    rows = skill_usage.agent_created_report()
    if not rows:
        return "No agent-created skills to review."
    lines = [f"Agent-created skills ({len(rows)}):\n"]
    for r in rows:
        lines.append(
            f"- {r['name']}  "
            f"state={r['state']}  "
            f"pinned={'yes' if r.get('pinned') else 'no'}  "
            f"activity={r.get('activity_count', 0)}  "
            f"use={r.get('use_count', 0)}  "
            f"view={r.get('view_count', 0)}  "
            f"patches={r.get('patch_count', 0)}  "
            f"last_activity={r.get('last_activity_at') or 'never'}"
        )
    return "\n".join(lines)


def run_curator_review(
    on_summary: Optional[Callable[[str], None]] = None,
    synchronous: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a single curator review pass.

    Steps:
      1. Apply automatic state transitions (pure, no LLM).
      2. If there are agent-created skills, spawn a forked AIAgent that runs
         the LLM review prompt against the current candidate list.
      3. Update .curator_state with last_run_at and a one-line summary.
      4. Invoke *on_summary* with a user-visible description.

    If *synchronous* is True, the LLM review runs in the calling thread; the
    default is to spawn a daemon thread so the caller returns immediately.

    If *dry_run* is True, the automatic stale/archive transitions are SKIPPED
    and the LLM review pass is instructed to produce a report only — no
    skill_manage mutations, no terminal archive moves. The REPORT.md still
    gets written and ``state.last_report_path`` still records it so users
    can read what the curator WOULD have done.
    """
    start = datetime.now(timezone.utc)
    if dry_run:
        # Count candidates without mutating state.
        try:
            report = skill_usage.agent_created_report()
            counts = {
                "checked": len(report),
                "marked_stale": 0,
                "archived": 0,
                "reactivated": 0,
            }
        except Exception:
            counts = {"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0}
    else:
        # Pre-mutation snapshot — best-effort, never blocks the run. A
        # failed snapshot logs at debug and continues (the alternative is
        # that a transient disk issue silently disables curator forever,
        # which is worse). Users who want to require snapshots can disable
        # curator entirely until they can fix disk space.
        try:
            from agent import curator_backup
            snap = curator_backup.snapshot_skills(reason="pre-curator-run")
            if snap is not None and on_summary:
                try:
                    on_summary(f"curator: snapshot created ({snap.name})")
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Curator pre-run snapshot failed: %s", e, exc_info=True)
        counts = apply_automatic_transitions(now=start)

    auto_summary_parts = []
    if counts["marked_stale"]:
        auto_summary_parts.append(f"{counts['marked_stale']} marked stale")
    if counts["archived"]:
        auto_summary_parts.append(f"{counts['archived']} archived")
    if counts["reactivated"]:
        auto_summary_parts.append(f"{counts['reactivated']} reactivated")
    auto_summary = ", ".join(auto_summary_parts) if auto_summary_parts else "no changes"

    # Persist state before the LLM pass so a crash mid-review still records
    # the run and doesn't immediately re-trigger. In dry-run we do NOT bump
    # last_run_at or run_count — a preview shouldn't push the next scheduled
    # real pass out. We still record a summary so `hermes curator status`
    # shows that a preview ran.
    state = load_state()
    if not dry_run:
        state["last_run_at"] = start.isoformat()
        state["run_count"] = int(state.get("run_count", 0)) + 1
    prefix = "dry-run auto: " if dry_run else "auto: "
    state["last_run_summary"] = f"{prefix}{auto_summary}"
    save_state(state)

    def _llm_pass():
        nonlocal auto_summary
        # Snapshot skill state BEFORE the LLM pass so the report can diff.
        try:
            before_report = skill_usage.agent_created_report()
        except Exception:
            before_report = []
        before_names = {r.get("name") for r in before_report if isinstance(r, dict)}

        llm_meta: Dict[str, Any] = {}
        try:
            candidate_list = _render_candidate_list()
            if "No agent-created skills" in candidate_list:
                final_summary = f"{prefix}{auto_summary}; llm: skipped (no candidates)"
                llm_meta = {
                    "final": "",
                    "summary": "skipped (no candidates)",
                    "model": "",
                    "provider": "",
                    "tool_calls": [],
                    "error": None,
                }
            else:
                if dry_run:
                    prompt = (
                        f"{CURATOR_DRY_RUN_BANNER}\n\n"
                        f"{CURATOR_REVIEW_PROMPT}\n\n"
                        f"{candidate_list}"
                    )
                else:
                    prompt = f"{CURATOR_REVIEW_PROMPT}\n\n{candidate_list}"
                llm_meta = _run_llm_review(prompt)
                final_summary = (
                    f"{prefix}{auto_summary}; llm: {llm_meta.get('summary', 'no change')}"
                )
        except Exception as e:
            logger.debug("Curator LLM pass failed: %s", e, exc_info=True)
            final_summary = f"{prefix}{auto_summary}; llm: error ({e})"
            llm_meta = {
                "final": "",
                "summary": f"error ({e})",
                "model": "",
                "provider": "",
                "tool_calls": [],
                "error": str(e),
            }

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        state2 = load_state()
        state2["last_run_duration_seconds"] = elapsed
        state2["last_run_summary"] = final_summary

        # Write the per-run report. Runs in a best-effort try so a
        # reporting bug never breaks the curator itself. Report path is
        # recorded in state so `hermes curator status` can point at it.
        try:
            after_report = skill_usage.agent_created_report()
        except Exception:
            after_report = []
        try:
            report_path = _write_run_report(
                started_at=start,
                elapsed_seconds=elapsed,
                auto_counts=counts,
                auto_summary=auto_summary,
                before_report=before_report,
                before_names=before_names,
                after_report=after_report,
                llm_meta=llm_meta,
            )
            if report_path is not None:
                state2["last_report_path"] = str(report_path)
        except Exception as e:
            logger.debug("Curator report write failed: %s", e, exc_info=True)

        save_state(state2)

        if on_summary:
            try:
                on_summary(f"curator: {final_summary}")
            except Exception:
                pass

    if synchronous:
        _llm_pass()
    else:
        t = threading.Thread(target=_llm_pass, daemon=True, name="curator-review")
        t.start()

    return {
        "started_at": start.isoformat(),
        "auto_transitions": counts,
        "summary_so_far": auto_summary,
    }


def _resolve_review_model(cfg: Dict[str, Any]) -> tuple[str, str]:
    """Pick (provider, model) for the curator review fork.

    Curator is a regular auxiliary task slot — ``auxiliary.curator.{provider,model}``
    — so it participates in the canonical aux-model plumbing (``hermes model`` →
    auxiliary picker, the dashboard Models tab, ``auxiliary.curator.{timeout,
    base_url,api_key,extra_body}``). ``provider: "auto"`` with an empty model
    means "use the main chat model" — same default as every other aux task.

    Legacy fallback: users who configured ``curator.auxiliary.{provider,model}``
    under the previous one-off schema still work. Precedence:
      1. ``auxiliary.curator.{provider,model}`` when both are set non-auto
      2. Legacy ``curator.auxiliary.{provider,model}`` when both are set
      3. Main ``model.{provider,default/model}`` pair
    """
    _main = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    _main_provider = _main.get("provider") or "auto"
    _main_model = _main.get("default") or _main.get("model") or ""

    # 1. Canonical aux task slot
    _aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    _cur_task = _aux.get("curator", {}) if isinstance(_aux.get("curator"), dict) else {}
    _task_provider = (_cur_task.get("provider") or "").strip() or None
    _task_model = (_cur_task.get("model") or "").strip() or None
    if _task_provider and _task_provider != "auto" and _task_model:
        return _task_provider, _task_model

    # 2. Legacy curator.auxiliary.{provider,model} (deprecated, pre-unification)
    _cur = cfg.get("curator", {}) if isinstance(cfg.get("curator"), dict) else {}
    _legacy = _cur.get("auxiliary", {}) if isinstance(_cur.get("auxiliary"), dict) else {}
    _legacy_provider = _legacy.get("provider") or None
    _legacy_model = _legacy.get("model") or None
    if _legacy_provider and _legacy_model:
        logger.info(
            "curator: using deprecated curator.auxiliary.{provider,model} "
            "config — please migrate to auxiliary.curator.{provider,model}"
        )
        return _legacy_provider, _legacy_model

    # 3. Fall through to the main chat model
    return _main_provider, _main_model


def _run_llm_review(prompt: str) -> Dict[str, Any]:
    """Spawn an AIAgent fork to run the curator review prompt.

    Returns a dict with:
      - final: full (untruncated) final response from the reviewer
      - summary: short summary suitable for state file (240-char cap)
      - model, provider: what the fork actually ran on
      - tool_calls: list of {name, arguments} for every tool call made during
        the pass (arguments may be truncated for readability)
      - error: set if the pass failed mid-run; final/summary may still be empty

    Never raises; callers get a structured failure instead.
    """
    import contextlib
    result_meta: Dict[str, Any] = {
        "final": "",
        "summary": "",
        "model": "",
        "provider": "",
        "tool_calls": [],
        "error": None,
    }
    try:
        from run_agent import AIAgent
    except Exception as e:
        result_meta["error"] = f"AIAgent import failed: {e}"
        result_meta["summary"] = result_meta["error"]
        return result_meta

    # Resolve provider + model the same way the CLI does, so the curator
    # fork inherits the user's active main config rather than falling
    # through to an empty provider/model pair (which sends HTTP 400
    # "No models provided"). AIAgent() without explicit provider/model
    # arguments hits an auto-resolution path that fails for OAuth-only
    # providers and for pool-backed credentials.
    #
    # `_resolve_review_model()` honors `auxiliary.curator.{provider,model}`
    # (canonical aux-task slot, wired through `hermes model` → auxiliary
    # picker and the dashboard Models tab), with a legacy fallback to
    # `curator.auxiliary.{provider,model}`. See docs/user-guide/features/curator.md.
    _api_key = None
    _base_url = None
    _api_mode = None
    _resolved_provider = None
    _model_name = ""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider
        _cfg = load_config()
        _provider, _model_name = _resolve_review_model(_cfg)
        _rp = resolve_runtime_provider(
            requested=_provider, target_model=_model_name
        )
        _api_key = _rp.get("api_key")
        _base_url = _rp.get("base_url")
        _api_mode = _rp.get("api_mode")
        _resolved_provider = _rp.get("provider") or _provider
    except Exception as e:
        logger.debug("Curator provider resolution failed: %s", e, exc_info=True)

    result_meta["model"] = _model_name
    result_meta["provider"] = _resolved_provider or ""

    review_agent = None
    try:
        review_agent = AIAgent(
            model=_model_name,
            provider=_resolved_provider,
            api_key=_api_key,
            base_url=_base_url,
            api_mode=_api_mode,
            # Umbrella-building over a large skill collection is worth a
            # high iteration ceiling — the pass typically takes 50-100
            # API calls against hundreds of candidate skills. The
            # single-session review path caps itself at a much smaller
            # number because it's not doing a curation sweep.
            max_iterations=9999,
            quiet_mode=True,
            platform="curator",
            skip_context_files=True,
            skip_memory=True,
        )
        # Disable recursive nudges — the curator must never spawn its own review.
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0

        # Redirect the forked agent's stdout/stderr to /dev/null while it
        # runs so its tool-call chatter doesn't pollute the foreground
        # terminal. The background-thread runner also hides it; this
        # belt-and-suspenders path matters when a caller invokes
        # run_curator_review(synchronous=True) from the CLI.
        with open(os.devnull, "w") as _devnull, \
             contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            conv_result = review_agent.run_conversation(user_message=prompt)

        final = ""
        if isinstance(conv_result, dict):
            final = str(conv_result.get("final_response") or "").strip()
        result_meta["final"] = final
        result_meta["summary"] = (final[:240] + "…") if len(final) > 240 else (final or "no change")

        # Collect tool calls for the report. Walk the forked agent's
        # session messages and extract every tool_call made during the
        # pass. Truncate argument payloads so a giant skill_manage create
        # doesn't blow up the report.
        _calls: List[Dict[str, Any]] = []
        for msg in getattr(review_agent, "_session_messages", []) or []:
            if not isinstance(msg, dict):
                continue
            tcs = msg.get("tool_calls") or []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args_raw = fn.get("arguments") or ""
                if isinstance(args_raw, str) and len(args_raw) > 400:
                    args_raw = args_raw[:400] + "…"
                _calls.append({"name": name, "arguments": args_raw})
        result_meta["tool_calls"] = _calls
    except Exception as e:
        result_meta["error"] = f"error: {e}"
        result_meta["summary"] = result_meta["error"]
    finally:
        if review_agent is not None:
            try:
                review_agent.close()
            except Exception:
                pass
    return result_meta


# ---------------------------------------------------------------------------
# Public entrypoint for the session-start hook
# ---------------------------------------------------------------------------

def maybe_run_curator(
    *,
    idle_for_seconds: Optional[float] = None,
    on_summary: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Best-effort: run a curator pass if all gates pass. Returns the result
    dict if a pass was started, else None. Never raises."""
    try:
        if not should_run_now():
            return None
        # Idle gating: only enforce when the caller provided a measurement.
        if idle_for_seconds is not None:
            min_idle_s = get_min_idle_hours() * 3600.0
            if idle_for_seconds < min_idle_s:
                return None
        return run_curator_review(on_summary=on_summary)
    except Exception as e:
        logger.debug("maybe_run_curator failed: %s", e, exc_info=True)
        return None
