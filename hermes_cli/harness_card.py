"""Opt-in, secret-safe Harness Card + machine-readable run export (issue #110680).

One deterministic, versioned document describing the *effective* harness a benchmark arm ran:
identity/runtime, model route + pricing snapshot, prompt/skill/project-context hashes, toolsets,
memory/compression/limits/delegation/retries, evaluator + verification, and — for a named session
— the per-task token/cost rows usage accounting already wrote.

Reuse, not a parallel telemetry stack:

* config values     -> ``load_config()`` (``DEFAULT_CONFIG`` merged) read through ``cfg_get``
* per-task usage    -> ``sessions`` / ``session_model_usage`` (``hermes_state_usage``)
* session record    -> ``SessionDB.export_session`` (``hermes_state_portability``)
* prompt/skill ctx  -> ``hermes_cli.prompt_size``'s offline inspection agent + skill breakdown
* verification      -> ``agent.verify.environment.load_or_detect`` (the ``hermes verify`` recipe)
* redaction         -> ``agent.redact.redact_sensitive_text`` in force mode

Nothing is produced unless ``harness_card.enabled`` is on (``hermes config set harness_card.enabled
true``), and every string leaf is redacted before it can reach stdout or the written artifact.
Unavailable telemetry is reported as ``null`` with a ``coverage`` entry — never as a fake zero
(``--solved`` is how a caller turns ``run.outcome.solved`` from unavailable into reported).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HARNESS_CARD_SCHEMA_ID = "hermes.harness-card"
HARNESS_CARD_SCHEMA_VERSION = 1

_ENABLE_HINT = "  Enable it with: hermes config set harness_card.enabled true"

# Project context files ``agent/prompt_builder.build_context_files_prompt`` picks from. Probed for
# existence/hash only: which one *wins* is already covered by the context tier hash below, so this
# deliberately does not re-implement the precedence walk.
_CONTEXT_FILE_CANDIDATES = (".hermes.md", "HERMES.md", "AGENTS.md", "CLAUDE.md", ".cursorrules")

# Billing modes whose pricing snapshot resolves from the bundled table (no network). Routes that
# resolve from a live metadata fetch (openrouter / nous / custom base_url) report the snapshot as
# unavailable instead of dialing out from a diagnostic — see _pricing_snapshot.
_OFFLINE_PRICING_MODES = frozenset({"official_docs_snapshot", "subscription_included"})


def harness_card_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """The opt-in gate: ``harness_card.enabled`` in config.yaml, default false."""
    from hermes_cli.config import cfg_get, load_config
    cfg = cfg if cfg is not None else load_config()
    return bool(cfg_get(cfg, "harness_card", "enabled", default=False))


# --------------------------------------------------------------------------- #
# Redaction / hashing primitives
# --------------------------------------------------------------------------- #

def _scrub(value: Any) -> Any:
    """Recursively redact every string leaf.

    ``force=True`` is the same boundary ``hermes debug share`` uses: the card must never carry a
    credential even when the operator turned ``security.redact_secrets`` off.
    ``redact_url_credentials=True`` is on here (it is off for ordinary tool output) because nothing
    in a card is a live OAuth callback / magic link, so a ``user:pass@`` base_url is pure leak.
    """
    from agent.redact import redact_sensitive_text

    if isinstance(value, str):
        return redact_sensitive_text(value, force=True, redact_url_credentials=True)
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decimal_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _iso(epoch: Any) -> Optional[str]:
    if not isinstance(epoch, (int, float)):
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Card sections
# --------------------------------------------------------------------------- #

def _hermes_identity() -> Dict[str, Any]:
    """Hermes version/commit, install method, and the runtime the card was taken on."""
    from hermes_cli.build_info import get_code_identity
    from hermes_cli.config import detect_install_method

    identity = get_code_identity()
    return {
        "version": identity.get("version"),
        "commit": identity.get("sha"),
        "short_commit": identity.get("short_sha"),
        "commit_source": identity.get("source"),
        "install_method": detect_install_method(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "frozen": bool(getattr(sys, "frozen", False)),
        },
    }


def _profile_identity() -> Dict[str, Any]:
    from hermes_cli.config import get_config_path
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import get_hermes_home

    return {
        "name": get_active_profile_name(),
        "home": str(get_hermes_home()),
        "config_path": str(get_config_path()),
    }


def _pricing_snapshot(model: str, provider: str, base_url: str) -> Dict[str, Any]:
    """Pricing snapshot identifier for the configured route, offline only.

    ``resolve_billing_route`` is a pure function; the snapshot lookup is only run for routes the
    bundled table answers. A ``official_models_api`` route (openrouter/nous) would need a network
    metadata fetch, so its snapshot is reported unavailable — a diagnostic never dials out.
    """
    from agent.usage_pricing import get_pricing_entry, resolve_billing_route

    route = resolve_billing_route(model, provider=provider, base_url=base_url)
    snapshot: Dict[str, Any] = {
        "billing_mode": route.billing_mode,
        "route_provider": route.provider,
        "route_model": route.model,
        "version": None,
        "source": None,
        "source_url": None,
        "input_cost_per_million": None,
        "output_cost_per_million": None,
    }
    if route.billing_mode not in _OFFLINE_PRICING_MODES:
        return snapshot
    entry = get_pricing_entry(route.model, provider=route.provider, base_url=route.base_url)
    if entry is None:
        return snapshot
    snapshot.update(
        version=entry.pricing_version, source=entry.source, source_url=entry.source_url,
        input_cost_per_million=_decimal_str(entry.input_cost_per_million),
        output_cost_per_million=_decimal_str(entry.output_cost_per_million),
    )
    return snapshot


def _model_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.config import cfg_get

    model = cfg_get(cfg, "model", "default") or cfg_get(cfg, "model", "model") or ""
    provider = cfg_get(cfg, "model", "provider") or ""
    base_url = cfg_get(cfg, "model", "base_url") or ""
    return {
        "name": model or None,
        "provider": provider or None,
        "base_url": base_url or None,
        "api_mode": cfg_get(cfg, "model", "api_mode") or None,
        "fallback_providers": cfg_get(cfg, "fallback_providers") or [],
        "pricing": _pricing_snapshot(model, provider, base_url),
    }


def _project_files(cwd: Optional[str]) -> List[Dict[str, Any]]:
    """Existing project-context candidates in *cwd*, as path/hash only (never content)."""
    root = Path(cwd or ".").resolve()
    files: List[Dict[str, Any]] = []
    for name in _CONTEXT_FILE_CANDIDATES:
        path = root / name
        try:
            if not path.is_file():
                continue
            data = path.read_bytes()
        except OSError:
            continue
        files.append({
            "path": name,
            "bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        })
    return files


def _context_section(cfg: Dict[str, Any], platform: str, cwd: Optional[str]) -> Dict[str, Any]:
    """Hashes of the rendered prompt tiers + skill identifiers.

    Raw system prompt / skill / project text never enters the card: the tiers are hashed and the
    skill index contributes names only.
    """
    from hermes_cli.config import cfg_get

    section: Dict[str, Any] = {
        "cwd": str(Path(cwd).resolve()) if cwd else str(Path.cwd().resolve()),
        "system_prompt_sha256": None,
        "tiers": {"stable_sha256": None, "context_sha256": None, "volatile_sha256": None},
        "skills": {"count": None, "names": [], "index_sha256": None},
        "project_files": _project_files(cwd),
        "memory": {
            "enabled": bool(cfg_get(cfg, "memory", "memory_enabled", default=True)),
            "user_profile_enabled": bool(cfg_get(cfg, "memory", "user_profile_enabled", default=True)),
            "provider": cfg_get(cfg, "memory", "provider") or None,
            "memory_char_limit": cfg_get(cfg, "memory", "memory_char_limit"),
            "user_char_limit": cfg_get(cfg, "memory", "user_char_limit"),
        },
    }
    # Best-effort: an inspection agent needs the install to be importable, and a card without
    # prompt hashes is still useful — it just says so in ``coverage``.
    try:
        from hermes_cli.prompt_size import _SKILLS_BLOCK_RE, _build_inspection_agent, _compute_skills_breakdown
        from agent.system_prompt import build_system_prompt, build_system_prompt_parts

        agent = _build_inspection_agent(platform)
        parts = build_system_prompt_parts(agent)
        stable, context, volatile = (parts.get(key, "") or "" for key in ("stable", "context", "volatile"))
        section["system_prompt_sha256"] = _sha256(build_system_prompt(agent))
        section["tiers"] = {
            "stable_sha256": _sha256(stable), "context_sha256": _sha256(context),
            "volatile_sha256": _sha256(volatile),
        }
        match = _SKILLS_BLOCK_RE.search(volatile) or _SKILLS_BLOCK_RE.search(stable)
        if match is None:
            # The tiers rendered, so an absent skills block is a real zero — not the ``null`` a
            # failed prompt inspection leaves behind.
            section["skills"] = {"count": 0, "names": [], "index_sha256": None}
        else:
            names = sorted({
                entry["name"] for entry in _compute_skills_breakdown(match.group(0)) if entry.get("name")
            })
            section["skills"] = {
                "count": len(names), "names": names, "index_sha256": _sha256(match.group(0)),
            }
    except Exception:  # noqa: BLE001 - prompt inspection is optional detail, never fatal
        pass
    return section


def _toolsets_section(cfg: Dict[str, Any], platform: str) -> Dict[str, Any]:
    from hermes_cli.config import cfg_get
    from hermes_cli.tools_config import _get_platform_tools, enabled_mcp_server_names
    from agent.skill_utils import parse_config_string_list

    try:
        enabled = sorted(_get_platform_tools(cfg, platform))
    except Exception:  # noqa: BLE001 - a bad platform key must not sink the card
        enabled = []
    return {
        "platform": platform,
        "enabled": enabled,
        "disabled": sorted(parse_config_string_list(cfg_get(cfg, "agent", "disabled_toolsets")) or []),
        # Names only: server definitions carry env vars and headers that are not card material.
        "mcp_servers": sorted(enabled_mcp_server_names(cfg)),
        "declared": sorted(parse_config_string_list(cfg_get(cfg, "toolsets")) or []),
    }


def _limits_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.config import TURN_LIMIT_UNLIMITED, cfg_get, resolve_turn_limit

    turns = resolve_turn_limit(cfg_get(cfg, "agent", "max_turns"))
    return {
        "max_turns": None if turns >= TURN_LIMIT_UNLIMITED else turns,
        "run_budget_seconds": cfg_get(cfg, "agent", "run_budget_seconds"),
        "gateway_timeout_seconds": cfg_get(cfg, "agent", "gateway_timeout"),
        "session_stall_timeout_seconds": cfg_get(cfg, "agent", "session_stall_timeout"),
        "goals_max_turns": cfg_get(cfg, "goals", "max_turns"),
        "context_file_max_chars": cfg_get(cfg, "context_file_max_chars"),
        "file_read_max_chars": cfg_get(cfg, "file_read_max_chars"),
        "tool_output": cfg_get(cfg, "tool_output"),
        "tool_loop_guardrails": cfg_get(cfg, "tool_loop_guardrails"),
    }


def _compression_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.config import cfg_get

    return {
        "engine": cfg_get(cfg, "context", "engine"),
        "enabled": cfg_get(cfg, "compression", "enabled"),
        "threshold": cfg_get(cfg, "compression", "threshold"),
        "threshold_tokens": cfg_get(cfg, "compression", "threshold_tokens"),
        "target_ratio": cfg_get(cfg, "compression", "target_ratio"),
        "tail_mode": cfg_get(cfg, "compression", "tail_mode"),
        "in_place": cfg_get(cfg, "compression", "in_place"),
        "protect_first_n": cfg_get(cfg, "compression", "protect_first_n"),
        "protect_last_n": cfg_get(cfg, "compression", "protect_last_n"),
        "max_attempts": cfg_get(cfg, "compression", "max_attempts"),
        "micro_compact": cfg_get(cfg, "compression", "micro_compact"),
        "idle_compact_after_seconds": cfg_get(cfg, "compression", "idle_compact_after_seconds"),
        "hygiene_hard_message_limit": cfg_get(cfg, "compression", "hygiene_hard_message_limit"),
        "proactive_prune_tokens": cfg_get(cfg, "compression", "proactive_prune_tokens"),
    }


def _delegation_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.config import cfg_get

    return {
        "model": cfg_get(cfg, "delegation", "model") or None,
        "provider": cfg_get(cfg, "delegation", "provider") or None,
        "max_iterations": cfg_get(cfg, "delegation", "max_iterations"),
        "max_concurrent_children": cfg_get(cfg, "delegation", "max_concurrent_children"),
        "max_spawn_depth": cfg_get(cfg, "delegation", "max_spawn_depth"),
        "child_timeout_seconds": cfg_get(cfg, "delegation", "child_timeout_seconds"),
        "compression_threshold_tokens": cfg_get(cfg, "delegation", "compression_threshold_tokens"),
        "inherit_mcp_toolsets": cfg_get(cfg, "delegation", "inherit_mcp_toolsets"),
        "orchestrator_enabled": cfg_get(cfg, "delegation", "orchestrator_enabled"),
    }


def _retries_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.config import cfg_get

    return {
        "api_max_retries": cfg_get(cfg, "agent", "api_max_retries"),
        "empty_response_guard": cfg_get(cfg, "agent", "empty_response_guard"),
        "compression_max_attempts": cfg_get(cfg, "compression", "max_attempts"),
        "sanitizer_heal_escalation_threshold": cfg_get(cfg, "agent", "sanitizer_heal_escalation_threshold"),
        "stall_guards": cfg_get(cfg, "agent", "stall_guards"),
        "turn_liveness": cfg_get(cfg, "agent", "turn_liveness"),
    }


def _evaluator_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The knobs that decide whether a change has to be proven before the run ends."""
    from hermes_cli.config import cfg_get

    return {
        "verify_guidance": cfg_get(cfg, "agent", "verify_guidance"),
        "max_verify_nudges": cfg_get(cfg, "agent", "max_verify_nudges"),
        "verify_on_stop": cfg_get(cfg, "agent", "verify_on_stop"),
        "task_completion_guidance": cfg_get(cfg, "agent", "task_completion_guidance"),
        "tool_use_enforcement": cfg_get(cfg, "agent", "tool_use_enforcement"),
    }


def _verification_section(cwd: Optional[str]) -> Dict[str, Any]:
    """The ``hermes verify`` recipe this run would be measured against, if one is discoverable."""
    from agent.verify.environment import MANIFEST_VERSION, load_or_detect, manifest_path

    root = Path(cwd or ".").resolve()
    section: Dict[str, Any] = {
        "manifest": {"path": str(manifest_path(root)), "version": MANIFEST_VERSION, "present": False},
        "source": "none",
        "recipe": None,
    }
    try:
        recipe, source = load_or_detect(root)
    except Exception:  # noqa: BLE001 - undetectable project is a legitimate card state
        return section
    section["source"] = source
    section["manifest"]["present"] = manifest_path(root).is_file()
    if recipe is not None:
        section["recipe"] = recipe.to_dict()
    return section


# --------------------------------------------------------------------------- #
# Card assembly
# --------------------------------------------------------------------------- #

def build_harness_card(
    *, platform: str = "cli", cwd: Optional[str] = None,
    session_id: Optional[str] = None, solved: Optional[bool] = None,
    failure_class: Optional[str] = None, db: Any = None,
) -> Dict[str, Any]:
    """Build the Harness Card. Assumes the opt-in gate has already been checked.

    ``db`` lets a caller (a gateway/dashboard holding a SessionDB) pass its own handle; when
    omitted and *session_id* is given, a read-only handle is opened and closed here.
    """
    from hermes_cli.config import load_config

    cfg = load_config()
    card: Dict[str, Any] = {
        "schema": {"id": HARNESS_CARD_SCHEMA_ID, "version": HARNESS_CARD_SCHEMA_VERSION},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "hermes": _hermes_identity(),
        "profile": _profile_identity(),
        "model": _model_section(cfg),
        "context": _context_section(cfg, platform, cwd),
        "toolsets": _toolsets_section(cfg, platform),
        "limits": _limits_section(cfg),
        "compression": _compression_section(cfg),
        "delegation": _delegation_section(cfg),
        "retries": _retries_section(cfg),
        "evaluator": _evaluator_section(cfg),
        "verification": _verification_section(cwd),
    }
    if session_id:
        card["run"] = build_run_export(
            session_id, solved=solved, failure_class=failure_class, db=db)
    card = _scrub(card)
    # Digest of the harness-defining body only: generated_at/run/coverage move between runs, so the
    # paired-arm key must not include them.
    digest_body = {key: value for key, value in card.items() if key not in ("generated_at", "run")}
    card["card_id"] = _sha256(json.dumps(digest_body, sort_keys=True, ensure_ascii=False, default=str))
    return card


# --------------------------------------------------------------------------- #
# Per-task run export (reuses usage accounting)
# --------------------------------------------------------------------------- #

def build_run_export(
    session_id: str, *, solved: Optional[bool] = None, failure_class: Optional[str] = None, db: Any = None,
) -> Dict[str, Any]:
    """Per-task trace fields for *session_id* from the existing session + usage rows.

    Tokens/cost come from the ``sessions`` summary row and ``session_model_usage`` (per model,
    provider and task). Turn-level counters (productive/no-action/repeated turns, verification
    attempts) need the per-turn trace pipeline, which this card deliberately does not build — they
    are reported as ``null`` with a ``coverage`` entry rather than faked.
    """
    owns_db = db is None
    if owns_db:
        from hermes_state import SessionDB
        try:
            db = SessionDB(read_only=True)
        except Exception as exc:  # noqa: BLE001 - no readable store means no run to report
            raise LookupError(f"no session store to read: {exc}") from exc
    try:
        resolved = db.resolve_session_id(session_id) or session_id
        session = db.export_session(resolved)
        if session is None:
            raise LookupError(f"session not found: {session_id}")
        usage_rows = db.session_usage_breakdown(session["id"])
        lineage = db.get_compression_lineage(session["id"]) or []
    finally:
        if owns_db:
            db.close()

    cost_status = session.get("cost_status")
    run: Dict[str, Any] = {
        "session_id": session.get("id"),
        "source": session.get("source"),
        "started_at": _iso(session.get("started_at")),
        "ended_at": _iso(session.get("ended_at")),
        "wall_seconds": _wall_seconds(session),
        "end_reason": session.get("end_reason"),
        "outcome": {
            "solved": solved,
            "failure_class": failure_class,
        },
        "model": {
            "name": session.get("model"),
            "provider": session.get("billing_provider"),
            "base_url": session.get("billing_base_url"),
            "billing_mode": session.get("billing_mode"),
        },
        "tokens": {
            "input": session.get("input_tokens"),
            "output": session.get("output_tokens"),
            "cache_read": session.get("cache_read_tokens"),
            "cache_write": session.get("cache_write_tokens"),
            "reasoning": session.get("reasoning_tokens"),
            "api_calls": session.get("api_call_count"),
        },
        "cost": {
            "estimated_usd": session.get("estimated_cost_usd"),
            "actual_usd": session.get("actual_cost_usd"),
            "status": cost_status,
            "source": session.get("cost_source"),
            "pricing_version": session.get("pricing_version"),
        },
        "messages": {
            "count": session.get("message_count"),
            "tool_calls": session.get("tool_call_count"),
            "tool_names": _tool_names(session),
        },
        "turns": {
            "count": None, "productive": None, "no_action": None, "repeated_action": None,
        },
        "verification": {
            "attempts": None,
            "api_retries": None,
            "compression_count": max(0, len(lineage) - 1) if lineage else None,
            "rewind_count": session.get("rewind_count"),
            "compression_ineffective_count": session.get("compression_ineffective_count"),
        },
        "usage_by_task": [
            {
                "task": row.get("task") or "",
                "model": row.get("model"),
                "provider": row.get("billing_provider"),
                "api_calls": row.get("api_call_count"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "cache_read_tokens": row.get("cache_read_tokens"),
                "cache_write_tokens": row.get("cache_write_tokens"),
                "reasoning_tokens": row.get("reasoning_tokens"),
                "estimated_cost_usd": row.get("estimated_cost_usd"),
                "actual_cost_usd": row.get("actual_cost_usd"),
                "cost_status": row.get("cost_status"),
                "cost_source": row.get("cost_source"),
            } for row in usage_rows
        ],
        # Raw metrics first: no composite score is derived here, and every class a reader might want
        # to divide by is labelled reported / unavailable so a zero is never mistaken for silence.
        "coverage": {
            "tokens": "reported",
            "cost": "reported" if cost_status or session.get("estimated_cost_usd") is not None else "unavailable",
            "usage_by_task": "reported" if usage_rows else "unavailable",
            "turns": "unavailable",
            "verification_attempts": "unavailable",
            "outcome_solved": "reported" if solved is not None else "unavailable",
        },
    }
    return run


def _wall_seconds(session: Dict[str, Any]) -> Optional[float]:
    started, ended = session.get("started_at"), session.get("ended_at")
    if not isinstance(started, (int, float)) or not isinstance(ended, (int, float)):
        return None
    return round(float(ended) - float(started), 3)


def _tool_names(session: Dict[str, Any]) -> List[str]:
    from utils import safe_json_loads

    raw = session.get("tool_names")
    if isinstance(raw, str):
        raw = safe_json_loads(raw, default=[])
    return sorted(str(name) for name in raw) if isinstance(raw, list) else []


# --------------------------------------------------------------------------- #
# Rendering / CLI
# --------------------------------------------------------------------------- #

def render_card(card: Dict[str, Any]) -> str:
    """Human-readable summary of a card (``--json`` carries the full document)."""
    hermes, model, context = card["hermes"], card["model"], card["context"]
    pricing = model["pricing"]
    skills = context["skills"]
    lines = [
        f"Harness Card {card['card_id']}",
        f"  schema     : {card['schema']['id']} v{card['schema']['version']}  (generated {card['generated_at']})",
        f"  hermes     : {hermes['version'] or 'unknown'} [{hermes['short_commit'] or 'unknown'}]"
        f"  {hermes['commit_source']}, install={hermes['install_method']}",
        f"  runtime    : python {hermes['runtime']['python']} on {hermes['runtime']['system']}"
        f" {hermes['runtime']['release']} ({hermes['runtime']['machine']})",
        f"  profile    : {card['profile']['name']}  ({card['profile']['home']})",
        f"  model      : {model['name'] or 'unset'}  provider={model['provider'] or 'auto'}"
        f"  billing={pricing['billing_mode']}",
        f"  pricing    : {pricing['version'] or 'unavailable (offline lookup)'}",
        f"  toolsets   : platform={card['toolsets']['platform']}"
        f"  enabled={len(card['toolsets']['enabled'])}  mcp={len(card['toolsets']['mcp_servers'])}",
        f"  context    : prompt={context['system_prompt_sha256'] or 'unavailable'}"
        f"  skills={skills['count'] if skills['count'] is not None else 'unavailable'}"
        f"  project_files={len(context['project_files'])}",
        f"  limits     : max_turns={card['limits']['max_turns'] or 'unlimited'}"
        f"  run_budget={card['limits']['run_budget_seconds']}",
        f"  verification: {card['verification']['source']}"
        f"  recipe={(card['verification']['recipe'] or {}).get('name') or 'none'}",
    ]
    if run := card.get("run"):
        tokens, cost = run["tokens"], run["cost"]
        wall = f"{run['wall_seconds']}s" if run["wall_seconds"] is not None else "wall time unavailable"
        lines += [
            "",
            f"  run {run['session_id']}  ({run['started_at']} → {run['ended_at'] or 'open'}, {wall})",
            f"    outcome   : solved={run['outcome']['solved']}"
            f"  failure_class={run['outcome']['failure_class']}  end_reason={run['end_reason']}",
            f"    tokens    : in={tokens['input']} out={tokens['output']}"
            f" cache_read={tokens['cache_read']} api_calls={tokens['api_calls']}",
            f"    cost      : estimated={cost['estimated_usd']} actual={cost['actual_usd']}"
            f" status={cost['status']}",
            f"    tasks     : {len(run['usage_by_task'])} usage row(s)",
            f"    coverage  : {run['coverage']}",
        ]
    return "\n".join(lines)


def _write_card(card: Dict[str, Any], out_path: str) -> Path:
    """Write the card as JSON, owner-only where the platform supports it.

    ``_secure_file`` is the same chmod-0600 path config.yaml/.env use (and it already no-ops in
    containers/managed installs). Content is redacted before it gets here.
    """
    from hermes_cli.config import _secure_file

    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    _secure_file(path)
    return path


def cmd_harness_card(args: Any) -> None:
    """Entry point for ``hermes harness-card``."""
    if not harness_card_enabled():
        # Loud, not silent: an automated benchmark caller must not mistake "opt-in is off" for an
        # empty-but-valid run record.
        print("Harness cards are opt-in and disabled.", file=sys.stderr)
        print(_ENABLE_HINT.strip(), file=sys.stderr)
        raise SystemExit(1)

    session_id = getattr(args, "session", None)
    solved_raw = getattr(args, "solved", None)
    solved = None if solved_raw is None else str(solved_raw).strip().lower() in {"1", "true", "yes", "on"}
    try:
        card = build_harness_card(
            platform=getattr(args, "platform", "cli") or "cli",
            cwd=getattr(args, "cwd", None),
            session_id=session_id,
            solved=solved,
            failure_class=getattr(args, "failure_class", None),
        )
    except LookupError as exc:
        print(f"Could not build harness card: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if out_path := getattr(args, "out", None):
        print(f"Wrote harness card to {_write_card(card, out_path)}")
        return
    if getattr(args, "json", False):
        print(json.dumps(card, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print(render_card(card))
