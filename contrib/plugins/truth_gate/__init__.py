"""Hermes Truth Gate plugin adapter.

Update-safe user plugin: validates final LLM output via SuperJarvis Truth Gate
source fork, writes Hermes-local packets, and blocks invalid output.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PLUGIN_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PLUGIN_DIR / "vendor"
DEFAULT_STATE_DIR = Path(os.path.expanduser("~")) / ".hermes" / "truth-gate"
SOURCE_HASHES = {
    "truth-stop-gate.py": "cbe01769b2d45c3c31708433f9bf926d10edda3c7d269cfeb627224751d73548",
}
_SECRETISH_RE = re.compile(r"(?i)(sk-[A-Za-z0-9_\-]{8,}|xox[baprs]-[A-Za-z0-9_\-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})")

_stop_mod = None


def _redact(text: str) -> str:
    return _SECRETISH_RE.sub("[REDACTED_SECRET_LIKE]", text or "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_stop_gate():
    global _stop_mod
    if _stop_mod is not None:
        return _stop_mod
    path = VENDOR_DIR / "truth-stop-gate.py"
    expected_hash = SOURCE_HASHES.get("truth-stop-gate.py")
    if expected_hash and _sha256(path) != expected_hash:
        raise RuntimeError(f"Truth Gate vendor hash mismatch: {path}")
    spec = importlib.util.spec_from_file_location("hermes_truth_gate_vendor_stop", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Truth Gate vendor module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _stop_mod = mod
    return mod


def _configure_state(mod, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    # Redirect Claude-specific state globals into Hermes-owned, update-safe state.
    # Hermes plugin scope is block-only validation: packets/logs are written;
    # no self-correction actuator or retry flag is produced.
    mod.GATE_LOG = state_dir / "stop-gate.log.jsonl"
    mod.DISCOVER_FLAG = state_dir / "discover-required.flag"
    mod.METRICS_GATE_FAILED_FLAG = state_dir / "metrics-gate-failed.flag"
    mod.REWRITE_FLAG = state_dir / "inactive-correction.flag"
    mod.REWRITE_FLAG_DIR = state_dir / "inactive-correction-flags"
    mod.PACKETS_DIR = state_dir / "packets"
    mod.STUCK_FLAG = state_dir / "inactive-correction-stuck.flag"
    mod.NEEDS_DISCOVERY_FLAG = state_dir / "needs-discovery.flag"
    mod.ARCHIVE_DIR = state_dir / "archive"
    mod.LEDGER = state_dir / "evidence-ledger.jsonl"
    for d in [mod.PACKETS_DIR, mod.ARCHIVE_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _plugin_enabled() -> bool:
    raw = os.getenv("HERMES_TRUTH_GATE_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_packet(state_dir: Path, session_id: str, response_text: str, violations: List[Dict[str, Any]], model: str, platform: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rule_ids = [str(v.get("rule", "")) for v in violations]
    seed = "|".join([session_id or "unknown", now, ",".join(rule_ids), str(len(response_text or ""))])
    packet_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    packet_path = state_dir / "packets" / f"{packet_id}.json"
    packet = {
        "version": 1,
        "adapter": "hermes-truth-gate-plugin",
        "source": "SuperJarvis Truth Gate fork",
        "source_hashes": SOURCE_HASHES,
        "packet_id": packet_id,
        "session_id": session_id or "",
        "model": model or "",
        "platform": platform or "",
        "created_at": now,
        "assistant_msg_bytes": len(response_text or ""),
        "assistant_msg_excerpt_redacted": _redact((response_text or "")[:500]),
        "rule_ids": rule_ids,
        "original_rule_ids": rule_ids,
        "violations": [{"rule": v.get("rule", ""), "match": _redact(str(v.get("match", ""))[:240]), "fix": v.get("fix", "")} for v in violations],
        "packet_path": str(packet_path),
        "enforcement_mode": "block_only",
        "correction_enabled": False,
        "gap": "Hermes plugin protects normal agent final responses; side-output paths are not universal enforcement surfaces yet.",
    }
    _write_json_atomic(packet_path, packet)
    return packet


def validate_response(response_text: str, session_id: str = "", model: str = "", platform: str = "", state_dir: str | None = None) -> Dict[str, Any]:
    state = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
    mod = _load_stop_gate()
    _configure_state(mod, state)
    # Hermes hook lacks reliable current-turn tool transcript; canonical footer is always required by the source evaluator.
    violations = mod.evaluate(response_text or "", False, "hermes-plugin", session_id or "", False)
    result: Dict[str, Any] = {"ok": not bool(violations), "violations": violations, "state_dir": str(state)}
    if violations:
        result["packet"] = _write_packet(state, session_id or "", response_text or "", violations, model or "", platform or "")
    return result


def _format_block(response_text: str, result: Dict[str, Any]) -> str:
    packet = result.get("packet") or {}
    violations = result.get("violations") or []
    lines = [
        "TRUTH GATE BLOCK -- final answer violates rules.",
        "The unsafe/unproven answer was withheld by the Hermes Truth Gate plugin.",
        f"packet_id: {packet.get('packet_id','')}",
        f"packet_path: {packet.get('packet_path','')}",
        "",
        "VIOLATIONS:",
    ]
    for v in violations[:12]:
        lines.append(f"- {v.get('rule','')}: {v.get('fix','')}")
    excerpt = _redact((response_text or "")[:300])
    if excerpt:
        lines.extend(["", "REDACTED_EXCERPT:", excerpt])
    lines.extend([
        "",
        "GAP:",
        "- Truth Gate protected this normal Hermes final response. Side-output paths are not universal enforcement surfaces yet.",
    ])
    return "\n".join(lines)


def _format_unavailable_block(error: Exception) -> str:
    return "\n".join([
        "TRUTH GATE BLOCK -- plugin unavailable.",
        "The original answer was withheld because Truth Gate could not validate it.",
        f"error: {type(error).__name__}",
        "",
        "GAP:",
        "- Front-door validation failed closed. Fix plugin/vendor integrity before trusting final output.",
    ])


def transform_llm_output(response_text: str = "", session_id: str = "", model: str = "", platform: str = "", state_dir: str | None = None, **_: Any) -> str | None:
    if not _plugin_enabled():
        return None
    try:
        result = validate_response(response_text or "", session_id=session_id or "", model=model or "", platform=platform or "", state_dir=state_dir)
    except Exception as exc:
        return _format_unavailable_block(exc)
    if result.get("ok"):
        return response_text
    return _format_block(response_text or "", result)


def get_status() -> Dict[str, Any]:
    """Return the current Hermes Truth Gate enforcement contract.

    Lame-terms metric: path first, violations second.
    Front door means normal agent final responses pass through the
    transform_llm_output hook. Side doors are intentionally reported as no
    until explicit tests/wrappers prove coverage.
    """
    return {
        "plugin": "truth_gate",
        "enabled_by_env_default": _plugin_enabled(),
        "enforcement_mode": "block_only",
        "correction_enabled": False,
        "front_door": {
            "agent_final_response": "yes",
        },
        "side_doors": {
            "raw_tool_stdout": "no",
            "no_agent_cron_stdout": "no",
            "direct_send_message": "no",
            "system_platform_messages": "no",
        },
        "trigger_metric": "front_door_yes_and_violation_count_gt_0",
        "pass_metric": "front_door_yes_and_violation_count_eq_0",
        "proof_rule": "0 violations only means the checker ran and found no rule break; proof must be present when the footer rules require proof.",
    }


def _tool_status(args: Dict[str, Any] | None = None, **_: Any) -> str:
    return json.dumps(get_status(), indent=2, sort_keys=True)


def _tool_validate(args: Dict[str, Any], **_: Any) -> str:
    text = args.get("text") or args.get("response_text") or ""
    session_id = args.get("session_id") or "manual"
    result = validate_response(text, session_id=session_id, model=args.get("model", ""), platform=args.get("platform", "tool"))
    return json.dumps(result, indent=2, default=str)


def register(ctx):
    ctx.register_hook("transform_llm_output", transform_llm_output)
    ctx.register_tool(
        name="truth_gate_status",
        toolset="truth_gate",
        description="Report Truth Gate front-door/side-door enforcement status and trigger metric.",
        schema={
            "name": "truth_gate_status",
            "description": "Report Hermes Truth Gate enforcement status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        handler=_tool_status,
    )
    ctx.register_tool(
        name="truth_gate_validate",
        toolset="truth_gate",
        description="Validate text with the SuperJarvis Truth Gate fork and write a Hermes-local packet on failure.",
        schema={
            "name": "truth_gate_validate",
            "description": "Validate response text with Hermes Truth Gate plugin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "session_id": {"type": "string"},
                    "model": {"type": "string"},
                    "platform": {"type": "string"},
                },
                "required": ["text"],
            },
        },
        handler=_tool_validate,
    )
