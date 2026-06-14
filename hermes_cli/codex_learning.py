"""Codex cockpit learning harvester.

This module is intentionally CLI/gateway neutral. The process registry calls
``handle_process_completed`` when a background process exits; this module
filters for `/codex launch` processes, builds a compact learning packet, runs
a memory/skill-only Hermes review, and records links to the staged approval
records created by ``tools.write_approval``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


DEFAULT_LEARNING = {
    "enabled": False,
    "harvest_launches": True,
    "review_model": "",
    "max_output_chars": 8000,
    "auto_promote_memory": True,
    "auto_promote_skills": True,
    "min_confidence": 0.75,
    "notify_on_stage": True,
}


@dataclass(frozen=True)
class LearningConfig:
    enabled: bool
    harvest_launches: bool
    review_model: str
    max_output_chars: int
    auto_promote_memory: bool
    auto_promote_skills: bool
    min_confidence: float
    notify_on_stage: bool


def load_learning_config(config: Mapping[str, Any] | None) -> LearningConfig:
    """Return normalized ``codex_cockpit.context_helper`` learning settings."""
    raw: dict[str, Any] = {}
    if isinstance(config, Mapping):
        cockpit = config.get("codex_cockpit", {})
        if isinstance(cockpit, Mapping):
            helper = cockpit.get("context_helper", {})
            if isinstance(helper, Mapping):
                raw = dict(helper)

    merged = dict(DEFAULT_LEARNING)
    merged.update(raw)
    return LearningConfig(
        enabled=_as_bool(merged.get("enabled")),
        harvest_launches=_as_bool(merged.get("harvest_launches")),
        review_model=str(merged.get("review_model") or "").strip(),
        max_output_chars=_positive_int(merged.get("max_output_chars"), 8000),
        auto_promote_memory=_as_bool(merged.get("auto_promote_memory")),
        auto_promote_skills=_as_bool(merged.get("auto_promote_skills")),
        min_confidence=_confidence(merged.get("min_confidence"), 0.75),
        notify_on_stage=_as_bool(merged.get("notify_on_stage")),
    )


def is_codex_launch_session(session: Any) -> bool:
    """True when a process registry session looks like `/codex launch` output."""
    task_id = str(getattr(session, "task_id", "") or "")
    command = str(getattr(session, "command", "") or "")
    if not task_id.startswith("codex_"):
        return False
    lowered = command.lower()
    return "codex exec" in lowered and "worktree add" in lowered


def handle_process_completed(session: Any) -> Optional[dict[str, Any]]:
    """Harvest and review a completed process registry session if configured."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception as exc:
        logger.debug("codex learning: could not load config: %s", exc)
        return None

    learning = load_learning_config(config)
    if not learning.enabled or not learning.harvest_launches:
        return None
    if not is_codex_launch_session(session):
        return None

    packet = build_learning_packet(session, config)
    if packet_exists(packet["id"]):
        return load_packet(packet["id"])
    save_packet(packet)
    start_review_thread(packet["id"], config)
    return packet


def build_learning_packet(session: Any, config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build the bounded, durable packet used by the Hermes reviewer."""
    learning = load_learning_config(config)
    command = str(getattr(session, "command", "") or "")
    cwd = str(getattr(session, "cwd", "") or "")
    process_id = str(getattr(session, "id", "") or uuid.uuid4().hex[:12])
    output = str(getattr(session, "output_buffer", "") or "")
    try:
        from tools.ansi_strip import strip_ansi

        output = strip_ansi(output)
    except Exception:
        pass
    output_tail = output[-learning.max_output_chars:] if output else ""

    repo = _resolve_repo_root(cwd) or _extract_git_repo_from_command(command) or cwd
    worktree = _extract_codex_worktree(command)
    branch = _extract_branch_from_command(command) or _git_branch(worktree or cwd)
    now = time.time()

    return {
        "id": _packet_id(process_id),
        "status": "harvested",
        "source": "process_registry",
        "process_id": process_id,
        "task_id": str(getattr(session, "task_id", "") or ""),
        "session_key": str(getattr(session, "session_key", "") or ""),
        "pid": getattr(session, "pid", None),
        "command": command,
        "cwd": cwd,
        "repo": repo,
        "branch": branch,
        "worktree": worktree,
        "exit_code": getattr(session, "exit_code", None),
        "started_at": getattr(session, "started_at", None),
        "completed_at": now,
        "output_tail": output_tail,
        "output_tail_chars": len(output_tail),
        "cockpit_status": _cockpit_status_snapshot(config),
        "proposal_ids": [],
        "created_at": now,
        "updated_at": now,
    }


def start_review_thread(packet_id: str, config: Mapping[str, Any] | None) -> threading.Thread:
    """Start the background Hermes reviewer for a harvested packet."""
    thread = threading.Thread(
        target=review_packet,
        args=(packet_id, dict(config or {})),
        daemon=True,
        name=f"codex-learning-review-{packet_id}",
    )
    thread.start()
    return thread


def review_packet(packet_id: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run Hermes review for a packet and record staged proposal links."""
    packet = load_packet(packet_id)
    if not packet:
        raise ValueError(f"No Codex learning packet with id {packet_id!r}")

    learning = load_learning_config(config)
    packet["status"] = "reviewing"
    packet["updated_at"] = time.time()
    save_packet(packet)

    before = _pending_id_snapshot()
    try:
        _run_hermes_reviewer(packet, learning)
    except Exception as exc:
        logger.warning("Codex learning review failed for %s: %s", packet_id, exc)
        packet["status"] = "failed"
        packet["error"] = str(exc)
        packet["updated_at"] = time.time()
        save_packet(packet)
        return packet

    after = _pending_id_snapshot()
    proposal_ids = []
    for subsystem in ("memory", "skills"):
        for pending_id in sorted(after[subsystem] - before[subsystem]):
            pending = _get_pending(subsystem, pending_id)
            proposal = _proposal_from_pending(packet, subsystem, pending_id, pending)
            save_proposal(proposal)
            proposal_ids.append(proposal["id"])

    packet["proposal_ids"] = proposal_ids
    packet["reviewed_at"] = time.time()
    packet["status"] = "staged" if proposal_ids else "reviewed"
    packet["updated_at"] = time.time()
    save_packet(packet)

    if proposal_ids:
        _auto_promote_if_configured(proposal_ids, learning)
        pending_after = [
            proposal_id
            for proposal_id in proposal_ids
            if (load_proposal(proposal_id) or {}).get("approval_state") == "pending"
        ]
        packet["status"] = "staged" if pending_after else "applied"
        packet["updated_at"] = time.time()
        save_packet(packet)
        if pending_after and learning.notify_on_stage:
            _queue_stage_notification(packet, pending_after)
    return packet


def render_learn_status(config: Mapping[str, Any] | None) -> str:
    learning = load_learning_config(config)
    packets = list_packets()
    proposals = list_proposals()
    pending = [p for p in proposals if p.get("approval_state") == "pending"]
    lines = [
        "**Codex Learn**",
        f"- Enabled: {_yes_no(learning.enabled)}",
        f"- Harvest launches: {_yes_no(learning.harvest_launches)}",
        f"- Review model: `{learning.review_model or 'default Hermes model'}`",
        f"- Max output chars: {learning.max_output_chars}",
        f"- Min confidence: {learning.min_confidence:.2f}",
        (
            "- Auto promote: "
            f"memory={_yes_no(learning.auto_promote_memory)}, "
            f"skills={_yes_no(learning.auto_promote_skills)}"
        ),
        f"- Curator scope: `{_curator_scope()}`",
        f"- Pending proposals: {len(pending)}",
    ]
    if packets:
        latest = packets[-1]
        lines.append(
            "- Last harvest: "
            f"`{latest.get('id')}` {latest.get('status')} "
            f"repo=`{latest.get('repo') or '?'}` branch=`{latest.get('branch') or '?'}`"
        )
    else:
        lines.append("- Last harvest: none")
    return "\n".join(lines)


def render_learn_pending() -> str:
    pending = [p for p in list_proposals() if p.get("approval_state") == "pending"]
    if not pending:
        return "**Codex Learn Pending**\nNo pending Codex learning proposals."
    lines = [f"**Codex Learn Pending** ({len(pending)})"]
    for proposal in pending:
        lines.append(
            "- "
            f"`{proposal['id']}` {proposal.get('subsystem')} "
            f"confidence={float(proposal.get('confidence') or 0):.2f} "
            f"approval={proposal.get('approval_state')} "
            f"repo=`{proposal.get('repo') or '?'}` branch=`{proposal.get('branch') or '?'}`"
        )
        summary = str(proposal.get("summary") or "")
        if summary:
            lines.append(f"  {summary[:180]}")
        command = str(proposal.get("source_command") or "")
        if command:
            lines.append(f"  command: `{_one_line(command, 180)}`")
    lines.append("")
    lines.append("Apply: `/codex learn apply <id|all>`   Discard: `/codex learn discard <id|all>`")
    return "\n".join(lines)


def apply_learning(target: str, config: Mapping[str, Any] | None = None) -> str:
    target = (target or "").strip()
    if not target:
        return "Usage: `/codex learn apply <id|all>`"
    learning = load_learning_config(config)
    proposals = _select_proposals(target)
    if not proposals:
        return f"No pending Codex learning proposal with id `{target}`."

    applied = 0
    failed: list[str] = []
    for proposal in proposals:
        ok, message = _apply_one_proposal(proposal, learning)
        if ok:
            applied += 1
        else:
            failed.append(f"{proposal['id']}: {message}")

    lines = [f"Applied {applied} Codex learning proposal(s)."]
    if failed:
        lines.append("Failed:")
        lines.extend(f"  {item}" for item in failed)
    return "\n".join(lines)


def discard_learning(target: str) -> str:
    target = (target or "").strip()
    if not target:
        return "Usage: `/codex learn discard <id|all>`"
    proposals = _select_proposals(target)
    if not proposals:
        return f"No pending Codex learning proposal with id `{target}`."

    discarded = 0
    for proposal in proposals:
        subsystem = str(proposal.get("subsystem") or "")
        pending_id = str(proposal.get("pending_id") or "")
        try:
            from tools import write_approval as wa

            wa.discard_pending(subsystem, pending_id)
        except Exception:
            pass
        proposal["status"] = "discarded"
        proposal["approval_state"] = "discarded"
        proposal["discarded_at"] = time.time()
        proposal["updated_at"] = time.time()
        save_proposal(proposal)
        discarded += 1
    return f"Discarded {discarded} Codex learning proposal(s)."


def count_pending_proposals() -> int:
    return sum(1 for proposal in list_proposals() if proposal.get("approval_state") == "pending")


def list_packets() -> list[dict[str, Any]]:
    packets = _read_records(_packets_dir())
    packets.sort(key=lambda p: float(p.get("created_at") or 0))
    return packets


def list_proposals() -> list[dict[str, Any]]:
    proposals = [_with_approval_state(p) for p in _read_records(_proposals_dir())]
    proposals.sort(key=lambda p: float(p.get("created_at") or 0))
    return proposals


def load_packet(packet_id: str) -> Optional[dict[str, Any]]:
    return _read_json(_packets_dir() / f"{packet_id}.json")


def save_packet(packet: dict[str, Any]) -> None:
    _write_json(_packets_dir() / f"{packet['id']}.json", packet)


def packet_exists(packet_id: str) -> bool:
    return (_packets_dir() / f"{packet_id}.json").exists()


def save_proposal(proposal: dict[str, Any]) -> None:
    _write_json(_proposals_dir() / f"{proposal['id']}.json", proposal)


def load_proposal(proposal_id: str) -> Optional[dict[str, Any]]:
    return _read_json(_proposals_dir() / f"{proposal_id}.json")


def _run_hermes_reviewer(packet: dict[str, Any], learning: LearningConfig) -> None:
    """Run a standalone Hermes agent with only memory/skills writes available."""
    from run_agent import AIAgent
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore
    from tools.terminal_tool import set_approval_callback

    def _auto_deny(command: str, description: str, **_kwargs: Any) -> str:
        logger.warning(
            "Codex learning review auto-denied dangerous command: %s (%s)",
            command,
            description,
        )
        return "deny"

    review_agent = None
    try:
        set_approval_callback(_auto_deny)
        with open(os.devnull, "w", encoding="utf-8") as devnull, (
            contextlib.redirect_stdout(devnull)
        ), contextlib.redirect_stderr(devnull):
            store = MemoryStore()
            store.load_from_disk()
            review_agent = AIAgent(
                model=learning.review_model,
                max_iterations=12,
                quiet_mode=True,
                enabled_toolsets=["memory", "skills"],
                skip_memory=True,
            )
            review_agent._memory_write_origin = "codex_learning"
            review_agent._memory_write_context = "codex_learning"
            review_agent._memory_store = store
            review_agent._memory_enabled = True
            review_agent._user_profile_enabled = True
            review_agent.suppress_status_output = True

            from hermes_cli.plugins import (
                clear_thread_tool_whitelist,
                set_thread_tool_whitelist,
            )
            from model_tools import get_tool_definitions

            whitelist = {
                tool["function"]["name"]
                for tool in get_tool_definitions(
                    enabled_toolsets=["memory", "skills"],
                    quiet_mode=True,
                )
            }
            set_thread_tool_whitelist(
                whitelist,
                deny_msg_fmt=(
                    "Codex learning review denied non-whitelisted tool: "
                    "{tool_name}. Only memory/skill tools are allowed."
                ),
            )
            try:
                with wa.force_write_approval(wa.MEMORY, wa.SKILLS):
                    review_agent.run_conversation(
                        user_message=_review_prompt(packet, learning),
                        conversation_history=[],
                    )
            finally:
                clear_thread_tool_whitelist()
    finally:
        if review_agent is not None:
            with contextlib.suppress(Exception):
                review_agent.shutdown_memory_provider()
            with contextlib.suppress(Exception):
                review_agent.close()
        with contextlib.suppress(Exception):
            set_approval_callback(None)


def _review_prompt(packet: dict[str, Any], learning: LearningConfig) -> str:
    compact = {
        "command": packet.get("command"),
        "cwd": packet.get("cwd"),
        "repo": packet.get("repo"),
        "branch": packet.get("branch"),
        "exit_code": packet.get("exit_code"),
        "output_tail": packet.get("output_tail"),
        "cockpit_status": packet.get("cockpit_status"),
    }
    return (
        "You are Hermes' Codex completion learning reviewer.\n\n"
        "Review this completed `/codex launch` packet and decide whether any "
        "durable memory or skill update is justified. Only save stable lessons "
        "that should affect future work. Do not save transient environment "
        "failures, one-off task narrative, or negative claims that a tool is "
        "broken. If there is nothing durable, reply exactly `Nothing to save.`\n\n"
        "If a user or workflow preference should persist, use the memory tool. "
        "If a reusable procedure, pitfall, verification pattern, or skill "
        "improvement should persist, use skill_manage. You can only use memory "
        "and skill-management tools, and every write will be staged for human "
        "approval. Only call a write tool when your confidence is at least "
        f"{learning.min_confidence:.2f}.\n\n"
        "Learning packet JSON:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def _proposal_from_pending(
    packet: dict[str, Any],
    subsystem: str,
    pending_id: str,
    pending: Optional[dict[str, Any]],
) -> dict[str, Any]:
    now = time.time()
    return {
        "id": f"learn_{uuid.uuid4().hex[:8]}",
        "packet_id": packet["id"],
        "process_id": packet.get("process_id"),
        "subsystem": subsystem,
        "pending_id": pending_id,
        "status": "staged",
        "approval_state": "pending",
        "confidence": 1.0,
        "rationale": "Hermes staged this write during Codex completion review.",
        "summary": (pending or {}).get("summary", ""),
        "repo": packet.get("repo", ""),
        "branch": packet.get("branch", ""),
        "source_command": packet.get("command", ""),
        "created_at": now,
        "updated_at": now,
    }


def _apply_one_proposal(
    proposal: dict[str, Any],
    learning: LearningConfig,
) -> tuple[bool, str]:
    proposal = _with_approval_state(proposal)
    if proposal.get("approval_state") != "pending":
        return False, f"approval state is {proposal.get('approval_state')}"

    confidence = float(proposal.get("confidence") or 0)
    if confidence < learning.min_confidence:
        return False, (
            f"confidence {confidence:.2f} is below min_confidence "
            f"{learning.min_confidence:.2f}"
        )

    subsystem = str(proposal.get("subsystem") or "")
    pending_id = str(proposal.get("pending_id") or "")
    try:
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa

        if wa.get_pending(subsystem, pending_id) is None:
            return False, f"pending {subsystem} write {pending_id!r} is missing"
        memory_store = None
        if subsystem == wa.MEMORY:
            from tools.memory_tool import MemoryStore

            memory_store = MemoryStore()
            memory_store.load_from_disk()
        handle_pending_subcommand(
            subsystem,
            ["approve", pending_id],
            memory_store=memory_store,
        )
        if wa.get_pending(subsystem, pending_id) is not None:
            return False, "approval did not remove the pending write"
    except Exception as exc:
        return False, str(exc)

    proposal["status"] = "applied"
    proposal["approval_state"] = "applied"
    proposal["applied_at"] = time.time()
    proposal["updated_at"] = time.time()
    save_proposal(proposal)
    return True, "applied"


def _auto_promote_if_configured(proposal_ids: list[str], learning: LearningConfig) -> None:
    for proposal_id in proposal_ids:
        proposal = load_proposal(proposal_id)
        if not proposal:
            continue
        subsystem = proposal.get("subsystem")
        enabled = (
            subsystem == "memory"
            and learning.auto_promote_memory
            or subsystem == "skills"
            and learning.auto_promote_skills
        )
        if not enabled:
            continue
        ok, message = _apply_one_proposal(proposal, learning)
        if not ok:
            proposal["auto_promote_error"] = message
            proposal["updated_at"] = time.time()
            save_proposal(proposal)


def _curator_scope() -> str:
    try:
        from tools.skill_usage import curator_scope

        return curator_scope()
    except Exception:
        return "all_usable"


def _select_proposals(target: str) -> list[dict[str, Any]]:
    proposals = [p for p in list_proposals() if p.get("approval_state") == "pending"]
    if target.lower() == "all":
        return proposals
    return [p for p in proposals if p.get("id") == target]


def _pending_id_snapshot() -> dict[str, set[str]]:
    return {
        "memory": {r.get("id") for r in _list_pending("memory") if r.get("id")},
        "skills": {r.get("id") for r in _list_pending("skills") if r.get("id")},
    }


def _list_pending(subsystem: str) -> list[dict[str, Any]]:
    try:
        from tools import write_approval as wa

        return wa.list_pending(subsystem)
    except Exception:
        return []


def _get_pending(subsystem: str, pending_id: str) -> Optional[dict[str, Any]]:
    try:
        from tools import write_approval as wa

        return wa.get_pending(subsystem, pending_id)
    except Exception:
        return None


def _with_approval_state(proposal: dict[str, Any]) -> dict[str, Any]:
    proposal = dict(proposal)
    if proposal.get("status") in {"applied", "discarded", "failed"}:
        proposal["approval_state"] = proposal.get("status")
        return proposal
    pending = _get_pending(
        str(proposal.get("subsystem") or ""),
        str(proposal.get("pending_id") or ""),
    )
    proposal["approval_state"] = "pending" if pending else "missing"
    return proposal


def _queue_stage_notification(packet: dict[str, Any], proposal_ids: list[str]) -> None:
    message = (
        f"Codex learning staged {len(proposal_ids)} proposal(s) from "
        f"{packet.get('process_id')}. Review with `/codex learn pending`."
    )
    try:
        from tools.process_registry import process_registry

        process_registry.completion_queue.put(
            {
                "type": "codex_learning_staged",
                "session_id": packet.get("process_id", ""),
                "session_key": packet.get("session_key", ""),
                "command": packet.get("command", ""),
                "message": message,
                "proposal_ids": proposal_ids,
            }
        )
    except Exception:
        logger.info(message)


def _cockpit_status_snapshot(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cockpit = config.get("codex_cockpit", {}) if isinstance(config, Mapping) else {}
    if not isinstance(cockpit, Mapping):
        cockpit = {}
    helper = cockpit.get("context_helper", {})
    if not isinstance(helper, Mapping):
        helper = {}
    return {
        "enabled": _as_bool(cockpit.get("enabled", True)),
        "driver": str(cockpit.get("driver") or "codex_app_server"),
        "default_model": str(cockpit.get("default_model") or "gpt-5.5"),
        "context_helper_enabled": _as_bool(helper.get("enabled", False)),
        "pending_learning": count_pending_proposals(),
    }


def _resolve_repo_root(cwd: str) -> str:
    if not cwd:
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(Path(proc.stdout.strip()).resolve())


def _git_branch(cwd: str) -> str:
    if not cwd:
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    branch = proc.stdout.strip()
    return "" if branch == "HEAD" else branch


def _extract_git_repo_from_command(command: str) -> str:
    tokens = _shell_tokens(command)
    for idx, token in enumerate(tokens[:-2]):
        if token == "git" and tokens[idx + 1] == "-C":
            return tokens[idx + 2]
    return ""


def _extract_branch_from_command(command: str) -> str:
    tokens = _shell_tokens(command)
    for idx, token in enumerate(tokens):
        if token != "worktree":
            continue
        if idx + 1 >= len(tokens) or tokens[idx + 1] != "add":
            continue
        for j in range(idx + 2, len(tokens) - 1):
            if tokens[j] == "&&":
                break
            if tokens[j] == "-b":
                return tokens[j + 1]
    return ""


def _extract_codex_worktree(command: str) -> str:
    tokens = _shell_tokens(command)
    for idx, token in enumerate(tokens):
        if token == "codex" and idx + 1 < len(tokens) and tokens[idx + 1] == "exec":
            for j in range(idx + 2, len(tokens) - 1):
                if tokens[j] == "-C":
                    return tokens[j + 1]
    return ""


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _packet_id(process_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", process_id).strip("-")
    return f"pkt_{safe or uuid.uuid4().hex[:12]}"


def _store_root() -> Path:
    return get_hermes_home() / "codex_learning"


def _packets_dir() -> Path:
    return _store_root() / "packets"


def _proposals_dir() -> Path:
    return _store_root() / "proposals"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for item in path.glob("*.json"):
        record = _read_json(item)
        if isinstance(record, dict):
            records.append(record)
    return records


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        logger.warning("Could not read Codex learning JSON: %s", path)
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    return bool(value)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _confidence(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, parsed))


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
