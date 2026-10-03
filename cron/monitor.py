"""Monitor-mode cron support — hash-suppressed change detection.

A monitor job attaches a cheap source (``monitor_script`` / ``monitor_url`` / ``monitor_tool``)
to an LLM cron job. Each tick runs the source FIRST and hashes its EXACT output bytes (no
timestamp/whitespace normalization — scripts must emit stable output) against the hash from the
last agent-triggering tick: unchanged → agent run suppressed (silent ``no_change`` run);
changed/first run → a "MONITOR CHANGE DETECTED" block (capped unified diff + new output) is
injected into the prompt; source failure → an ERROR, never a change, and the stored hash is left
untouched. State: ``job["monitor_state"]`` in jobs.json (hash + last_changed_at) and
``OUTPUT_DIR/<job_id>/monitor_last_output.txt`` (for the diff).

``monitor_tool`` dispatches ONE registered tool (a connector such as
``connectors__gmail__search``, an MCP tool, ``web_extract`` …) with fixed args and no LLM, so a
job can watch an inbox, a tracker or an API for an event and wake the agent only when the
result changes: an event trigger that spends nothing while it watches. Hosted connectors have no
CLI, so this is the only token-free way to poll them.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Prompt-injection caps: unified diff, and new-output block (mirrors the 8k context_from truncation
# in cron/scheduler.py). Then bounded-GET limits for monitor_url sources.
MAX_DIFF_CHARS = 4000
MAX_OUTPUT_CHARS = 8000
URL_TIMEOUT_SECONDS = 30
MAX_URL_BYTES = 262_144  # 256 KiB

_SNAPSHOT_FILENAME = "monitor_last_output.txt"


@dataclass
class MonitorOutcome:
    """Result of one monitor-source evaluation."""

    ok: bool
    changed: bool = False
    first_run: bool = False
    context_block: Optional[str] = None
    error: Optional[str] = None


def hash_monitor_output(output: str) -> str:
    """Hash the monitor output as exact UTF-8 bytes (no normalization)."""
    return hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()


def build_monitor_diff(old: str, new: str) -> str:
    """Unified diff of old vs new monitor output, capped at MAX_DIFF_CHARS."""
    diff = "\n".join(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), fromfile="previous", tofile="current", lineterm="",
        )
    )
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated]"
    return diff


def _snapshot_path(job_id: str):
    from cron.jobs import _job_output_dir

    return _job_output_dir(job_id) / _SNAPSHOT_FILENAME


def _read_last_output(job_id: str) -> str:
    try:
        path = _snapshot_path(job_id)
        if path.exists():
            return path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("Monitor: failed to read last output for %r: %s", job_id, exc)
    return ""


def _write_last_output(job_id: str, output: str) -> None:
    try:
        from cron.jobs import _ensure_cron_dir

        path = _snapshot_path(job_id)
        _ensure_cron_dir(path.parent)
        path.write_text(output, encoding="utf-8")
    except Exception as exc:
        logger.warning("Monitor: failed to persist last output for %r: %s", job_id, exc)


def _fetch_monitor_url(url: str) -> tuple[bool, str]:
    """Bounded GET of a monitor URL. Returns (ok, body-or-error)."""
    import urllib.request

    if not str(url).lower().startswith(("http://", "https://")):
        return False, f"monitor_url must be http(s): {url!r}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-cron-monitor"})
        with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as resp:  # nosec B310 — scheme checked above
            body = resp.read(MAX_URL_BYTES + 1)
        return True, body[:MAX_URL_BYTES].decode("utf-8", errors="replace")
    except Exception as exc:
        return False, f"monitor_url fetch failed: {exc}"


def _field(job: dict, key: str) -> str:
    value = job.get(key) or ""
    return value.strip() if isinstance(value, str) else ""


# ---- monitor_tool: dispatch one registered tool with fixed args, no LLM ------------------------

MONITOR_TOOL_PREFIX = "tool:"


def parse_monitor_tool_spec(text: str) -> dict:
    """``"<tool_name> {json args}"`` (the ``tool:`` prefix optional) → ``{"name", "args"}``.

    Args are optional and must be a JSON object; anything else raises ValueError."""
    value = str(text or "").strip()
    if value.lower().startswith(MONITOR_TOOL_PREFIX):
        value = value[len(MONITOR_TOOL_PREFIX):].strip()
    name, _, raw_args = value.partition(" ")
    name = name.strip()
    if not name:
        raise ValueError("monitor_tool needs a tool name: 'tool:<tool_name> {\"arg\": ...}'.")
    raw_args = raw_args.strip()
    if not raw_args:
        return {"name": name, "args": {}}
    try:
        args = json.loads(raw_args)
    except ValueError as exc:
        raise ValueError(f"monitor_tool args for {name!r} must be a JSON object: {exc}") from None
    if not isinstance(args, dict):
        raise ValueError(f"monitor_tool args for {name!r} must be a JSON object, got {type(args).__name__}.")
    return {"name": name, "args": args}


def normalize_monitor_tool(value: Any) -> Optional[dict]:
    """Stored shape for a job's ``monitor_tool``: ``{"name": str, "args": dict}`` or None.

    Accepts the string grammar of :func:`parse_monitor_tool_spec` or a dict; empty clears."""
    if value in (None, "", {}, False):
        return None
    if isinstance(value, str):
        return parse_monitor_tool_spec(value)
    if isinstance(value, dict):
        name = str(value.get("name") or "").strip()
        if not name:
            raise ValueError("monitor_tool.name is required.")
        args = value.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError(f"monitor_tool.args for {name!r} must be a JSON object.")
        return {"name": name, "args": args}
    raise ValueError("monitor_tool must be 'tool:<name> {json}' or {\"name\": ..., \"args\": {...}}.")


def monitor_tool_display(spec: Any) -> str:
    """One-line ``tool:<name> {args}`` rendering for listings; '' when not a monitor_tool job."""
    if not isinstance(spec, dict) or not spec.get("name"):
        return ""
    args = spec.get("args") or {}
    suffix = f" {json.dumps(args, sort_keys=True)}" if args else ""
    return f"{MONITOR_TOOL_PREFIX}{spec['name']}{suffix}"


def _load_cron_cfg() -> dict:
    """config.yaml as the scheduler sees it (effective view); {} when absent or unreadable."""
    try:
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "config.yaml"
        return load_user_config_effective(path) if path.exists() else {}
    except Exception as exc:
        logger.warning("Monitor: failed to load config.yaml, using defaults: %s", exc)
        return {}


def _run_monitor_tool(job: dict, spec: dict) -> tuple[bool, str]:
    """Dispatch the monitor tool through the same hooks/approval/toolset path the cron agent
    uses, so it can reach connectors and MCP servers but never widens what a cron run may do.

    A tool error (``{"error": ...}``, headless approval refusal included) is a source FAILURE,
    never a change; JSON results are re-serialized with sorted keys so key order cannot fake a
    change."""
    from cron.scheduler import _init_cron_mcp_tools, _resolve_cron_disabled_toolsets
    from model_tools import handle_function_call

    job_id = str(job.get("id") or "")
    name = str(spec.get("name") or "")
    args = dict(spec.get("args") or {})
    _init_cron_mcp_tools(job_id)  # idempotent; MCP tools only exist in the registry after this
    disabled = _resolve_cron_disabled_toolsets(_load_cron_cfg())
    from tools.registry import registry

    toolset = registry.get_toolset_for_tool(name)
    if toolset and toolset in disabled:
        # handle_function_call only scopes the search bridge by toolset; a cron run's denylist
        # (messaging/clarify/cronjob + agent.disabled_toolsets) has to hold here too.
        return False, f"monitor_tool {name!r} belongs to toolset {toolset!r}, which cron runs may not use."
    # Fresh task_id per tick: tools keep per-task state (read_file's "unchanged since last read"
    # dedup, browser sessions), which would make an unchanged source LOOK changed on tick two.
    task_id = f"cron_monitor_{job_id}_{uuid.uuid4().hex[:8]}"
    raw = handle_function_call(name, args, task_id=task_id, disabled_toolsets=disabled)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return True, str(raw)
    if isinstance(parsed, dict) and parsed.get("error"):
        return False, f"monitor_tool {name!r} failed: {parsed['error']}"
    return True, json.dumps(parsed, sort_keys=True, indent=2, ensure_ascii=False)


def _run_monitor_source(job: dict) -> tuple[bool, str]:
    """Run the job's monitor source (script, URL or tool). Returns (ok, output)."""
    monitor_script = _field(job, "monitor_script")
    if monitor_script:
        # Same containment + interpreter rules as the existing `script` field.
        from cron.scheduler_script import _run_job_script

        return _run_job_script(monitor_script, workdir=_field(job, "workdir") or None,
                               interpreter=job.get("interpreter"))
    monitor_url = _field(job, "monitor_url")
    if monitor_url:
        return _fetch_monitor_url(monitor_url)
    monitor_tool = job.get("monitor_tool")
    if isinstance(monitor_tool, dict) and monitor_tool.get("name"):
        return _run_monitor_tool(job, monitor_tool)
    return False, "monitor job has neither monitor_script, monitor_url nor monitor_tool"


def job_has_monitor(job: dict) -> bool:
    monitor_tool = job.get("monitor_tool")
    return bool(_field(job, "monitor_script") or _field(job, "monitor_url")
                or (isinstance(monitor_tool, dict) and monitor_tool.get("name")))


def check_monitor(job: dict) -> MonitorOutcome:
    """Run the monitor source and decide whether the agent should run.

    On change (or first run) the new hash + snapshot are persisted BEFORE the agent runs — detection
    time is the state boundary, so a failed agent run doesn't re-alert on the same content forever.
    On failure nothing is persisted.
    """
    job_id = str(job.get("id") or "")
    ok, output = _run_monitor_source(job)
    if not ok:
        return MonitorOutcome(ok=False, error=output)

    new_hash = hash_monitor_output(output)
    raw_state = job.get("monitor_state")
    last_hash = raw_state.get("last_output_hash") if isinstance(raw_state, dict) else None

    if last_hash is not None and new_hash == last_hash:
        return MonitorOutcome(ok=True, changed=False)

    first_run = last_hash is None
    old_output = "" if first_run else _read_last_output(job_id)

    shown_output = output
    if len(shown_output) > MAX_OUTPUT_CHARS:
        shown_output = shown_output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

    current = f"### Current output\n\n```\n{shown_output}\n```"
    if first_run:
        context_block = (
            "## Monitor Baseline (first run)\n\n"
            "This is the first observation of the monitored source — there is "
            "no previous output to diff against.\n\n" + current
        )
    else:
        diff = build_monitor_diff(old_output, output)
        context_block = (
            "## MONITOR CHANGE DETECTED\n\n"
            "The monitored source's output changed since the last run.\n\n"
            f"### Diff (previous → current)\n\n```diff\n{diff}\n```\n\n" + current
        )

    _persist_monitor_state(job_id, new_hash, output)
    return MonitorOutcome(ok=True, changed=True, first_run=first_run, context_block=context_block)


def _persist_monitor_state(job_id: str, new_hash: str, output: str) -> None:
    from cron.jobs import _hermes_now, update_job

    _write_last_output(job_id, output)
    try:
        update_job(
            job_id,
            {
                "monitor_state": {
                    "last_output_hash": new_hash,
                    "last_changed_at": _hermes_now().isoformat(),
                }
            },
        )
    except Exception as exc:
        logger.warning("Monitor: failed to persist state for %r: %s", job_id, exc)
