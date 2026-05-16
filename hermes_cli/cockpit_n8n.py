"""Read-only Cockpit projection for daily n8n check inventory.

BIF-525: display-safe daily n8n check inventory with an optional live latest-execution
read path. This module must not trigger workflows, send messages, expose
credentials, webhook URLs, channel IDs, execution IDs, or payload blobs, or add
repair/retry controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Literal, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hermes_cli.cockpit import COCKPIT_SCHEMA_VERSION
from hermes_cli.config import get_hermes_home

N8nAuthState = Literal["configured", "not_required", "unknown"]

_N8N_API_BASE = "http://100.126.93.7:5678/api/v1"
_N8N_CREDENTIAL_HELPER = "/Users/marco/.local/bin/get_credential.sh"
_N8N_CREDENTIAL_NAME = "n8n_api_key"
_N8N_TIMEOUT_SECONDS = 3.0
_MAX_SAFE_SUMMARY_CHARS = 240

_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)
_WEBHOOK_PATH_RE = re.compile(r"/webhook(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?", re.IGNORECASE)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SECRET_WORD_RE = re.compile(r"\b(api[-_ ]?key|token|secret|password)\b\s*[:=]?\s*[^\s,;]{3,}", re.IGNORECASE)
_CHANNEL_RE = re.compile(r"\b[CGD][A-Z0-9]{8,}\b")
_CREDENTIAL_PHRASE_RE = re.compile(r"\bcredential(?:s| name)?\s+[^,.;\n]+", re.IGNORECASE)
_LONG_ID_RE = re.compile(r"\b[A-Za-z0-9_-]{24,}\b")
_LOCAL_PATH_RE = re.compile(r"(?<!\w)/(?:Users|opt|var|tmp|private|Volumes)/[^\s,;:\"']+")
_TRACEBACK_OR_COMMAND_RE = re.compile(r"\b(command failed|traceback|file \"|line \d+|exception|stack trace)\b", re.IGNORECASE)


@dataclass(frozen=True)
class N8nDailyCheck:
    id: str
    name: str
    status: str
    last_run: str
    next_schedule: str
    delivery: str
    auth: N8nAuthState
    action_needed: str
    summary: str

    def to_display_dict(self) -> dict[str, str]:
        row = asdict(self)
        row.update(
            {
                "live_source": "fixture_fallback",
                "execution_status": "unknown",
                "last_started": "",
                "last_completed": "",
                "output_summary": self.summary,
            }
        )
        return row


class N8nClientProtocol(Protocol):
    def list_workflows(self) -> list[Mapping[str, Any]]: ...

    def latest_execution(self, workflow_id: str) -> Mapping[str, Any] | None: ...


class N8nApiClient:
    """Minimal read-only n8n API client for workflow/execution status."""

    def __init__(
        self,
        *,
        base_url: str = _N8N_API_BASE,
        credential_helper: str = _N8N_CREDENTIAL_HELPER,
        credential_name: str = _N8N_CREDENTIAL_NAME,
        timeout: float = _N8N_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credential_helper = credential_helper
        self.credential_name = credential_name
        self.timeout = timeout
        self._api_key: str | None = None

    def _get_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        result = subprocess.run(
            [self.credential_helper, self.credential_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        self._api_key = result.stdout.strip()
        if not self._api_key:
            raise RuntimeError("n8n API key helper returned an empty value")
        return self._api_key

    def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self.base_url}{path}{query}",
            headers={"X-N8N-API-KEY": self._get_api_key(), "Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - fixed configured LAN n8n API base
                body = response.read(1_000_000)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"n8n API read failed: {exc}") from exc
        return json.loads(body.decode("utf-8"))

    def list_workflows(self) -> list[Mapping[str, Any]]:
        payload = self._get_json("/workflows", {"limit": 250})
        data = payload.get("data", payload)
        return data if isinstance(data, list) else []

    def latest_execution(self, workflow_id: str) -> Mapping[str, Any] | None:
        payload = self._get_json(
            "/executions",
            {"workflowId": workflow_id, "limit": 1, "includeData": "true"},
        )
        data = payload.get("data", payload)
        if isinstance(data, list) and data:
            first = data[0]
            return first if isinstance(first, Mapping) else None
        return None


# Fixture derived from the BIF-525 inventory / BIF-524 blueprint fields. Keep
# values intentionally high-level and secret-free: no webhook URLs, execution
# IDs, credential names, recipient IDs, message payloads, or repair controls.
_N8N_DAILY_CHECKS: tuple[N8nDailyCheck, ...] = (
    N8nDailyCheck(
        id="morning-briefing",
        name="Morning Briefing",
        status="observed",
        last_run="daily morning window",
        next_schedule="next daily morning window",
        delivery="cockpit read-only summary",
        auth="configured",
        action_needed="review when briefing is missing or stale",
        summary="Daily agenda/context briefing output is expected once each morning; Cockpit only displays the latest observed check state.",
    ),
    N8nDailyCheck(
        id="workflow-health-daily-report",
        name="Workflow Health Daily Report",
        status="observed",
        last_run="daily health window",
        next_schedule="next daily health window",
        delivery="cockpit read-only summary",
        auth="configured",
        action_needed="review failed or stale workflow rows",
        summary="Daily workflow health report summarizes recent n8n workflow status without exposing execution payloads.",
    ),
    N8nDailyCheck(
        id="auto-remediation-monitor",
        name="Auto-Remediation Monitor",
        status="monitoring",
        last_run="daily monitor window",
        next_schedule="next daily monitor window",
        delivery="cockpit read-only summary",
        auth="configured",
        action_needed="manual review only; no repair buttons in Cockpit",
        summary="Auto-remediation monitor is represented as status metadata only; Cockpit never launches retries or repairs.",
    ),
    N8nDailyCheck(
        id="immich-nightly-sync-monitor",
        name="Immich Nightly Sync Monitor",
        status="observed",
        last_run="nightly sync window",
        next_schedule="next nightly sync window",
        delivery="cockpit read-only summary",
        auth="configured",
        action_needed="review if sync is stale or reports attention needed",
        summary="Nightly Immich sync monitor output is shown as a concise health row with no attachments or media payloads.",
    ),
    N8nDailyCheck(
        id="obsidian-inbox-processor",
        name="Obsidian Inbox Processor",
        status="observed",
        last_run="daily inbox processing window",
        next_schedule="next daily inbox processing window",
        delivery="cockpit read-only summary",
        auth="configured",
        action_needed="review unprocessed inbox items if action-needed is non-empty",
        summary="Obsidian inbox processing check reports completion/attention state only; note content stays out of this view.",
    ),
    N8nDailyCheck(
        id="alexa-bring-sync",
        name="Alexa Bring Sync",
        status="observed",
        last_run="daily sync window",
        next_schedule="next daily sync window",
        delivery="cockpit read-only summary",
        auth="configured",
        action_needed="review list sync drift if reported",
        summary="Alexa Bring Sync status is displayed as a daily integration check without account identifiers or list contents.",
    ),
    N8nDailyCheck(
        id="n8n-nightly-workflow-backup",
        name="n8n Nightly Workflow Backup",
        status="observed",
        last_run="nightly backup window",
        next_schedule="next nightly backup window",
        delivery="cockpit read-only summary",
        auth="configured",
        action_needed="review backup freshness if stale",
        summary="Nightly workflow backup check summarizes backup freshness only; no exported workflow files are exposed.",
    ),
)


def _sanitize_text(value: Any, *, max_chars: int = _MAX_SAFE_SUMMARY_CHARS) -> str:
    text = "" if value is None else str(value)
    text = _URL_RE.sub("[redacted-url]", text)
    text = _WEBHOOK_PATH_RE.sub("/[redacted-route]", text)
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _SECRET_WORD_RE.sub(lambda m: f"{m.group(1)} [redacted]", text)
    text = _CHANNEL_RE.sub("[redacted-channel]", text)
    text = _CREDENTIAL_PHRASE_RE.sub("credential [redacted]", text)
    text = _LOCAL_PATH_RE.sub("[redacted-path]", text)
    text = _LONG_ID_RE.sub("[redacted-id]", text)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _safe_error_summary(value: Any) -> str:
    raw = "" if value is None else str(value)
    if _LOCAL_PATH_RE.search(raw) or _TRACEBACK_OR_COMMAND_RE.search(raw):
        return "Execution error; command and local diagnostic details hidden."
    return _sanitize_text(raw)


def _execution_status(execution: Mapping[str, Any]) -> str:
    status = execution.get("status")
    if isinstance(status, str) and status:
        return _sanitize_text(status, max_chars=40).lower()
    if execution.get("finished") is True:
        return "success"
    if execution.get("stoppedAt") and execution.get("finished") is False:
        return "error"
    return "running" if execution.get("startedAt") else "unknown"


def _execution_output_summary(execution: Mapping[str, Any]) -> str:
    data = execution.get("data")
    result_data = data.get("resultData") if isinstance(data, Mapping) else None
    if not isinstance(result_data, Mapping):
        return "Latest execution metadata available; output payload not displayed."

    parts: list[str] = []
    error = result_data.get("error")
    if isinstance(error, Mapping):
        message = error.get("message") or error.get("description") or error.get("name")
        if message:
            parts.append(f"Error: {_safe_error_summary(message)}")
    elif error:
        parts.append(f"Error: {_safe_error_summary(error)}")

    last_node = result_data.get("lastNodeExecuted")
    if last_node:
        parts.append(f"Last node: {_sanitize_text(last_node, max_chars=80)}")

    run_data = result_data.get("runData")
    if isinstance(run_data, Mapping):
        node_count = len(run_data)
        if node_count:
            parts.append(f"{node_count} node(s) reported execution data.")

    if not parts:
        return "Latest execution completed; detailed output payload hidden."
    return _sanitize_text(" ".join(parts))


def _workflow_id_by_name(workflows: list[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for workflow in workflows:
        name = workflow.get("name")
        workflow_id = workflow.get("id")
        if isinstance(name, str) and workflow_id is not None:
            key = _normalized_workflow_name(name)
            if key:
                result[key] = str(workflow_id)
    return result


def _normalized_workflow_name(value: Any) -> str:
    text = _sanitize_text(value, max_chars=120).casefold()
    text = text.replace("→", " ").replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _local_inventory_rows() -> tuple[dict[str, Mapping[str, Any]], float | None]:
    hermes_home = Path(get_hermes_home())
    candidates = [
        hermes_home / "state" / "n8n_workflow_inventory_summary.json",
        (
            hermes_home.parents[1] / "state" / "n8n_workflow_inventory_summary.json"
            if len(hermes_home.parents) >= 2 and hermes_home.parent.name == "profiles"
            else hermes_home / "state" / "n8n_workflow_inventory_summary.json"
        ),
        Path.home() / ".hermes" / "state" / "n8n_workflow_inventory_summary.json",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            return {}, mtime
        rows: dict[str, Mapping[str, Any]] = {}
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            key = _normalized_workflow_name(item.get("name"))
            if key:
                rows[key] = item
        return rows, mtime
    return {}, None


def _local_inventory_check_rows(now: float) -> tuple[list[dict[str, str]], bool, bool, float | None]:
    inventory, inventory_mtime = _local_inventory_rows()
    if not inventory:
        return [check.to_display_dict() for check in _N8N_DAILY_CHECKS], True, True, inventory_mtime

    stale = inventory_mtime is None or now - inventory_mtime > 36 * 60 * 60
    rows: list[dict[str, str]] = []
    for check in _N8N_DAILY_CHECKS:
        row = check.to_display_dict()
        item = inventory.get(_normalized_workflow_name(check.name))
        if item is None:
            row.update(
                {
                    "live_source": "local_inventory_missing_workflow",
                    "execution_status": "unknown",
                    "output_summary": "Workflow was not found in the local n8n inventory; fixture metadata shown.",
                }
            )
            rows.append(row)
            continue

        try:
            error_count = int(item.get("errors") or 0)
        except (TypeError, ValueError):
            error_count = 0
        latest_error = item.get("latest_error")
        has_error = error_count > 0 or bool(latest_error)
        status = "error" if has_error else "success"
        updated_at = _sanitize_text(item.get("updatedAt") or "", max_chars=80)
        if has_error:
            if latest_error:
                output_summary = f"Local inventory reports {error_count} recent error(s). Latest error: {_safe_error_summary(latest_error)}"
            else:
                output_summary = f"Local inventory reports {error_count} recent error(s); detailed error payload hidden."
        else:
            output_summary = "Local inventory reports no recent errors for this workflow."
        row.update(
            {
                "status": status,
                "last_run": updated_at or check.last_run,
                "action_needed": "review local n8n inventory error signal" if has_error else check.action_needed,
                "live_source": "local_inventory",
                "execution_status": status,
                "last_started": "",
                "last_completed": updated_at,
                "output_summary": _sanitize_text(output_summary),
            }
        )
        rows.append(row)
    return rows, False, stale, inventory_mtime


def _live_check_rows(client: N8nClientProtocol) -> list[dict[str, str]]:
    workflow_ids = _workflow_id_by_name(client.list_workflows())
    rows: list[dict[str, str]] = []
    for check in _N8N_DAILY_CHECKS:
        row = check.to_display_dict()
        workflow_id = workflow_ids.get(_normalized_workflow_name(check.name))
        if not workflow_id:
            row.update(
                {
                    "live_source": "live_missing_workflow",
                    "execution_status": "unknown",
                    "output_summary": "Workflow was not found in the live n8n workflow list; fixture metadata shown.",
                }
            )
            rows.append(row)
            continue

        execution = client.latest_execution(workflow_id)
        if not execution:
            row.update(
                {
                    "live_source": "live_no_execution",
                    "execution_status": "unknown",
                    "output_summary": "Workflow found in n8n; no latest execution was returned.",
                }
            )
            rows.append(row)
            continue

        status = _execution_status(execution)
        started = _sanitize_text(execution.get("startedAt") or execution.get("createdAt") or "", max_chars=80)
        completed = _sanitize_text(execution.get("stoppedAt") or execution.get("finishedAt") or "", max_chars=80)
        row.update(
            {
                "status": status,
                "last_run": started or check.last_run,
                "action_needed": "review latest n8n execution status" if status in {"error", "failed", "crashed"} else check.action_needed,
                "live_source": "live",
                "execution_status": status,
                "last_started": started,
                "last_completed": completed,
                "output_summary": _execution_output_summary(execution),
            }
        )
        rows.append(row)
    return rows


def get_n8n_daily_checks_payload(
    *,
    now: float | None = None,
    n8n_client: N8nClientProtocol | None = None,
) -> dict[str, object]:
    """Return the display-safe, read-only daily n8n checks payload."""
    generated_at = time.time() if now is None else float(now)
    client = n8n_client or N8nApiClient()
    live_error = ""
    try:
        checks = _live_check_rows(client)
        source = "n8n_live_latest_execution"
        live = True
        fallback = False
        stale = False
    except Exception as exc:  # read-only projection must degrade to fixture
        checks = [check.to_display_dict() for check in _N8N_DAILY_CHECKS]
        source = "fixture_bif_525_inventory"
        live = False
        fallback = True
        stale = True
        live_error = _sanitize_text(exc, max_chars=160)

    return {
        "schema_version": COCKPIT_SCHEMA_VERSION,
        "read_only": True,
        "source": source,
        "generated_at": generated_at,
        "actions_enabled": False,
        "external_delivery_enabled": False,
        "live": live,
        "fallback": fallback,
        "stale": stale,
        "live_error": live_error,
        "checks": checks,
    }


def get_n8n_daily_checks_local_payload(*, now: float | None = None) -> dict[str, object]:
    """Return the local-only, display-safe daily n8n checks payload.

    This path is intentionally narrower than ``get_n8n_daily_checks_payload``:
    it never instantiates ``N8nApiClient``, never asks the credential helper for
    secrets, and never performs workflow/execution live reads. Automation
    Health uses this function because that cockpit section is constrained to
    safe local persisted/sanitized sources only.
    """
    generated_at = time.time() if now is None else float(now)
    checks, fallback, stale, inventory_mtime = _local_inventory_check_rows(generated_at)
    return {
        "schema_version": COCKPIT_SCHEMA_VERSION,
        "read_only": True,
        "source": "fixture_bif_525_inventory" if fallback else "local_n8n_inventory_summary",
        "generated_at": generated_at,
        "actions_enabled": False,
        "external_delivery_enabled": False,
        "live": False,
        "fallback": fallback,
        "stale": stale,
        "live_error": "",
        "inventory_checked_at": inventory_mtime or "",
        "checks": checks,
    }
