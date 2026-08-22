"""Send-time paging allowlist for cron chat deliveries.

Cron jobs do not own the paging decision: a job record can be hand-edited or
misconfigured to deliver into a chat, and a recurring job can flood a chat
with notifications the user never asked for. This module enforces an
out-of-band allowlist (a plain text file, independent of job records) at
send time: chat delivery is suppressed unless the job name or id is on the
allowlist, and every suppression is audited to a log plus the withheld
payload saved to disk.

The gate is config-gated via ``cron.paging`` (default disabled): when
disabled it is a no-op and delivery behavior is unchanged. The allowlist file
is re-read on every check so an out-of-band edit takes effect immediately and
jobs cannot cache paging permission.
"""

from datetime import datetime, timezone
from pathlib import Path
import os
import re

# Chat transports are the paging surfaces this gate controls. Unknown or
# non-chat transports are never suppressed, so the gate cannot change
# delivery behavior for file/webhook-only adapters.
CHAT_PLATFORMS = {
    "telegram", "discord", "whatsapp", "whatsapp_cloud", "slack", "signal",
    "mattermost", "matrix", "dingtalk", "feishu", "wecom", "weixin",
    "bluebubbles", "qqbot", "yuanbao", "relay", "irc",
}


def _default_settings() -> dict:
    # Default paths live under HERMES_HOME so a fresh install has a sensible,
    # writable location without any configuration.
    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    return {
        "enabled": False,
        "allowlist_path": str(hermes_home / "cron" / "page-allowlist.txt"),
        "audit_log_path": str(hermes_home / "cron" / "alerts" / "delivery-gate.log"),
        "output_root": str(hermes_home / "cron" / "output"),
    }


def _load_settings() -> dict:
    # Read cron.paging from config.yaml on every call so a running scheduler
    # picks up out-of-band changes without a restart (same immediacy contract
    # as the allowlist file itself).
    settings = _default_settings()
    try:
        from hermes_cli.config import load_config

        paging = (load_config().get("cron") or {}).get("paging") or {}
        settings.update({k: v for k, v in paging.items() if v is not None})
    except Exception:
        pass  # Defaults are safe (disabled); never let a config error page.
    return settings


def _utc_timestamp() -> str:
    # UTC with an explicit offset keeps audit records timezone-safe.
    return datetime.now(timezone.utc).isoformat()


def _load_allowlist(path: str) -> set:
    # A missing or unreadable control file fails CLOSED for chat paging so a
    # deployment mistake cannot silently re-enable notification floods.
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


def _is_chat_target(target: dict) -> bool:
    return str(target.get("platform", "")).strip().lower() in CHAT_PLATFORMS


def delivery_gate_check(job, targets, settings=None) -> tuple:
    """Return (allowed, reason). Suppresses only chat deliveries for jobs
    not present in the out-of-band allowlist."""
    settings = settings if settings is not None else _load_settings()
    if not settings.get("enabled", False):
        return True, "gate disabled"
    if not targets or not any(_is_chat_target(target) for target in targets):
        return True, "no chat target"
    allowlist = _load_allowlist(str(settings["allowlist_path"]))
    job_name = str(job.get("name", ""))
    job_id = str(job.get("id", ""))
    if job_name in allowlist or job_id in allowlist:
        return True, "allowlisted"
    return False, "not in page-allowlist"


def suppress_and_audit(job, targets, content, settings=None) -> None:
    """Audit a suppressed delivery: one log line plus the withheld payload
    saved under output_root, so operators can inspect what was held back."""
    settings = settings if settings is not None else _load_settings()
    _, reason = delivery_gate_check(job, targets, settings)
    timestamp = _utc_timestamp()
    job_id = str(job.get("id", "unknown"))
    job_name = str(job.get("name", ""))
    deliver = str(job.get("deliver", "local"))
    # One audit line per suppressed delivery result, not one per target.
    target_text = ",".join(
        f"{target.get('platform', '')}:{target.get('chat_id', '')}"
        for target in targets
        if _is_chat_target(target)
    )
    audit_path = Path(str(settings["audit_log_path"]))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # Append rather than overwrite so suppression evidence is durable.
    with audit_path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(
            f"SUPPRESS {timestamp} job={job_id} name={job_name} deliver={deliver} "
            f"target={target_text} reason={reason}\n"
        )
    # Restrict the directory component to a safe filename while retaining the
    # job id for normal ids, preventing a job record from escaping the root.
    safe_job_id = re.sub(r"[^A-Za-z0-9._-]", "_", job_id) or "unknown"
    if safe_job_id in (".", ".."):
        # Dot-only ids would resolve to the output root itself or its parent;
        # prefix them so the payload stays inside output_root.
        safe_job_id = "_" + safe_job_id
    output_dir = Path(str(settings["output_root"])) / safe_job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{timestamp}.md").write_text(str(content), encoding="utf-8")
