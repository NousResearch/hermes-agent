from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from obsidian.audit_log import ObsidianAdapter


@dataclass(frozen=True)
class Commitment:
    id: str
    source_message: str
    inferred_intent: str
    due_at: str | None
    condition: str | None
    risk_level: str
    next_action: str
    status: str = "open"


def _commitment_id(message: str) -> str:
    return "commitment-" + hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]


def _tomorrow_iso() -> str:
    due = datetime.now(timezone.utc) + timedelta(days=1)
    return due.replace(second=0, microsecond=0).isoformat()


def infer_commitment(message: str) -> Commitment | None:
    text = message.strip()
    if not text:
        return None

    due_at: str | None = None
    condition: str | None = None
    if re.search(r"(明天提醒我|tomorrow remind me|remind me tomorrow)", text, re.IGNORECASE):
        due_at = _tomorrow_iso()
    elif re.search(r"(下週再看|next week)", text, re.IGNORECASE):
        due_at = (datetime.now(timezone.utc) + timedelta(days=7)).replace(second=0, microsecond=0).isoformat()
    elif re.search(r"(之後幫我追蹤|follow up later|track later)", text, re.IGNORECASE):
        condition = "follow_up_requested"
    elif match := re.search(r"如果(.+?)(?:就)?通知我", text):
        condition = match.group(1).strip()
    else:
        return None

    return Commitment(
        id=_commitment_id(text),
        source_message=text,
        inferred_intent="reminder" if due_at else "conditional_follow_up",
        due_at=due_at,
        condition=condition,
        risk_level="low",
        next_action="notify KJ" if due_at else "check condition during heartbeat",
    )


def infer_waiting_for_kj_commitment(
    *,
    user_message: str,
    assistant_response: str,
) -> Commitment | None:
    """Infer a waiting-for-KJ record from Hermes' own assistant response.

    This is deliberately conservative: ordinary questions should not create
    durable follow-ups. We only record when Hermes explicitly asks KJ to provide
    or confirm missing information and implies work will continue afterward.
    """
    response = assistant_response.strip()
    if not response:
        return None

    asks_for_input = re.search(
        r"(請\s*(?:KJ\s*)?(?:提供|補充|確認|回覆|給我|上傳)|"
        r"(?:需要|還需要).{0,24}(?:資料|資訊|照片|文件|確認|決定)|"
        r"(?:provide|send|upload|confirm).{0,40}(?:photo|file|info|information|detail|decision))",
        response,
        re.IGNORECASE,
    )
    continuation = re.search(
        r"(拿到|收到|提供後|確認後|回覆後|之後|我會繼續|再繼續|continue|follow up|after you)",
        response,
        re.IGNORECASE,
    )
    if not asks_for_input or not continuation:
        return None

    source = response
    if user_message.strip():
        source = f"user: {user_message.strip()} | assistant: {response}"

    return Commitment(
        id=_commitment_id(source),
        source_message=source,
        inferred_intent="missing_information_follow_up",
        due_at=None,
        condition="awaiting_kj_input",
        risk_level="low",
        next_action="ask KJ whether the requested information is ready",
        status="waiting_for_kj",
    )


def _entry(commitment: Commitment, cron_job_id: str | None = None) -> str:
    data = asdict(commitment)
    if cron_job_id:
        data["cron_job_id"] = cron_job_id
    lines = ["```yaml"]
    for key, value in data.items():
        if value is None:
            rendered = "null"
        else:
            rendered = str(value).replace("\n", "\\n")
        lines.append(f"{key}: {rendered}")
    lines.append("```")
    return "\n".join(lines)


def create_commitment_record(
    commitment: Commitment,
    obsidian_vault: str | Path,
    *,
    create_cron: bool = False,
) -> Path:
    cron_job_id: str | None = None
    if create_cron and commitment.due_at:
        from cron.jobs import create_job

        job = create_job(
            prompt=f"Commitment due: {commitment.source_message}",
            schedule=commitment.due_at,
            name=f"commitment:{commitment.id}",
            deliver="origin",
            origin={"source": "proactive.commitments", "commitment_id": commitment.id},
            enabled_toolsets=["cronjob"],
        )
        cron_job_id = str(job.get("id", ""))

    adapter = ObsidianAdapter(obsidian_vault)
    ts = commitment.due_at or datetime.now(timezone.utc).isoformat()
    return adapter.append_daily_note("System/Commitments", _entry(commitment, cron_job_id), timestamp=ts)


def create_waiting_for_kj_record_from_response(
    *,
    user_message: str,
    assistant_response: str,
    obsidian_vault: str | Path,
) -> Path | None:
    commitment = infer_waiting_for_kj_commitment(
        user_message=user_message,
        assistant_response=assistant_response,
    )
    if commitment is None:
        return None
    return create_commitment_record(commitment, obsidian_vault)
