from __future__ import annotations

import hashlib
import json
import os
import signal
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")
DEFAULT_RECIPIENT = "+15163843337"


class JobContractError(RuntimeError):
    """Raised when a job violates an explicit runtime contract."""


class JobTimeoutError(TimeoutError):
    """Raised when a job exceeds its declared timeout."""


@dataclass(frozen=True)
class DeliveryOutcome:
    status: str
    error: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def torben_home() -> Path:
    return Path(os.getenv("TORBEN_PROFILE_HOME") or os.getenv("HERMES_HOME") or DEFAULT_TORBEN_HOME)


def state_dir(profile_home: Path | None = None) -> Path:
    path = (profile_home or torben_home()) / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def health_path(job_name: str, profile_home: Path | None = None) -> Path:
    return state_dir(profile_home) / f"{job_name}-health.json"


def redact_text(value: Any) -> str:
    text = str(value)
    replacements = [
        ("ghp_", "ghp_[REDACTED]"),
        ("gho_", "gho_[REDACTED]"),
        ("github_pat_", "github_pat_[REDACTED]"),
        ("AIza", "AIza[REDACTED]"),
        ("xoxb-", "xoxb-[REDACTED]"),
        ("xoxp-", "xoxp-[REDACTED]"),
        ("sk-", "sk-[REDACTED]"),
    ]
    for needle, replacement in replacements:
        if needle in text:
            head, _, tail = text.partition(needle)
            text = head + replacement + tail[:4] + "[REDACTED]"
    return text


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _previous_failures(path: Path) -> int:
    payload = load_json(path, {})
    if isinstance(payload, dict):
        try:
            return int(payload.get("consecutive_failures") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": redact_text(str(exc))[:800],
        "traceback_tail": redact_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))[-4000:],
    }


def write_health(
    job_name: str,
    *,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    exit_code: int = 0,
    error: BaseException | str | None = None,
    delivery: DeliveryOutcome | None = None,
    profile_home: Path | None = None,
) -> dict[str, Any]:
    path = health_path(job_name, profile_home)
    previous = _previous_failures(path)
    failed = status != "ok" or exit_code != 0 or (delivery and delivery.status == "failed")
    if failed:
        consecutive_failures = previous + 1
    else:
        consecutive_failures = 0
    payload: dict[str, Any] = {
        "schema": "torben.job-health.v1",
        "job": job_name,
        "status": "failed" if failed else "ok",
        "consecutive_failures": consecutive_failures,
        "last_run_at": finished_at or isoformat(),
        "started_at": started_at,
        "finished_at": finished_at or isoformat(),
        "exit_code": exit_code,
    }
    if error:
        payload["last_error"] = _error_payload(error) if isinstance(error, BaseException) else {"message": redact_text(error)}
    if delivery:
        payload["last_delivery_status"] = delivery.status
        if delivery.error:
            payload["last_delivery_error"] = redact_text(delivery.error)[:800]
    write_json_atomic(path, payload)
    return payload


@contextmanager
def timeout_after(seconds: int | None) -> Iterator[None]:
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise JobTimeoutError(f"job exceeded timeout_seconds={seconds}")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _exit_code(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, bool):
        return 0 if result else 1
    if isinstance(result, int):
        return result
    return 0


def run_job(
    job_name: str,
    body: Callable[[], Any],
    *,
    timeout_seconds: int | None = None,
    expect_non_empty: Callable[[Any], bool] | None = None,
    empty_error: str = "empty_result",
    profile_home: Path | None = None,
) -> int:
    started = isoformat()
    started_monotonic = time.monotonic()
    try:
        with timeout_after(timeout_seconds):
            result = body()
        exit_code = _exit_code(result)
        if exit_code != 0:
            raise JobContractError(f"job returned nonzero exit_code={exit_code}")
        if expect_non_empty and not expect_non_empty(result):
            raise JobContractError(empty_error)
        finished = isoformat()
        health = write_health(
            job_name,
            status="ok",
            started_at=started,
            finished_at=finished,
            exit_code=0,
            profile_home=profile_home,
        )
        health["duration_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json_atomic(health_path(job_name, profile_home), health)
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            finished = isoformat()
            write_health(job_name, status="ok", started_at=started, finished_at=finished, exit_code=0, profile_home=profile_home)
            return 0
        finished = isoformat()
        write_health(
            job_name,
            status="failed",
            started_at=started,
            finished_at=finished,
            exit_code=code,
            error=f"SystemExit({code})",
            profile_home=profile_home,
        )
        return code
    except BaseException as exc:  # noqa: BLE001 - cron boundary must persist every failure.
        finished = isoformat()
        write_health(
            job_name,
            status="failed",
            started_at=started,
            finished_at=finished,
            exit_code=1,
            error=exc,
            profile_home=profile_home,
        )
        print(f"{job_name} failed: {type(exc).__name__}: {redact_text(exc)}")
        return 1


def delivery_failed(job_name: str, error: str, *, profile_home: Path | None = None) -> dict[str, Any]:
    now = isoformat()
    return write_health(
        job_name,
        status="failed",
        started_at=now,
        finished_at=now,
        exit_code=1,
        error="delivery_failed",
        delivery=DeliveryOutcome(status="failed", error=error),
        profile_home=profile_home,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_alert_outcome(
    *,
    job_name: str,
    text: str,
    kind: str = "alert",
    recipient: str = DEFAULT_RECIPIENT,
    dedupe_seconds: int = 3600,
    rate_limit_per_hour: int = 10,
    now: datetime | None = None,
    deliver: Callable[[str, str], None] | None = None,
    profile_home: Path | None = None,
) -> dict[str, Any]:
    current = now or utc_now()
    current_ts = current.timestamp()
    sdir = state_dir(profile_home)
    rate_path = sdir / "torben-signal-rate-state.json"
    outbox_path = sdir / "torben-signal-outbox.jsonl"
    state = load_json(rate_path, {"dedupe": {}, "sent": []})
    if not isinstance(state, dict):
        state = {"dedupe": {}, "sent": []}
    dedupe = state.setdefault("dedupe", {})
    sent = [float(ts) for ts in state.setdefault("sent", []) if current_ts - float(ts) < 3600]
    fingerprint = _sha256(f"{kind}:{recipient}:{text}")
    previous_ts = float(dedupe.get(fingerprint) or 0)
    status = "sent"
    error = None
    if previous_ts and current_ts - previous_ts < dedupe_seconds:
        status = "skipped_duplicate"
    elif len(sent) >= rate_limit_per_hour:
        status = "skipped_rate_limited"
    else:
        try:
            if deliver:
                deliver(recipient, text)
            sent.append(current_ts)
            dedupe[fingerprint] = current_ts
        except Exception as exc:  # noqa: BLE001 - delivery failures are state, not crashes.
            status = "failed"
            error = redact_text(exc)
            delivery_failed(job_name, error, profile_home=profile_home)
    state["sent"] = sent
    write_json_atomic(rate_path, state)
    record = {
        "id": _sha256(f"{isoformat(current)}:{kind}:{recipient}:{text}"),
        "ts": isoformat(current),
        "job": job_name,
        "kind": kind,
        "recipient": recipient,
        "fingerprint": fingerprint,
        "status": status,
    }
    if error:
        record["error"] = error
    with outbox_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record
