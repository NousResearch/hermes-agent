"""Deterministic daily quality review for Gemini-routed delegations."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent.gemini_route_receipts import GeminiReceiptStore, _canonical_json


ReviewCallable = Callable[[str], Mapping[str, Any] | str]
ReviewerFactory = Callable[[], ReviewCallable]
AlertSender = Callable[[str], Any]
Clock = Callable[[], datetime]


def _seeded_digest(seed: bytes, counter: int) -> bytes:
    return hmac.new(seed, counter.to_bytes(16, "big"), hashlib.sha256).digest()


def _unbiased_index(seed: bytes, counter: int, upper: int) -> tuple[int, int]:
    """Return a deterministic rejection-sampled integer in ``range(upper)``."""
    if upper <= 0:
        raise ValueError("upper must be positive")
    modulus = 1 << 256
    limit = modulus - (modulus % upper)
    while True:
        value = int.from_bytes(_seeded_digest(seed, counter), "big")
        counter += 1
        if value < limit:
            return value % upper, counter


def sample_receipt_ids(
    receipt_ids: Sequence[str], sample_size: int, seed: bytes
) -> list[str]:
    """Select uniformly without replacement using a persisted 256-bit seed."""
    if len(seed) < 32:
        raise ValueError("sampling seed must contain at least 256 bits")
    sample_size = min(max(0, int(sample_size)), len(receipt_ids))
    pool = sorted(str(receipt_id) for receipt_id in receipt_ids)
    counter = 0
    for index in range(len(pool) - 1, len(pool) - sample_size - 1, -1):
        chosen, counter = _unbiased_index(seed, counter, index + 1)
        pool[index], pool[chosen] = pool[chosen], pool[index]
    return pool[len(pool) - sample_size :] if sample_size else []


def build_review_prompt(attempt: Mapping[str, Any]) -> str:
    """Build a self-contained rubric prompt; no parent memory or SOUL is needed."""
    evidence = {
        "receipt_id": attempt.get("receipt_id"),
        "route_requested": attempt.get("route_requested"),
        "route_decision": attempt.get("route_decision"),
        "route_reason": attempt.get("route_reason"),
        "data_classification": attempt.get("data_classification"),
        "output_contract": attempt.get("output_contract"),
        "goal": attempt.get("goal_text"),
        "context": attempt.get("context_text"),
        "worker_status": attempt.get("worker_status"),
        "response": attempt.get("response_text"),
        "fallback_used": attempt.get("fallback_used"),
        "error_code": attempt.get("error_code"),
        "error_message": attempt.get("error_message"),
    }
    return (
        "You are an isolated quality reviewer. Evaluate the recorded Gemini result for "
        "correctness, completeness, grounding in the supplied evidence, calibrated "
        "uncertainty, and compliance with the requested output contract. A failed or "
        "fallback attempt is not automatically a quality failure; judge whether its "
        "recorded behavior was correct, useful, and honest.\n\n"
        "Attempt JSON:\n"
        f"{_canonical_json(evidence)}\n\n"
        "Return JSON only, with exactly this semantic shape:\n"
        '{"verdict": "faithful|unfaithful", "reason": "brief concrete reason"}\n'
        "The verdict must be the literal string faithful or unfaithful. The reason must be a "
        "non-empty string. Do not include Markdown fences or other text."
    )


def _parse_verdict(value: Mapping[str, Any] | str) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("reviewer_output_invalid: not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("reviewer_output_invalid: expected an object")
    verdict = value.get("verdict")
    reason = value.get("reason")
    verdict_map = {
        "faithful": "pass",
        "unfaithful": "fail",
        # Internal/test callers may already use the persisted vocabulary.
        "pass": "pass",
        "fail": "fail",
    }
    if verdict not in verdict_map:
        raise ValueError("reviewer_output_invalid: verdict must be faithful or unfaithful")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reviewer_output_invalid: reason must be non-empty")
    return {"verdict": verdict_map[verdict], "reason": reason.strip()}


class SolReviewer:
    """Fresh, tool-free, persistence-free AIAgent wrapper for one review."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str = "medium",
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not provider or not model:
            raise ValueError("review provider and model are required")
        if agent_factory is None:
            from run_agent import AIAgent

            agent_factory = AIAgent
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.agent_factory = agent_factory

    def __call__(self, prompt: str) -> dict[str, str]:
        agent = self.agent_factory(
            provider=self.provider,
            model=self.model,
            reasoning_config={"enabled": True, "effort": self.reasoning_effort},
            max_iterations=3,
            enabled_toolsets=[],
            disabled_toolsets=[],
            save_trajectories=False,
            quiet_mode=True,
            skip_context_files=True,
            load_soul_identity=False,
            skip_memory=True,
            session_db=None,
            ephemeral_system_prompt=(
                "Review only the evidence in the user prompt. Do not use tools, memory, "
                "workspace context, or outside knowledge. Return strict JSON only."
            ),
        )
        # Prevent the lazy session-DB fallback from persisting this isolated review.
        setattr(agent, "_persist_disabled", True)
        try:
            result = agent.run_conversation(prompt)
            raw = result.get("final_response") if isinstance(result, Mapping) else result
            if not isinstance(raw, (str, Mapping)):
                raise ValueError("reviewer_output_invalid: missing final response")
            return _parse_verdict(raw)
        finally:
            close = getattr(agent, "close", None)
            if callable(close):
                close()


def _safe_fragment(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    text = "".join(ch for ch in text if ch.isprintable())
    return text[:limit] or "unspecified"


class DailyReviewRunner:
    """Claim, sample, review, persist, and optionally alert one routing day."""

    def __init__(
        self,
        *,
        store: GeminiReceiptStore,
        reviewer_factory: ReviewerFactory,
        reviewer_provider: str,
        reviewer_model: str,
        alert_sender: AlertSender,
        alert_channel_id: str,
        sample_size: int = 5,
        timezone_name: str = "America/Los_Angeles",
        clock: Clock | None = None,
        lease_timeout_seconds: int = 150,
    ) -> None:
        if not reviewer_provider or not reviewer_model:
            raise ValueError("reviewer provider and model are required")
        if not alert_channel_id:
            raise ValueError("alert_channel_id is required")
        self.store = store
        self.reviewer_factory = reviewer_factory
        self.reviewer_provider = reviewer_provider
        self.reviewer_model = reviewer_model
        self.alert_sender = alert_sender
        self.alert_channel_id = alert_channel_id
        self.sample_size = max(0, int(sample_size))
        self.timezone_name = timezone_name
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lease_timeout_seconds = max(1, int(lease_timeout_seconds))

    def run(
        self,
        *,
        target_day: str | date,
        seed: bytes | None = None,
    ) -> dict[str, Any]:
        day = target_day.isoformat() if isinstance(target_day, date) else str(target_day)
        existing = self.store.get_review_batch(day)
        if existing is not None:
            if existing["status"] in {"passed", "failed", "pipeline_failed"}:
                if existing["alert_status"] == "pending":
                    alert = self._alert_for_batch(existing)
                    self._deliver_alert(existing, alert)
                    existing = self.store.get_review_batch(day) or existing
                return self._result_from_batch(existing)
            if existing["status"] in {"preparing", "reviewing"} and self._lease_is_stale(
                existing
            ):
                alert = self._failure_alert(
                    day,
                    [{"receipt_id": "pipeline", "reason": "stale_review_lease"}],
                    pipeline=True,
                    batch=existing,
                )
                now = self.clock()
                stale_before = now - timedelta(seconds=self.lease_timeout_seconds)
                terminalized = self.store.fail_stale_review_batch(
                    str(existing["batch_id"]),
                    stale_before=stale_before,
                    pipeline_error="stale_review_lease",
                    alert_message=alert,
                    completed_at=now,
                )
                current = self.store.get_review_batch(day) or existing
                if terminalized and (
                    current["status"] == "pipeline_failed"
                    and current["alert_status"] == "pending"
                ):
                    self._deliver_alert(current, alert)
                    current = self.store.get_review_batch(day) or current
                return self._result_from_batch(current)
            if existing["status"] == "reviewing":
                return self._result_from_batch(existing, status="in_progress")

        if existing is None:
            cohort = self.store.list_started_attempts_for_day(day)
            sample_seed = seed or secrets.token_bytes(32)
            selected = sample_receipt_ids(
                [row["receipt_id"] for row in cohort], self.sample_size, sample_seed
            )
            batch = self.store.create_or_get_review_batch(
                routing_day=day,
                timezone_name=self.timezone_name,
                sample_size_requested=self.sample_size,
                eligible_count=len(cohort),
                sample_seed_hex=sample_seed.hex(),
                sample_receipt_ids=selected,
            )
        else:
            batch = existing

        if not self.store.claim_review_batch(batch["batch_id"]):
            current = self.store.get_review_batch(day) or batch
            terminal = current["status"] in {"passed", "failed", "pipeline_failed"}
            return self._result_from_batch(
                current, status=current["status"] if terminal else "in_progress"
            )

        selected = json.loads(batch["sample_receipt_ids_json"])
        failures: list[dict[str, str]] = []
        pipeline_error: str | None = None
        for ordinal, receipt_id in enumerate(selected):
            if any(
                item["receipt_id"] == receipt_id
                for item in self.store.list_review_items(batch["batch_id"])
            ):
                continue
            attempt = self.store.get_attempt(receipt_id)
            if attempt.get("completed_at_utc") is None:
                pipeline_error = "stale_started_attempt"
                self.store.add_review_item(
                    batch_id=batch["batch_id"],
                    receipt_id=receipt_id,
                    ordinal=ordinal,
                    reviewer_provider=self.reviewer_provider,
                    reviewer_model=self.reviewer_model,
                    review_status="failed",
                    error_code="stale_started_attempt",
                    error_message=pipeline_error,
                    completed_at=self.clock(),
                )
                break
            try:
                reviewer = self.reviewer_factory()
                verdict = _parse_verdict(reviewer(build_review_prompt(attempt)))
                self.store.add_review_item(
                    batch_id=batch["batch_id"],
                    receipt_id=receipt_id,
                    ordinal=ordinal,
                    reviewer_provider=self.reviewer_provider,
                    reviewer_model=self.reviewer_model,
                    review_status="completed",
                    verdict=verdict["verdict"],
                    reason=verdict["reason"],
                    review_json=verdict,
                    completed_at=self.clock(),
                )
                if verdict["verdict"] == "fail":
                    failures.append(
                        {"receipt_id": receipt_id, "reason": verdict["reason"]}
                    )
            except Exception as exc:
                pipeline_error = _safe_fragment(exc)
                self.store.add_review_item(
                    batch_id=batch["batch_id"],
                    receipt_id=receipt_id,
                    ordinal=ordinal,
                    reviewer_provider=self.reviewer_provider,
                    reviewer_model=self.reviewer_model,
                    review_status="failed",
                    error_code=(
                        "reviewer_output_invalid"
                        if "reviewer_output_invalid" in pipeline_error
                        else "reviewer_failed"
                    ),
                    error_message=pipeline_error,
                    completed_at=self.clock(),
                )
                break

        if pipeline_error is not None:
            status = "pipeline_failed"
            alert = self._failure_alert(
                day,
                [{"receipt_id": "pipeline", "reason": pipeline_error}],
                pipeline=True,
                batch=batch,
            )
        elif failures:
            status = "failed"
            alert = self._failure_alert(day, failures, pipeline=False, batch=batch)
        else:
            status = "passed"
            alert = None

        self.store.update_review_batch(
            batch["batch_id"],
            status=status,
            pipeline_error=pipeline_error,
            alert_status="pending" if alert else "not_needed",
            alert_message=alert,
        )
        if alert:
            current_batch = self.store.get_review_batch(day) or {**batch, "status": status}
            self._deliver_alert(current_batch, alert)
        completed = self.store.get_review_batch(day) or batch
        return self._result_from_batch(completed)

    def _lease_is_stale(self, batch: Mapping[str, Any]) -> bool:
        started = datetime.fromisoformat(str(batch["started_at_utc"]))
        if started.tzinfo is None:
            raise ValueError("review batch started_at_utc must be timezone-aware")
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("review clock must return a timezone-aware datetime")
        elapsed = now.astimezone(timezone.utc) - started.astimezone(timezone.utc)
        return elapsed >= timedelta(seconds=self.lease_timeout_seconds)

    def _failure_alert(
        self,
        day: str,
        failures: Sequence[Mapping[str, str]],
        *,
        pipeline: bool,
        batch: Mapping[str, Any],
    ) -> str:
        sampled = len(json.loads(str(batch["sample_receipt_ids_json"])))
        reviewed = len(self.store.list_review_items(str(batch["batch_id"])))
        receipt_path = self._display_receipt_path()
        if pipeline:
            return (
                f"Gemini daily review {day}: PIPELINE FAIL — {reviewed}/{sampled} "
                f"reviews completed. Receipt: {receipt_path} batch {batch['batch_id']}"
            )
        return (
            f"Gemini daily review {day}: FAIL — {len(failures)}/{sampled} sampled "
            f"tasks failed; pipeline=ok. Receipt: {receipt_path} batch {batch['batch_id']}"
        )

    def _display_receipt_path(self) -> str:
        try:
            relative = self.store.path.resolve().relative_to(Path.home().resolve())
        except ValueError:
            return str(self.store.path)
        return f"~/{relative}"

    def _alert_for_batch(self, batch: Mapping[str, Any]) -> str:
        pipeline = batch["status"] == "pipeline_failed"
        items = self.store.list_review_items(str(batch["batch_id"]))
        failures = [item for item in items if item.get("verdict") == "fail"]
        if pipeline and not failures:
            failures = [{"receipt_id": "pipeline", "reason": "pipeline failure"}]
        return self._failure_alert(
            str(batch["routing_day"]), failures, pipeline=pipeline, batch=batch
        )

    def _deliver_alert(self, batch: Mapping[str, Any], alert: str) -> None:
        try:
            delivery = self.alert_sender(alert)
            if not isinstance(delivery, Mapping) or delivery.get("success") is not True:
                raise RuntimeError("Slack delivery was not confirmed")
            channel = str(delivery.get("chat_id") or delivery.get("channel") or "")
            message_ts = str(delivery.get("message_id") or delivery.get("ts") or "")
            if channel != self.alert_channel_id or not message_ts:
                raise RuntimeError("Slack delivery target or message receipt mismatch")
        except Exception:
            self.store.update_review_batch(
                str(batch["batch_id"]),
                status=str(batch["status"]),
                pipeline_error=batch.get("pipeline_error"),
                alert_status="pending",
            )
            return
        self.store.update_review_batch(
            str(batch["batch_id"]),
            status=str(batch["status"]),
            pipeline_error=batch.get("pipeline_error"),
            alert_status="sent",
            slack_channel_id=channel,
            slack_message_ts=message_ts,
        )

    def _result_from_batch(
        self, batch: Mapping[str, Any], *, status: str | None = None
    ) -> dict[str, Any]:
        items = self.store.list_review_items(str(batch["batch_id"]))
        return {
            "batch_id": batch["batch_id"],
            "routing_day": batch["routing_day"],
            "status": status or batch["status"],
            "eligible_count": int(batch["eligible_count"]),
            "sample_size": len(json.loads(batch["sample_receipt_ids_json"])),
            "reviewed_count": len(items),
            "alert_status": batch["alert_status"],
        }


def main() -> dict[str, Any]:
    """Run one configured scheduler tick without writing successful output."""
    from agent.gemini_daily_review_runtime import run_configured_review

    return run_configured_review()
