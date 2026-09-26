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
SOL_REVIEWER_PROVIDER = "openai-codex"
SOL_REVIEWER_MODEL = "gpt-5.6-sol"


class AlertDeliveryIndeterminateError(RuntimeError):
    """The Slack send may have succeeded but cannot yet be reconciled."""


def sample_receipt_ids(
    receipt_ids: Sequence[str], sample_size: int, seed: bytes
) -> list[str]:
    """Rank receipt IDs by the persisted HMAC-SHA256 replay contract."""
    if len(seed) < 32:
        raise ValueError("sampling seed must contain at least 256 bits")
    sample_size = min(max(0, int(sample_size)), len(receipt_ids))
    ranked = sorted(
        (str(receipt_id) for receipt_id in receipt_ids),
        key=lambda receipt_id: (
            hmac.new(seed, receipt_id.encode("utf-8"), hashlib.sha256).digest(),
            receipt_id,
        ),
    )
    return ranked[:sample_size]


def build_review_prompt(attempt: Mapping[str, Any]) -> str:
    """Build a metadata-only receipt-integrity prompt without payload evidence."""
    evidence = {
        key: attempt.get(key)
        for key in (
            "receipt_id",
            "route_requested",
            "route_decision",
            "route_reason",
            "data_classification",
            "output_contract",
            "worker_status",
            "fallback_used",
            "error_code",
            "goal_sha256",
            "context_sha256",
            "prompt_sha256",
            "response_sha256",
            "response_bytes",
        )
    }
    return (
        "You are an isolated receipt-integrity evaluator. Review only the metadata below. "
        "Prompt and response bodies are intentionally unavailable. Do not infer semantic "
        "correctness from hashes or byte counts; use fail with failure_kind unreviewable when "
        "content would be required.\n\n"
        "Attempt JSON:\n"
        f"{_canonical_json(evidence)}\n\n"
        "Return JSON only, with exactly this semantic shape:\n"
        '{"verdict": "pass|fail", "reason": "one concrete sentence", '
        '"failure_kind": "none|correctness|instruction|omission|hallucination|worker_failure|unreviewable"}\n'
        "Use failure_kind none only with pass. The reason must be a non-empty string. "
        "Do not include Markdown fences or other text."
    )


def _parse_verdict(value: Mapping[str, Any] | str) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("reviewer_output_invalid: not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("reviewer_output_invalid: expected an object")
    expected_keys = {"verdict", "reason", "failure_kind"}
    if set(value) != expected_keys:
        raise ValueError(
            "reviewer_output_invalid: object must contain exactly verdict, reason, and failure_kind"
        )
    verdict = value.get("verdict")
    reason = value.get("reason")
    failure_kind = value.get("failure_kind")
    allowed_failure_kinds = {
        "none",
        "correctness",
        "instruction",
        "omission",
        "hallucination",
        "worker_failure",
        "unreviewable",
    }
    if verdict not in {"pass", "fail"}:
        raise ValueError("reviewer_output_invalid: verdict must be pass or fail")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reviewer_output_invalid: reason must be non-empty")
    if failure_kind not in allowed_failure_kinds:
        raise ValueError("reviewer_output_invalid: unsupported failure_kind")
    if (verdict == "pass") != (failure_kind == "none"):
        raise ValueError(
            "reviewer_output_invalid: failure_kind must be none exactly when verdict is pass"
        )
    return {
        "verdict": str(verdict),
        "reason": reason.strip(),
        "failure_kind": str(failure_kind),
    }


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
        if provider != SOL_REVIEWER_PROVIDER or model != SOL_REVIEWER_MODEL:
            raise ValueError("reviewer_identity_mismatch")
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
        actual_provider = getattr(agent, "provider", None)
        actual_model = getattr(agent, "model", None)
        if actual_provider != self.provider or actual_model != self.model:
            close = getattr(agent, "close", None)
            if callable(close):
                close()
            raise RuntimeError("reviewer_identity_mismatch")
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
        alert_workspace_id: str | None = None,
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
        self.alert_workspace_id = alert_workspace_id
        self.sample_size = max(0, int(sample_size))
        self.timezone_name = timezone_name
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lease_timeout_seconds = max(1, int(lease_timeout_seconds))

    def run(
        self,
        *,
        target_day: str | date,
        seed: bytes | None = None,
        pipeline_preflight_error: str | None = None,
    ) -> dict[str, Any]:
        day = target_day.isoformat() if isinstance(target_day, date) else str(target_day)
        existing = self.store.get_review_batch(day)
        if existing is not None:
            if existing["status"] in {"passed", "failed", "pipeline_failed"}:
                if existing["alert_status"] in {"pending", "sending"}:
                    alert = self._alert_for_batch(existing)
                    if existing["alert_status"] == "pending":
                        existing = self._persist_alert_outbox(existing, alert)
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
                    slack_channel_id=self.alert_channel_id,
                    completed_at=now,
                    slack_workspace_id=self.alert_workspace_id,
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

        review_lease_token = secrets.token_hex(16)
        if not self.store.claim_review_batch(
            batch["batch_id"], lease_token=review_lease_token, now=self.clock()
        ):
            current = self.store.get_review_batch(day) or batch
            terminal = current["status"] in {"passed", "failed", "pipeline_failed"}
            return self._result_from_batch(
                current, status=current["status"] if terminal else "in_progress"
            )

        if pipeline_preflight_error is not None:
            alert = self._failure_alert(
                day,
                [{"receipt_id": "pipeline", "reason": pipeline_preflight_error}],
                pipeline=True,
                batch=batch,
            )
            try:
                self.store.update_review_batch(
                    batch["batch_id"],
                    lease_token=review_lease_token,
                    status="pipeline_failed",
                    pipeline_error=pipeline_preflight_error,
                    alert_status="pending",
                    alert_message=alert,
                    alert_delivery_key=self._alert_delivery_key(
                        str(batch["batch_id"]), alert
                    ),
                    slack_channel_id=self.alert_channel_id,
                    slack_workspace_id=self.alert_workspace_id,
                )
            except KeyError:
                current = self.store.get_review_batch(day) or batch
                return self._result_from_batch(current)
            current = self.store.get_review_batch(day) or batch
            self._deliver_alert(current, alert)
            completed = self.store.get_review_batch(day) or current
            return self._result_from_batch(completed)

        selected = json.loads(batch["sample_receipt_ids_json"])
        failures: list[dict[str, str]] = []
        pipeline_error: str | None = None
        for ordinal, receipt_id in enumerate(selected):
            if any(
                item["receipt_id"] == receipt_id
                for item in self.store.list_review_items(batch["batch_id"])
            ):
                continue
            try:
                attempt = self.store.get_attempt(receipt_id)
            except KeyError:
                pipeline_error = "sampled_receipt_missing"
                break
            if attempt.get("completed_at_utc") is None:
                pipeline_error = "stale_started_attempt"
                try:
                    self.store.add_review_item(
                        batch_id=batch["batch_id"],
                        lease_token=review_lease_token,
                        receipt_id=receipt_id,
                        ordinal=ordinal,
                        reviewer_provider=self.reviewer_provider,
                        reviewer_model=self.reviewer_model,
                        review_status="failed",
                        error_code="stale_started_attempt",
                        error_message=pipeline_error,
                        completed_at=self.clock(),
                    )
                except KeyError:
                    current = self.store.get_review_batch(day) or batch
                    return self._result_from_batch(current)
                break
            try:
                reviewer = self.reviewer_factory()
                raw_verdict = _parse_verdict(reviewer(build_review_prompt(attempt)))
                verdict = raw_verdict
                if attempt.get("worker_status") != "completed" or attempt.get("error_code"):
                    verdict = {
                        "verdict": "fail",
                        "reason": "Gemini worker did not produce a completed result.",
                        "failure_kind": "worker_failure",
                    }
                self.store.add_review_item(
                    batch_id=batch["batch_id"],
                    lease_token=review_lease_token,
                    receipt_id=receipt_id,
                    ordinal=ordinal,
                    reviewer_provider=self.reviewer_provider,
                    reviewer_model=self.reviewer_model,
                    review_status="completed",
                    verdict=verdict["verdict"],
                    failure_kind=verdict["failure_kind"],
                    reason=verdict["reason"],
                    review_json=raw_verdict,
                    completed_at=self.clock(),
                )
                if verdict["verdict"] == "fail":
                    failures.append(
                        {"receipt_id": receipt_id, "reason": verdict["reason"]}
                    )
            except Exception as exc:
                raw_error = str(exc)
                pipeline_error = (
                    "reviewer_output_invalid"
                    if "reviewer_output_invalid" in raw_error
                    else "reviewer_failed"
                )
                try:
                    self.store.add_review_item(
                        batch_id=batch["batch_id"],
                        lease_token=review_lease_token,
                        receipt_id=receipt_id,
                        ordinal=ordinal,
                        reviewer_provider=self.reviewer_provider,
                        reviewer_model=self.reviewer_model,
                        review_status="failed",
                        error_code=pipeline_error,
                        error_message=None,
                        completed_at=self.clock(),
                    )
                except KeyError:
                    current = self.store.get_review_batch(day) or batch
                    return self._result_from_batch(current)
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

        delivery_key = self._alert_delivery_key(str(batch["batch_id"]), alert) if alert else None
        try:
            self.store.update_review_batch(
                batch["batch_id"],
                lease_token=review_lease_token,
                status=status,
                pipeline_error=pipeline_error,
                alert_status="pending" if alert else "not_needed",
                alert_message=alert,
                alert_delivery_key=delivery_key,
                slack_channel_id=self.alert_channel_id if alert else None,
                slack_workspace_id=self.alert_workspace_id if alert else None,
            )
        except KeyError:
            current = self.store.get_review_batch(day) or batch
            return self._result_from_batch(current)
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
        reviewed = self._completed_review_count(str(batch["batch_id"]))
        if pipeline:
            return self._pipeline_failure_alert(day, batch, reviewed=reviewed, sampled=sampled)
        receipt_path = self._display_receipt_path()
        return (
            f"Gemini daily review {day}: FAIL — {len(failures)}/{sampled} sampled "
            f"tasks failed; pipeline=ok. Receipt: {receipt_path} batch {batch['batch_id']}"
        )

    def _pipeline_failure_alert(
        self,
        day: str,
        batch: Mapping[str, Any],
        *,
        reviewed: int,
        sampled: int,
    ) -> str:
        return (
            f"Gemini daily review {day}: PIPELINE FAIL — {reviewed}/{sampled} "
            f"reviews completed. Receipt: {self._display_receipt_path()} "
            f"batch {batch['batch_id']}"
        )

    def _display_receipt_path(self) -> str:
        try:
            relative = self.store.path.resolve().relative_to(Path.home().resolve())
        except ValueError:
            return str(self.store.path)
        return f"~/{relative}"

    def _alert_for_batch(self, batch: Mapping[str, Any]) -> str:
        persisted = batch.get("alert_message")
        if isinstance(persisted, str) and persisted:
            return persisted
        pipeline = batch["status"] == "pipeline_failed"
        items = self.store.list_review_items(str(batch["batch_id"]))
        failures = [item for item in items if item.get("verdict") == "fail"]
        if pipeline and not failures:
            failures = [{"receipt_id": "pipeline", "reason": "pipeline failure"}]
        alert = self._failure_alert(
            str(batch["routing_day"]), failures, pipeline=pipeline, batch=batch
        )
        persisted_hash = str(batch.get("alert_message_sha256") or "")
        if pipeline and persisted_hash:
            current_hash = hashlib.sha256(alert.encode("utf-8")).hexdigest()
            if current_hash != persisted_hash:
                sampled = len(json.loads(str(batch["sample_receipt_ids_json"])))
                legacy_alert = self._pipeline_failure_alert(
                    str(batch["routing_day"]),
                    batch,
                    reviewed=len(items),
                    sampled=sampled,
                )
                if hashlib.sha256(legacy_alert.encode("utf-8")).hexdigest() == persisted_hash:
                    return legacy_alert
        return alert

    def _deliver_alert(self, batch: Mapping[str, Any], alert: str) -> None:
        lease_token = secrets.token_hex(16)
        now = self.clock()
        stale_before = now - timedelta(seconds=self.lease_timeout_seconds)
        if not self.store.claim_alert_delivery(
            str(batch["batch_id"]),
            lease_token=lease_token,
            now=now,
            stale_before=stale_before,
        ):
            return
        claimed = self.store.get_review_batch(str(batch["routing_day"])) or batch
        try:
            expected_hash = hashlib.sha256(alert.encode("utf-8")).hexdigest()
            expected_key = self._alert_delivery_key(str(batch["batch_id"]), alert)
            expected_channel = str(claimed.get("slack_channel_id") or "")
            if (
                claimed.get("alert_status") != "sending"
                or claimed.get("alert_lease_token") != lease_token
                or claimed.get("alert_message") != alert
                or claimed.get("alert_message_sha256") != expected_hash
                or claimed.get("alert_delivery_key") != expected_key
                or not expected_channel
            ):
                raise RuntimeError("Slack alert outbox identity mismatch")
            delivery = self.alert_sender(alert)
            if not isinstance(delivery, Mapping) or delivery.get("success") is not True:
                raise RuntimeError("Slack delivery was not confirmed")
            channel = str(delivery.get("chat_id") or delivery.get("channel") or "")
            message_ts = str(delivery.get("message_id") or delivery.get("ts") or "")
            if channel != expected_channel or not message_ts:
                raise RuntimeError("Slack delivery target or message receipt mismatch")
        except AlertDeliveryIndeterminateError:
            return
        except Exception:
            self.store.release_alert_delivery(
                str(batch["batch_id"]), lease_token=lease_token
            )
            return
        self.store.complete_alert_delivery(
            str(batch["batch_id"]),
            lease_token=lease_token,
            slack_channel_id=channel,
            slack_message_ts=message_ts,
        )

    @staticmethod
    def _alert_delivery_key(batch_id: str, alert: str) -> str:
        digest = hashlib.sha256(alert.encode("utf-8")).hexdigest()
        return f"gemini-daily-review:{batch_id}:{digest}"

    def _persist_alert_outbox(
        self, batch: Mapping[str, Any], alert: str
    ) -> dict[str, Any]:
        digest = hashlib.sha256(alert.encode("utf-8")).hexdigest()
        delivery_key = self._alert_delivery_key(str(batch["batch_id"]), alert)
        persisted_hash = str(batch.get("alert_message_sha256") or "")
        persisted_key = str(batch.get("alert_delivery_key") or "")
        persisted_channel = str(batch.get("slack_channel_id") or "")
        persisted_workspace = str(batch.get("slack_workspace_id") or "")
        if (
            (persisted_hash and persisted_hash != digest)
            or (persisted_key and persisted_key != delivery_key)
        ):
            return dict(batch)
        destination_channel = persisted_channel or self.alert_channel_id
        destination_workspace = persisted_workspace or self.alert_workspace_id
        self.store.initialize_alert_outbox(
            str(batch["batch_id"]),
            alert_message=alert,
            alert_delivery_key=delivery_key,
            slack_channel_id=destination_channel,
            slack_workspace_id=destination_workspace,
        )
        return self.store.get_review_batch(str(batch["routing_day"])) or dict(batch)

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
            "reviewed_count": sum(
                item.get("review_status") == "completed" for item in items
            ),
            "alert_status": batch["alert_status"],
        }

    def _completed_review_count(self, batch_id: str) -> int:
        return sum(
            item.get("review_status") == "completed"
            for item in self.store.list_review_items(batch_id)
        )


def main() -> dict[str, Any]:
    """Run one configured scheduler tick without writing successful output."""
    from agent.gemini_daily_review_runtime import run_configured_review

    return run_configured_review()
