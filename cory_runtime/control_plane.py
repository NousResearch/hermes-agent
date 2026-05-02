from __future__ import annotations

from typing import Any

import httpx

from .models import ClaimedJobEnvelope, InterpretationSubmission


class ControlPlaneApiError(RuntimeError):
    """Raised when the control plane returns a non-success response."""


class ControlPlaneClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
            },
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ControlPlaneClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def claim_next_job(self) -> ClaimedJobEnvelope | None:
        payload = self._request("POST", "/api/internal/request-interpretation-jobs/claim")
        envelope = ClaimedJobEnvelope.model_validate(payload)
        return envelope if envelope.claimed else None

    def complete_job(
        self,
        claim: ClaimedJobEnvelope,
        submission: InterpretationSubmission,
    ) -> dict[str, Any]:
        assert claim.harness is not None
        return self._request(
            "POST",
            claim.harness.outputContract.completeEndpoint,
            json=submission.model_dump(exclude_none=True),
        )

    def fail_job(self, claim: ClaimedJobEnvelope, error_message: str) -> dict[str, Any]:
        assert claim.harness is not None
        return self._request(
            "POST",
            claim.harness.outputContract.failEndpoint,
            json={"errorMessage": error_message.strip()},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(method, path, json=json)
        if response.status_code >= 400:
            message = self._extract_error(response)
            raise ControlPlaneApiError(f"{method} {path} failed: {message}")
        return response.json()

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            message = payload.get("error")
            if isinstance(message, str) and message.strip():
                return message.strip()
        return response.text.strip() or f"http_{response.status_code}"
