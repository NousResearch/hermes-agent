"""Bounded, credential-free provider endpoint reachability probes."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import httpx


# A misbehaving transport can ignore both socket timeouts and thread cancellation.
# Keep that failure mode process-bounded: once these isolated one-shot probes are
# occupied, later diagnostics fail closed instead of creating more daemon threads.
_MAX_IN_FLIGHT_PROBES = 4
_PROBE_SLOTS = threading.BoundedSemaphore(_MAX_IN_FLIGHT_PROBES)


@dataclass(frozen=True)
class ProbeOutcome:
    """A low-cardinality description of provider endpoint reachability."""

    status: Literal["reachable", "unreachable", "unavailable", "disabled"]
    http_status: int | None = None
    detail: str = ""


def _sanitized_probe_target(base_url: str) -> str | None:
    """Return an HTTP(S) target without credentials, query, or fragment."""

    if (
        not isinstance(base_url, str)
        or not base_url
        or any(character.isspace() for character in base_url)
    ):
        return None

    try:
        parsed = urlsplit(base_url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None

        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"

        return urlunsplit((parsed.scheme.lower(), host, parsed.path or "", "", ""))
    except (TypeError, ValueError):
        return None


def probe_provider_endpoint(base_url: str, timeout_seconds: float) -> ProbeOutcome:
    """Probe once, with ``timeout_seconds`` bounding total caller wall time.

    The daemon owner thread creates, uses, and closes its fresh client.  If a
    transport ignores httpx's phase timeouts, the caller still returns at the
    total deadline; the owner retains the client and closes it when the blocked
    transport eventually unwinds.
    """

    target = _sanitized_probe_target(base_url)
    if (
        target is None
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return ProbeOutcome(status="unavailable", detail="invalid probe configuration")

    timeout = httpx.Timeout(
        connect=timeout_seconds,
        read=timeout_seconds,
        write=timeout_seconds,
        pool=timeout_seconds,
    )
    if not _PROBE_SLOTS.acquire(blocking=False):
        return ProbeOutcome(status="unavailable", detail="ProbeCapacity")
    completed = threading.Event()
    outcome: list[ProbeOutcome] = []
    client_factory = httpx.Client

    def _owner() -> None:
        try:
            with client_factory(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream(
                    "HEAD",
                    target,
                    headers={"User-Agent": "hermes-provider-probe"},
                ) as response:
                    http_status = response.status_code
            outcome.append(
                ProbeOutcome(
                    status="reachable",
                    http_status=http_status,
                    detail=f"endpoint returned HTTP {http_status}",
                )
            )
        except httpx.HTTPError as exc:
            outcome.append(
                ProbeOutcome(status="unreachable", detail=type(exc).__name__)
            )
        except (TypeError, ValueError) as exc:
            outcome.append(
                ProbeOutcome(status="unavailable", detail=type(exc).__name__)
            )
        except Exception as exc:
            outcome.append(
                ProbeOutcome(status="unavailable", detail=type(exc).__name__)
            )
        finally:
            _PROBE_SLOTS.release()
            completed.set()

    try:
        threading.Thread(
            target=_owner, name="provider-health-probe", daemon=True
        ).start()
    except Exception as exc:
        _PROBE_SLOTS.release()
        return ProbeOutcome(status="unavailable", detail=type(exc).__name__)
    if not completed.wait(timeout_seconds):
        return ProbeOutcome(status="unreachable", detail="ProbeTimeout")
    return outcome[0]
