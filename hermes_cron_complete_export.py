"""Typed contract for a complete export of Hermes' stock cron state.

The stock cron store is a profile-local collection of JSON, Markdown, SQLite, and
JSONL records.  It is not one database and the ``jobs.py`` lock is not a complete
export fence: it is an in-process lock plus a best-effort cross-process lock that
degrades after its bounded wait.  This module therefore defines the narrow record
shape that a future owner-scoped implementation must provide without opening a
caller-selected path, connection, or credential store.

The owner-scoped factory is delegated to the sibling runtime module.  Keeping
the delegate lazy avoids importing the normal ``cron`` package while an empty
profile is being inspected: that package's constructors create business
directories and repair state.  The runtime resolves only the active profile,
holds the qualified custody lease through full consumption, and leaves outer
Cron drain/publisher qualification to Main/native composition.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ContextManager, Iterable, Protocol, runtime_checkable


CRON_CAPTURE_API_VERSION = "agc1663-stock-cron-capture/v1"
STOCK_CRON_SOURCE_ID = "stock-cron"
STOCK_CRON_SCHEMA_VERSION = 1

# These are the concrete durable source families produced by the stock cron
# package.  Their absence is meaningful when the corresponding feature has not
# been used; the implementation must not create an empty replacement store.
CRON_JOBS_SOURCE_ID = "stock-cron.jobs"
CRON_OUTPUTS_SOURCE_ID = "stock-cron.outputs"
CRON_EXECUTIONS_SOURCE_ID = "stock-cron.executions"
CRON_DELIVERIES_SOURCE_ID = "stock-cron.deliveries"
CRON_ADMISSIONS_SOURCE_ID = "stock-cron.admissions"
CRON_NOTEPAD_SOURCE_ID = "stock-cron.notepad"
CRON_SUGGESTIONS_SOURCE_ID = "stock-cron.suggestions"
CRON_BOT_CHAT_PENDING_SOURCE_ID = "stock-cron.bot-chat-pending"
CRON_CONTROL_MARKERS_SOURCE_ID = "stock-cron.control-markers"
CRON_RECOVERY_TELEMETRY_SOURCE_ID = "stock-cron.recovery-telemetry"
CRON_USAGE_AUDIT_SOURCE_ID = "stock-cron.usage-audit"

STOCK_CRON_SOURCE_IDS = (
    CRON_JOBS_SOURCE_ID,
    CRON_OUTPUTS_SOURCE_ID,
    CRON_EXECUTIONS_SOURCE_ID,
    CRON_DELIVERIES_SOURCE_ID,
    CRON_ADMISSIONS_SOURCE_ID,
    CRON_NOTEPAD_SOURCE_ID,
    CRON_SUGGESTIONS_SOURCE_ID,
    CRON_BOT_CHAT_PENDING_SOURCE_ID,
    CRON_CONTROL_MARKERS_SOURCE_ID,
    CRON_RECOVERY_TELEMETRY_SOURCE_ID,
    CRON_USAGE_AUDIT_SOURCE_ID,
)


class CronFieldDisposition(StrEnum):
    """How a captured field participates in the outer export policy."""

    BUSINESS_PAYLOAD = "business_payload"
    BUSINESS_METADATA = "business_metadata"
    CONTROL_RECOVERY = "control_recovery"
    DELIVERY_METADATA = "delivery_metadata"
    SECURITY_IDENTITY = "security_identity"
    DERIVED_INTERNAL = "derived_internal"


class CronCaptureError(RuntimeError):
    """Base class for a cron capture that cannot produce a complete stream."""

    def __init__(self, code: str, detail: str = "") -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("cron capture error code must be non-empty")
        self.code = code
        self.detail = detail if isinstance(detail, str) else ""
        super().__init__(code if not self.detail else f"{code}: {self.detail}")


class CronCaptureRefused(CronCaptureError):
    """The observed cron state is malformed, unknown, unsafe, or incomplete."""


class CronCaptureUnavailable(CronCaptureError):
    """The source-owner capture capability is not installed or bound yet."""


class CronCaptureFailed(CronCaptureError):
    """A capture failed after opening and cannot be published."""


@dataclass(frozen=True)
class CronCaptureBudget:
    """Bound one cron capture by record count, encoded bytes, and elapsed time."""

    max_records: int = 5_000_000
    max_bytes: int = 512 * 1024 * 1024
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name, value in (("max_records", self.max_records), ("max_bytes", self.max_bytes)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"cron capture {name} must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, numbers.Real)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("cron capture timeout_seconds must be a finite positive real")


@dataclass(frozen=True)
class CronCaptureManifest:
    """Provenance for one bounded stock-cron stream, without a readiness claim."""

    api_version: str = CRON_CAPTURE_API_VERSION
    source_id: str = STOCK_CRON_SOURCE_ID
    schema_version: int = STOCK_CRON_SCHEMA_VERSION
    source_ids: tuple[str, ...] = STOCK_CRON_SOURCE_IDS
    absent_source_ids: tuple[str, ...] = ()
    excluded_transient_source_ids: tuple[str, ...] = (
        "stock-cron.coordination-locks",
        "stock-cron.external-worker-handoff",
        "stock-cron.in-memory-runtime",
    )


@dataclass(frozen=True)
class CronCaptureField:
    """One typed value; ``raw_bytes`` preserves an original scalar/file encoding."""

    name: str
    disposition: CronFieldDisposition
    value: Any
    raw_bytes: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("cron capture field name must be non-empty")
        if not isinstance(self.disposition, CronFieldDisposition):
            raise TypeError("cron capture field disposition must be CronFieldDisposition")
        if self.raw_bytes is not None and not isinstance(self.raw_bytes, bytes):
            raise TypeError("cron capture field raw_bytes must be bytes or None")


@dataclass(frozen=True)
class CronCaptureRelation:
    """A stable edge between two captured cron records."""

    relation: str
    source_key: str
    target_key: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (
            self.relation, self.source_key, self.target_key
        )):
            raise ValueError("cron capture relation identity must be non-empty")


@dataclass(frozen=True)
class CronCaptureRecord:
    """One lossless cron record or evidence row.

    ``raw_bytes`` is populated for file-backed records such as ``jobs.json`` and
    run output.  SQLite and JSONL records use typed fields and retain each raw
    scalar in its field when the source exposes one. Credential/auth stores and
    source-classified credential fields must be excluded before capture reads
    them; an outer publisher cannot retroactively establish that boundary. The
    outer publisher owns canonical wire encoding only.
    """

    source_id: str
    record_type: str
    key: str
    fields: tuple[CronCaptureField, ...] = ()
    relations: tuple[CronCaptureRelation, ...] = ()
    raw_bytes: bytes | None = None
    evidence_class: str = "cron"

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (
            self.source_id, self.record_type, self.key, self.evidence_class
        )):
            raise ValueError("cron capture record identity must be non-empty")
        if not isinstance(self.fields, tuple) or not isinstance(self.relations, tuple):
            raise TypeError("cron capture fields and relations must be tuples")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("cron capture record fields must be unique")
        if self.raw_bytes is not None and not isinstance(self.raw_bytes, bytes):
            raise TypeError("cron capture record raw_bytes must be bytes or None")


@runtime_checkable
class CronStateCapture(Protocol):
    """A live capture whose source-owner custody remains held while consumed."""

    @property
    def manifest(self) -> CronCaptureManifest:
        """Return stable source provenance and explicit absence/exclusion facts."""
        ...

    def iter_records(self) -> Iterable[CronCaptureRecord]:
        """Yield every retained cron record exactly once in stable order."""
        ...


def open_existing_cron_complete_capture(
    *, budget: CronCaptureBudget = CronCaptureBudget(),
) -> ContextManager[CronStateCapture]:
    """Open the active profile's complete stock-cron capture.

    The implementation is imported only after the caller has selected a valid
    contract budget.  It owns profile resolution and all source reads; callers
    cannot provide a path, SQL statement, connection, or readiness flag.
    """

    if not isinstance(budget, CronCaptureBudget):
        raise TypeError("budget must be a CronCaptureBudget")
    try:
        from hermes_cron_complete_export_runtime import (
            open_existing_cron_complete_capture as _open_capture,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise CronCaptureUnavailable("cron_capture_unavailable") from exc

    return _open_capture(budget=budget)


__all__ = [
    "CRON_ADMISSIONS_SOURCE_ID",
    "CRON_BOT_CHAT_PENDING_SOURCE_ID",
    "CRON_CAPTURE_API_VERSION",
    "CRON_CONTROL_MARKERS_SOURCE_ID",
    "CRON_DELIVERIES_SOURCE_ID",
    "CRON_EXECUTIONS_SOURCE_ID",
    "CRON_JOBS_SOURCE_ID",
    "CRON_NOTEPAD_SOURCE_ID",
    "CRON_OUTPUTS_SOURCE_ID",
    "CRON_RECOVERY_TELEMETRY_SOURCE_ID",
    "CRON_SUGGESTIONS_SOURCE_ID",
    "CRON_USAGE_AUDIT_SOURCE_ID",
    "STOCK_CRON_SCHEMA_VERSION",
    "STOCK_CRON_SOURCE_ID",
    "STOCK_CRON_SOURCE_IDS",
    "CronCaptureBudget",
    "CronCaptureError",
    "CronCaptureFailed",
    "CronCaptureField",
    "CronCaptureManifest",
    "CronCaptureRecord",
    "CronCaptureRefused",
    "CronCaptureRelation",
    "CronCaptureUnavailable",
    "CronFieldDisposition",
    "CronStateCapture",
    "open_existing_cron_complete_capture",
]
