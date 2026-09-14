"""Append-only audit log.

**The invariant: model-visible means logged.** Any change to state that a model will
later see — an agent's instructions, its tools, its model, the branding wrapped around
its replies — must be recorded in this log *before* it takes effect, and the record must
be sufficient to reconstruct what changed.

Enforced structurally rather than by convention: model-visible changes are performed
through :meth:`AuditLog.model_visible_change`, which writes an ``intent`` record, runs
the change, and then writes ``committed`` or ``failed``. A crash between the two leaves
an open intent — which is the point. An unexplained intent in the log is a question
worth asking; a silent change is not even a question.

Storage is JSONL, one event per line, opened in append mode and flushed per write. It is
deliberately not SQLite: the log must survive and stay readable when the runtime's own
storage is the thing that broke, and ``tail -f`` is a feature during an incident.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from nova.audit.integrity import DEFAULT_KEEP, DEFAULT_MAX_BYTES, rotate, segments
from nova.errors import AuditError

#: Event kinds that change what a model can see. Every one of these MUST pass through
#: :meth:`AuditLog.model_visible_change`. Adding a kind here without routing it through
#: that method is caught by ``tests/platform/test_audit.py``.
MODEL_VISIBLE_KINDS: frozenset[str] = frozenset(
    {
        "agent.materialized",
        "agent.removed",
        "identity.applied",
        # Indexing a corpus changes what an agent can retrieve and quote, and granting one
        # changes which corpora it can reach at all. Both are squarely model-visible: the
        # question "why did the agent say that?" is unanswerable without them.
        "knowledge.indexed",
        "knowledge.granted",
        # A work item's title and body are read by a worker as its instructions. Placing
        # one on the board is therefore model-visible in the most direct sense there is.
        "work.submitted",
        # An operator's note lands in the same place and is read the same way. Releasing,
        # rejecting or resuming an item changes *when* a worker runs rather than *what it
        # reads*, so those are recorded rather than write-ahead — see `AgentRuntime.decide_work`.
        "work.annotated",
        # Connecting a channel makes an external conversation the text a worker reads. That
        # is model-visible in the most direct sense there is, and the write-ahead record is
        # what answers "when did this channel start reaching that agent" after the fact.
        "channel.connected",
    }
)

_DEFAULT_LOG_NAME = "audit.jsonl"


def new_correlation_id() -> str:
    """An id tying every event produced by one operator action together."""
    return uuid.uuid4().hex


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class AuditEvent:
    """One durable fact.

    ``phase`` is ``intent`` / ``committed`` / ``failed`` for model-visible changes and
    ``record`` for everything else. ``model_visible`` is stored explicitly rather than
    derived at read time, so a later change to :data:`MODEL_VISIBLE_KINDS` cannot
    retroactively reinterpret history.
    """

    event_id: str
    correlation_id: str
    ts: str
    kind: str
    phase: str
    actor: str
    tenant_id: str
    model_visible: bool
    subject: str = ""
    digest: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "AuditEvent":
        data = json.loads(line)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class AuditLog:
    """A JSONL audit log for one tenant.

    Instances are safe to share across threads within a process. Cross-process appends
    of whole lines under O_APPEND are atomic on POSIX for the sizes written here; the
    log is an operator record, not a coordination primitive, and nothing in NOVA reads
    it back to make decisions.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        tenant_id: str,
        actor: str = "nova",
        max_bytes: int = DEFAULT_MAX_BYTES,
        keep: int = DEFAULT_KEEP,
    ) -> None:
        self.path = Path(path)
        self.tenant_id = tenant_id
        self.actor = actor
        self.max_bytes = max_bytes
        self.keep = keep
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AuditError(f"cannot create audit log directory {self.path.parent}: {exc}") from exc

    def with_actor(self, actor: str) -> "AuditLog":
        """The same log, written as somebody else.

        The control plane needs this and nothing else does. A long-running server is one
        process writing on behalf of many people, and an event whose ``actor`` is the
        process rather than the person answers "what happened" but not "who decided" — the
        question an approval exists to answer. The alternative, leaving the human in
        ``detail`` only, means an auditor filtering on ``actor`` silently sees none of them.

        Shares the lock and the path deliberately: two instances writing the same file with
        separate locks would interleave within a process, which is the one case the lock is
        there to prevent.
        """
        if not actor:
            raise AuditError("an audit actor cannot be empty")
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.actor = actor
        return clone

    @classmethod
    def for_home(cls, home: Path, *, tenant_id: str, actor: str = "nova") -> "AuditLog":
        """The conventional log location inside a NOVA home directory."""
        return cls(Path(home) / "nova" / _DEFAULT_LOG_NAME, tenant_id=tenant_id, actor=actor)

    def _append(self, event: AuditEvent) -> AuditEvent:
        line = event.to_json() + "\n"
        try:
            with self._lock:
                # Checked before the write so the live segment never exceeds the bound by
                # more than one event.
                #
                # Its own try/except, deliberately: inside the enclosing one, an OSError
                # from rotation would be converted to AuditError and fail the append —
                # contradicting the rule that housekeeping never blocks the record. A full
                # disk must not also stop the record of why the disk filled.
                if self.max_bytes:
                    try:
                        rotate(self.path, max_bytes=self.max_bytes, keep=self.keep)
                    except OSError:
                        pass
                # O_APPEND so concurrent writers never interleave a partial line.
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, line.encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except OSError as exc:
            raise AuditError(f"cannot append to audit log {self.path}: {exc}") from exc
        return event

    def record(
        self,
        kind: str,
        *,
        correlation_id: str,
        subject: str = "",
        digest: str = "",
        detail: Optional[Mapping[str, Any]] = None,
        phase: str = "record",
        error: str = "",
    ) -> AuditEvent:
        """Append one event.

        Refuses model-visible kinds: those must go through
        :meth:`model_visible_change` so the intent/commit pair is never skipped.
        """
        if kind in MODEL_VISIBLE_KINDS and phase == "record":
            raise AuditError(
                f"{kind!r} changes model-visible state and must be written through "
                "model_visible_change(), not record()"
            )
        return self._append(
            AuditEvent(
                event_id=uuid.uuid4().hex,
                correlation_id=correlation_id,
                ts=_utc_now(),
                kind=kind,
                phase=phase,
                actor=self.actor,
                tenant_id=self.tenant_id,
                model_visible=kind in MODEL_VISIBLE_KINDS,
                subject=subject,
                digest=digest,
                detail=dict(detail or {}),
                error=error,
            )
        )

    @contextmanager
    def model_visible_change(
        self,
        kind: str,
        *,
        correlation_id: str,
        subject: str,
        digest: str = "",
        detail: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]:
        """Write-ahead guard around a change to model-visible state.

        Writes ``intent`` before the body runs and ``committed`` after it succeeds; on
        an exception it writes ``failed`` and re-raises. The yielded dict is merged into
        the committed event's detail, so a caller records what it actually did (paths
        written, values resolved) rather than only what it meant to do.
        """
        if kind not in MODEL_VISIBLE_KINDS:
            raise AuditError(
                f"{kind!r} is not declared in MODEL_VISIBLE_KINDS; add it there or use record()"
            )
        base = dict(detail or {})
        self._append(
            AuditEvent(
                event_id=uuid.uuid4().hex,
                correlation_id=correlation_id,
                ts=_utc_now(),
                kind=kind,
                phase="intent",
                actor=self.actor,
                tenant_id=self.tenant_id,
                model_visible=True,
                subject=subject,
                digest=digest,
                detail=base,
            )
        )
        outcome: dict[str, Any] = {}
        try:
            yield outcome
        except Exception as exc:
            self._append(
                AuditEvent(
                    event_id=uuid.uuid4().hex,
                    correlation_id=correlation_id,
                    ts=_utc_now(),
                    kind=kind,
                    phase="failed",
                    actor=self.actor,
                    tenant_id=self.tenant_id,
                    model_visible=True,
                    subject=subject,
                    digest=digest,
                    detail={**base, **outcome},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        self._append(
            AuditEvent(
                event_id=uuid.uuid4().hex,
                correlation_id=correlation_id,
                ts=_utc_now(),
                kind=kind,
                phase="committed",
                actor=self.actor,
                tenant_id=self.tenant_id,
                model_visible=True,
                subject=subject,
                digest=digest,
                detail={**base, **outcome},
            )
        )

    def read(self) -> list[AuditEvent]:
        """Every event, oldest first, across every rotated segment.

        Spanning segments rather than reading one file is what keeps rotation invisible to
        callers: ``open_intents`` must still find an intent whose commit landed after a
        rotation, or rotating the log would manufacture unfinished changes that never
        existed.
        """
        events: list[AuditEvent] = []
        for segment in segments(self.path):
            try:
                with open(segment, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            events.append(AuditEvent.from_json(line))
            except OSError:
                # A segment removed mid-read is expected under rotation; the rest is still
                # a truthful answer to "what is in the log".
                continue
        return events

    def open_intents(self) -> list[AuditEvent]:
        """Model-visible intents with no matching commit or failure.

        An entry here means a change began and its outcome was never recorded — a crash,
        a kill, or a bug. Surfacing these is the reason the log is write-ahead.
        """
        settled = {
            (event.correlation_id, event.kind, event.subject)
            for event in self.read()
            if event.phase in ("committed", "failed")
        }
        return [
            event
            for event in self.read()
            if event.phase == "intent"
            and (event.correlation_id, event.kind, event.subject) not in settled
        ]


class NullAuditLog(AuditLog):
    """An audit log that discards events.

    For dry runs and for tests that are not asserting on audit behaviour. Deliberately
    a subclass so a caller cannot accidentally be handed ``None`` and skip logging: the
    only way to not log is to ask for it by name.
    """

    def __init__(self, *, tenant_id: str = "unknown", actor: str = "nova") -> None:
        self.path = Path(os.devnull)
        self.tenant_id = tenant_id
        self.actor = actor
        self._lock = threading.Lock()

    def _append(self, event: AuditEvent) -> AuditEvent:
        return event

    def read(self) -> list[AuditEvent]:
        return []
