"""Trusted, request-scoped provenance for exact local file slices.

This module deliberately has no provider-facing API.  It records only grants
created by the two trusted file-read surfaces; the firewall is the later
consumer that decides whether a grant is eligible for a particular egress.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import os
from pathlib import Path
import stat
from threading import Lock, RLock
from typing import Iterator
import uuid

from agent.file_safety import get_read_block_error
from agent.llm_egress_firewall import SourceGrant, source_grant_digest
from agent.redact import redact_sensitive_text


MAX_SOURCE_SLICE_LINES = 2_000
MAX_SOURCE_SLICE_BYTES = 262_144
MAX_GRANTS_PER_REQUEST = 64
DEFAULT_POLICY_DIGEST = sha256(b"hermes-llm-egress-policy-v1").hexdigest()


class SourceProvenanceError(ValueError):
    """A trusted producer could not establish file-slice provenance."""


@dataclass(frozen=True, slots=True)
class SourceProvenanceContext:
    """Per-call identity available only while a trusted file read executes."""

    registry: "SourceProvenanceRegistry"
    session_id: str
    turn_id: str
    request_id: str
    policy_digest: str


_active_context: ContextVar[SourceProvenanceContext | None] = ContextVar(
    "source_provenance_context",
    default=None,
)


def active_source_provenance() -> SourceProvenanceContext | None:
    """Return the trusted context for the current tool dispatch, if any."""

    return _active_context.get()


@contextmanager
def activate_source_provenance(
    registry: "SourceProvenanceRegistry",
    *,
    session_id: str,
    turn_id: str,
    request_id: str,
    policy_digest: str,
) -> Iterator[SourceProvenanceContext]:
    """Temporarily make an authenticated request identity available to reads."""

    values = (session_id, turn_id, request_id, policy_digest)
    if not isinstance(registry, SourceProvenanceRegistry) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise SourceProvenanceError("missing_identity")
    context = SourceProvenanceContext(registry, *values)
    token = _active_context.set(context)
    try:
        yield context
    finally:
        _active_context.reset(token)


class SourceProvenanceRegistry:
    """In-memory grants keyed solely by their bound request identity."""

    def __init__(self) -> None:
        self._grants: dict[str, list[SourceGrant]] = {}
        self._validated_presentations: dict[str, SourceGrant] = {}
        self._lock = RLock()

    def issue_file_slice(
        self,
        *,
        path: Path,
        line_start: int,
        line_end: int,
        content: bytes,
        session_id: str,
        turn_id: str,
        request_id: str,
        policy_digest: str,
    ) -> SourceGrant:
        """Grant exactly the current canonical bytes, or reject the producer.

        The caller supplies the bytes it just read, but they are never trusted
        by themselves: this method re-reads the requested bounded source slice
        from the resolved regular file, checks the sensitive-path policy and
        forced redaction, then compares the two byte strings.
        """

        if not isinstance(path, Path):
            path = Path(path)
        original_path = Path(path).expanduser()
        if _contains_symlink_component(original_path):
            raise SourceProvenanceError("symlink_path")
        if not isinstance(content, bytes):
            raise SourceProvenanceError("invalid_content")
        if (
            not isinstance(line_start, int)
            or not isinstance(line_end, int)
            or line_start < 1
            or line_end < line_start
            or line_end - line_start + 1 > MAX_SOURCE_SLICE_LINES
        ):
            raise SourceProvenanceError("invalid_line_range")
        if len(content) > MAX_SOURCE_SLICE_BYTES:
            raise SourceProvenanceError("slice_too_large")
        identities = (session_id, turn_id, request_id, policy_digest)
        if not all(isinstance(value, str) and value for value in identities):
            raise SourceProvenanceError("missing_identity")

        try:
            descriptor = _open_verified_source(original_path)
        except SourceProvenanceError:
            raise
        except OSError as exc:
            raise SourceProvenanceError("canonical_path_unavailable") from exc
        try:
            canonical = original_path.resolve(strict=True)
            if not canonical.is_file():
                raise SourceProvenanceError("not_regular_file")
            opened_stat = os.fstat(descriptor)
            named_stat = os.stat(canonical, follow_symlinks=False)
            if (opened_stat.st_dev, opened_stat.st_ino) != (named_stat.st_dev, named_stat.st_ino):
                raise SourceProvenanceError("source_changed")
        except SourceProvenanceError:
            os.close(descriptor)
            raise
        except OSError as exc:
            os.close(descriptor)
            raise SourceProvenanceError("canonical_path_unavailable") from exc
        try:
            blocked = get_read_block_error(str(canonical))
        except SourceProvenanceError:
            os.close(descriptor)
            raise
        except Exception as exc:
            os.close(descriptor)
            raise SourceProvenanceError("read_policy_unavailable") from exc
        if blocked is not None:
            os.close(descriptor)
            raise SourceProvenanceError("sensitive_path")

        try:
            approved = _read_bounded_slice_fd(descriptor, line_start, line_end)
            if not compare_digest(sha256(content).digest(), sha256(approved).digest()):
                raise SourceProvenanceError("content_mismatch")
            try:
                source_text = approved.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SourceProvenanceError("non_text_source") from exc
            try:
                if redact_sensitive_text(
                    source_text,
                    force=True,
                    file_read=True,
                    redact_url_credentials=True,
                ) != source_text:
                    raise SourceProvenanceError("redaction_changed_content")
            except SourceProvenanceError:
                raise
            except Exception as exc:
                raise SourceProvenanceError("redaction_unavailable") from exc
        finally:
            os.close(descriptor)

        grant = SourceGrant(
            canonical_path=canonical,
            display_path=_safe_display_path(original_path),
            line_start=line_start,
            line_end=line_end,
            content_sha256=sha256(approved).hexdigest(),
            byte_count=len(approved),
            session_id=session_id,
            turn_id=turn_id,
            request_id=request_id,
            policy_digest=policy_digest,
        )
        with self._lock:
            if len(self._grants.get(request_id, ())) >= MAX_GRANTS_PER_REQUEST:
                raise SourceProvenanceError("grant_limit_exceeded")
            self._grants.setdefault(request_id, []).append(grant)
        return grant

    @contextmanager
    def request_scope(self, request_id: str) -> Iterator["SourceProvenanceRegistry"]:
        """Bound grants to one request and clear them on every exit path."""

        if not isinstance(request_id, str) or not request_id:
            raise SourceProvenanceError("missing_identity")
        try:
            yield self
        finally:
            self.clear_request(request_id)

    def grants_for_request(self, request_id: str) -> tuple[SourceGrant, ...]:
        """Return an immutable snapshot of grants bound to ``request_id``."""

        if not isinstance(request_id, str) or not request_id:
            return ()
        with self._lock:
            return tuple(self._grants.get(request_id, ()))

    def clear_request(self, request_id: str) -> None:
        """Discard grants at the end of a request without touching other turns."""

        if not isinstance(request_id, str) or not request_id:
            return
        with self._lock:
            self._grants.pop(request_id, None)

    def clear_turn(self, turn_id: str) -> None:
        """Discard every request grant belonging to one completed turn."""

        if not isinstance(turn_id, str) or not turn_id:
            return
        with self._lock:
            stale = [
                request_id
                for request_id, grants in self._grants.items()
                if any(grant.turn_id == turn_id for grant in grants)
            ]
            for request_id in stale:
                self._grants.pop(request_id, None)
            stale_presentations = [
                digest
                for digest, grant in self._validated_presentations.items()
                if grant.turn_id == turn_id
            ]
            for digest in stale_presentations:
                self._validated_presentations.pop(digest, None)

    def remember_validated_presentations(
        self, grants: tuple[SourceGrant, ...]
    ) -> None:
        """Retain grants only after their presentation passed final egress."""

        with self._lock:
            for grant in grants:
                if isinstance(grant, SourceGrant):
                    self._validated_presentations[source_grant_digest(grant)] = grant

    def rebind_validated_presentation(
        self,
        digest: str,
        *,
        original_request_id: str,
        session_id: str,
        turn_id: str,
        request_id: str,
        policy_digest: str,
    ) -> SourceGrant | None:
        """Re-verify and bind an earlier validated presentation to a later call."""

        with self._lock:
            original = self._validated_presentations.get(digest)
        if original is None:
            return None

        if (
            original.session_id != session_id
            or original.turn_id != turn_id
            or original.policy_digest != policy_digest
            or original.request_id != original_request_id
            or not _is_later_request(original.request_id, request_id, turn_id)
        ):
            return None
        try:
            content = _read_bounded_slice(
                original.canonical_path,
                original.line_start,
                original.line_end,
            )
            return self.issue_file_slice(
                path=original.canonical_path,
                line_start=original.line_start,
                line_end=original.line_end,
                content=content,
                session_id=session_id,
                turn_id=turn_id,
                request_id=request_id,
                policy_digest=policy_digest,
            )
        except (OSError, SourceProvenanceError):
            return None


_registry_install_lock = Lock()


def source_provenance_registry_for_agent(agent) -> SourceProvenanceRegistry:
    """Return one atomically installed provenance registry per agent.

    Concurrent first-use file reads may both construct a candidate, but the
    double-check under this lock ensures every caller receives the same
    installed registry.  Construction stays outside the lock so alternate
    registry implementations cannot serialize or deadlock unrelated agents.
    """

    registry = getattr(agent, "_source_provenance_registry", None)
    if isinstance(registry, SourceProvenanceRegistry):
        return registry
    candidate = SourceProvenanceRegistry()
    with _registry_install_lock:
        registry = getattr(agent, "_source_provenance_registry", None)
        if not isinstance(registry, SourceProvenanceRegistry):
            registry = candidate
            agent._source_provenance_registry = registry
    return registry


def provenance_kwargs_for_agent(
    agent,
    *,
    request_id: str | None = None,
    establish_turn: bool = False,
) -> dict[str, object]:
    """Return authenticated context-reference kwargs for a live agent turn."""

    session_id = str(getattr(agent, "session_id", "") or "")
    registry = source_provenance_registry_for_agent(agent)

    if establish_turn:
        previous_turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        previous_request_id = str(getattr(agent, "_current_api_request_id", "") or "")
        if previous_request_id:
            registry.clear_request(previous_request_id)
        if previous_turn_id:
            registry.clear_turn(previous_turn_id)
        turn_id = f"{session_id or 'session'}:context:{uuid.uuid4().hex[:8]}"
        agent._source_provenance_pending_turn_id = turn_id
    else:
        turn_id = str(getattr(agent, "_current_turn_id", "") or "")
    resolved_request_id = str(
        request_id
        # The conversation loop increments ``api_call_count`` before building
        # its first provider request, so the first consuming identity is 1.
        or (f"{turn_id}:api:1" if establish_turn and turn_id else "")
        or getattr(agent, "_current_api_request_id", "")
        or (f"{turn_id}:context" if turn_id else "")
    )
    policy_digest = str(
        getattr(agent, "_llm_egress_policy_digest", "")
        or getattr(agent, "llm_egress_policy_digest", "")
        or DEFAULT_POLICY_DIGEST
    )
    if not all((session_id, turn_id, resolved_request_id, policy_digest)):
        return {}
    return {
        "source_provenance_registry": registry,
        "session_id": session_id,
        "turn_id": turn_id,
        "request_id": resolved_request_id,
        "policy_digest": policy_digest,
    }


def clear_agent_source_provenance(agent, *, request_id: str | None = None) -> None:
    """Clear request/turn grants when a caller abandons context expansion."""

    registry = getattr(agent, "_source_provenance_registry", None)
    if not isinstance(registry, SourceProvenanceRegistry):
        return
    pending_turn_id = str(
        getattr(agent, "_source_provenance_pending_turn_id", "") or ""
    )
    if pending_turn_id:
        registry.clear_turn(pending_turn_id)
        agent._source_provenance_pending_turn_id = None
    resolved_request_id = str(request_id or getattr(agent, "_current_api_request_id", "") or "")
    if resolved_request_id:
        registry.clear_request(resolved_request_id)
        return
    registry.clear_turn(str(getattr(agent, "_current_turn_id", "") or ""))


def following_api_request_id(request_id: str, turn_id: str) -> str:
    """Bind trusted tool output to the API request that will consume it."""

    if not turn_id:
        return request_id
    prefix = f"{turn_id}:api:"
    if request_id.startswith(prefix):
        try:
            return f"{prefix}{int(request_id[len(prefix):]) + 1}"
        except ValueError:
            pass
    # An unparseable current request cannot be projected to an exact future
    # request. Keep the read available but issue no usable grant.
    return ""


def _is_later_request(original: str, candidate: str, turn_id: str) -> bool:
    """Require a strictly later numeric API request in the same turn."""

    prefix = f"{turn_id}:api:"
    if not original.startswith(prefix) or not candidate.startswith(prefix):
        return False
    try:
        return int(candidate[len(prefix) :]) > int(original[len(prefix) :])
    except ValueError:
        return False


def _read_bounded_slice(path: Path, line_start: int, line_end: int) -> bytes:
    """Read the inclusive raw-byte line interval without materializing a file."""

    selected: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < line_start:
                    continue
                if line_number > line_end:
                    break
                total += len(line)
                if total > MAX_SOURCE_SLICE_BYTES:
                    raise SourceProvenanceError("slice_too_large")
                selected.append(line)
    except SourceProvenanceError:
        raise
    except OSError as exc:
        raise SourceProvenanceError("canonical_read_failed") from exc
    if len(selected) != line_end - line_start + 1:
        raise SourceProvenanceError("line_range_unavailable")
    return b"".join(selected)


def _contains_symlink_component(path: Path) -> bool:
    """Check the original spelling before any resolution can erase a link."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                return True
        except OSError as exc:
            raise SourceProvenanceError("symlink_check_failed") from exc
    return False


def _open_verified_source(path: Path) -> int:
    """Open once with no-follow semantics and reject non-regular/hardlinked files."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise SourceProvenanceError("not_regular_file")
        if file_stat.st_nlink != 1:
            raise SourceProvenanceError("unsafe_hardlink")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_bounded_slice_fd(descriptor: int, line_start: int, line_end: int) -> bytes:
    """Read complete selected lines from the already-verified descriptor."""

    selected: list[bytes] = []
    total = 0
    try:
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < line_start:
                    continue
                if line_number > line_end:
                    break
                total += len(line)
                if total > MAX_SOURCE_SLICE_BYTES:
                    raise SourceProvenanceError("slice_too_large")
                selected.append(line)
    except SourceProvenanceError:
        raise
    except OSError as exc:
        raise SourceProvenanceError("canonical_read_failed") from exc
    if len(selected) != line_end - line_start + 1:
        raise SourceProvenanceError("line_range_unavailable")
    return b"".join(selected)


def _safe_display_path(path: Path) -> str:
    """Keep grant metadata portable and free of an absolute home path."""

    try:
        absolute = path if path.is_absolute() else Path.cwd() / path
        return absolute.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name
