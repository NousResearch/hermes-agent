#!/usr/bin/env python3
"""Offline-safe YouTube / transcript acquisition hardening layer.

Implements the taxonomy, result/evidence schema, retry/backoff policy, auth
policy, and redaction helper described in the design artifact:

    /home/filip/spearhead-execution/20260528-source-spikes/yt-dlp/followups/
    youtube-transcript-acquisition-hardening.md

This module is the offline foundation for Hermes/Spearhead/Mystra YouTube and
transcript acquisition. It performs **no** network access, **no** media
download, **no** browser/cookie use, and **no** credential storage. Providers
are injected by the caller; the orchestrator only sequences attempts,
classifies outcomes, applies retry/backoff policy, preserves transcript
provenance, and emits redaction-safe evidence.

Authenticated modes are disabled by default. A provider that declares a
non-anonymous ``auth_mode`` is never selected unless an :class:`AuthPolicy`
explicitly approves that mode — see the safety boundaries in the design
artifact and ``docs/security/mystra-source-acquisition-policy.md``.

The module has no third-party dependencies. It optionally layers Hermes'
``agent.redact.redact_sensitive_text`` for generic credential coverage when
importable, but its own YouTube-specific redaction stands alone.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence, Union, runtime_checkable

# ---------------------------------------------------------------------------
# Optional reuse of the Hermes generic secret redactor. Import is best-effort:
# the module must remain usable as a standalone skill script where the repo
# root is not on sys.path. When unavailable we fall back to a no-op and rely
# solely on the YouTube-specific patterns below.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised indirectly; import shape is trivial
    from agent.redact import redact_sensitive_text as _generic_redact
except Exception:  # pragma: no cover - fallback path

    def _generic_redact(text: str, *, force: bool = False, code_file: bool = False) -> str:
        return text


# ===========================================================================
# 1. Taxonomy
# ===========================================================================


class SourceType(str, Enum):
    """Kinds of YouTube source a request can target."""

    VIDEO = "video"
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    TRANSCRIPT = "transcript"        # direct subtitle/caption reference
    AUDIO_MEDIA = "audio_media"      # audio/media fallback (local input only)
    UNSUPPORTED = "unsupported"      # ambiguous / non-YouTube / malformed


class AcquisitionStatus(str, Enum):
    """Typed acquisition outcomes (design §4)."""

    OK = "OK"
    PARTIAL_METADATA_ONLY = "PARTIAL_METADATA_ONLY"
    NO_TRANSCRIPT_AVAILABLE = "NO_TRANSCRIPT_AVAILABLE"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    AUTH_COOKIE_STALE = "AUTH_COOKIE_STALE"
    CAPTCHA_REQUIRED = "CAPTCHA_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    GEO_RESTRICTED = "GEO_RESTRICTED"
    DRM_OR_MEDIA_BLOCKED = "DRM_OR_MEDIA_BLOCKED"
    TOKEN_REQUIRED = "TOKEN_REQUIRED"
    NETWORK_TRANSIENT = "NETWORK_TRANSIENT"
    UNSUPPORTED_URL = "UNSUPPORTED_URL"


class TranscriptKind(str, Enum):
    """How a transcript was produced (design §6, transcript provenance)."""

    NONE = "none"
    MANUAL = "manual"
    AUTOMATIC = "automatic"          # ASR captions
    TRANSLATED = "translated"
    LIVE_CHAT = "live_chat"


class AuthMode(str, Enum):
    """Acquisition auth modes (design §6). Only ``ANONYMOUS_PUBLIC`` is on by
    default; every other mode requires explicit approval via :class:`AuthPolicy`."""

    ANONYMOUS_PUBLIC = "anonymous_public"
    APPROVED_COOKIE_FILE = "approved_cookie_file"
    APPROVED_OAUTH_OR_API_KEY = "approved_oauth_or_api_key"
    BROWSER_SESSION_IMPORT = "browser_session_import"
    MANUAL_HUMAN_BROWSER = "manual_human_browser"


# Statuses that may be retried in place (bounded). Everything else is a hard
# class: auth/human gates, geo, DRM, token, unsupported, and the success-ish
# partial states are never blindly retried (design §4 retry table).
RETRYABLE_STATUSES = frozenset({
    AcquisitionStatus.NETWORK_TRANSIENT,
    AcquisitionStatus.RATE_LIMITED,
})

# Hard stops: encountering one halts the whole acquisition — no cross-provider
# fallback, because falling back would mean improvising auth or bypassing a
# gate (design §3 "auth/browser fallback never auto-enabled", §7).
HARD_STOP_STATUSES = frozenset({
    AcquisitionStatus.LOGIN_REQUIRED,
    AcquisitionStatus.AUTH_COOKIE_STALE,
    AcquisitionStatus.CAPTCHA_REQUIRED,
    AcquisitionStatus.GEO_RESTRICTED,
    AcquisitionStatus.UNSUPPORTED_URL,
})

# Precedence for choosing the final status across multiple provider attempts.
# Earlier = preferred. A hard stop short-circuits before this is consulted.
_FINAL_PRECEDENCE = (
    AcquisitionStatus.OK,
    AcquisitionStatus.PARTIAL_METADATA_ONLY,
    AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE,
    AcquisitionStatus.TOKEN_REQUIRED,
    AcquisitionStatus.DRM_OR_MEDIA_BLOCKED,
    AcquisitionStatus.RATE_LIMITED,
    AcquisitionStatus.NETWORK_TRANSIENT,
    AcquisitionStatus.LOGIN_REQUIRED,
    AcquisitionStatus.AUTH_COOKIE_STALE,
    AcquisitionStatus.CAPTCHA_REQUIRED,
    AcquisitionStatus.GEO_RESTRICTED,
    AcquisitionStatus.UNSUPPORTED_URL,
)

# Default next-safe-action per terminal status (design §7).
_NEXT_SAFE_ACTION = {
    AcquisitionStatus.OK: "Transcript/metadata acquired; proceed with extraction.",
    AcquisitionStatus.PARTIAL_METADATA_ONLY: (
        "Use metadata only; do not fabricate transcript text. Ask EMA/Filip "
        "whether an approved auth/token mode is permitted for the transcript."
    ),
    AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE: (
        "No public transcript exists for the requested language(s); record "
        "observed languages and proceed metadata-only or try a language fallback."
    ),
    AcquisitionStatus.LOGIN_REQUIRED: (
        "Stop. Sign-in/account state required — request explicit EMA/Filip "
        "approval for an authenticated case, or use a public alternative."
    ),
    AcquisitionStatus.AUTH_COOKIE_STALE: (
        "Stop. Approved auth appears stale; do not refresh/rotate/login. "
        "Emit a redacted warning and request re-approval."
    ),
    AcquisitionStatus.CAPTCHA_REQUIRED: (
        "Stop. Anti-abuse challenge encountered; never solve automatically."
    ),
    AcquisitionStatus.RATE_LIMITED: (
        "Back off and retry later within deadline; no credential workaround."
    ),
    AcquisitionStatus.GEO_RESTRICTED: (
        "Stop. Region-blocked; proxy/VPN requires a separate approval card."
    ),
    AcquisitionStatus.DRM_OR_MEDIA_BLOCKED: (
        "Continue with transcript/metadata if available; media path needs a "
        "separate approval card."
    ),
    AcquisitionStatus.TOKEN_REQUIRED: (
        "Stop unless an approved token adapter is configured; degrade to "
        "metadata-only otherwise."
    ),
    AcquisitionStatus.NETWORK_TRANSIENT: (
        "Transient network failure persisted after bounded retries; try again later."
    ),
    AcquisitionStatus.UNSUPPORTED_URL: (
        "URL/provider unsupported or ambiguous; ask for a valid public YouTube reference."
    ),
}


def next_safe_action(status: AcquisitionStatus) -> str:
    """Return the default next-safe-action string for a terminal status."""
    return _NEXT_SAFE_ACTION.get(AcquisitionStatus(status), "Review acquisition result manually.")


# ===========================================================================
# 2. Source classification / URL normalization (design §3 step 0)
# ===========================================================================

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

_YOUTUBE_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be",
})


@dataclass
class SourceRef:
    """A normalized, non-secret reference to a YouTube source.

    The raw input URL is **never** retained — only its sha256 hash and the
    extracted non-secret identifiers — so signed URLs and tokens embedded in
    query strings cannot leak through the evidence object.
    """

    source_type: SourceType
    url_hash: str
    video_id: Optional[str] = None
    playlist_id: Optional[str] = None
    channel: Optional[str] = None      # channel id (UC...) or @handle
    reason: Optional[str] = None       # why UNSUPPORTED, if applicable

    @property
    def is_supported(self) -> bool:
        return self.source_type is not SourceType.UNSUPPORTED


def hash_url(url: str) -> str:
    """Return a stable, non-secret ``sha256:`` digest of a URL/identifier."""
    digest = hashlib.sha256(url.strip().encode("utf-8", "replace")).hexdigest()
    return f"sha256:{digest}"


def _parse_query(query: str) -> dict:
    out: dict = {}
    for pair in query.split("&"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        out.setdefault(key, value)
    return out


def classify_source(url_or_id: str) -> SourceRef:
    """Classify a URL/reference into a :class:`SourceRef`.

    Recognizes watch / youtu.be / shorts / embed / live video URLs, playlist
    URLs, channel URLs (``/channel/UC...``, ``/@handle``, ``/c/``, ``/user/``),
    and bare 11-char video ids. Anything else returns ``UNSUPPORTED``.
    Performs pure string parsing only — no network.
    """
    if url_or_id is None:
        return SourceRef(SourceType.UNSUPPORTED, hash_url(""), reason="empty input")
    raw = url_or_id.strip()
    url_hash = hash_url(raw)
    if not raw:
        return SourceRef(SourceType.UNSUPPORTED, url_hash, reason="empty input")

    # Bare 11-char video id (no scheme, no slash).
    if "/" not in raw and "." not in raw and _VIDEO_ID_RE.match(raw):
        return SourceRef(SourceType.VIDEO, url_hash, video_id=raw)

    # Tolerate scheme-less URLs like "youtu.be/abc".
    work = raw if "://" in raw else "https://" + raw
    m = re.match(r"^[a-z][a-z0-9+.-]*://([^/?#]+)([^?#]*)(?:\?([^#]*))?", work, re.I)
    if not m:
        return SourceRef(SourceType.UNSUPPORTED, url_hash, reason="unparseable URL")
    host = m.group(1).lower().split("@")[-1].split(":")[0]
    path = m.group(2) or ""
    query = _parse_query(m.group(3) or "")

    if host not in _YOUTUBE_HOSTS:
        return SourceRef(SourceType.UNSUPPORTED, url_hash, reason=f"non-YouTube host: {host}")

    # youtu.be/<id>
    if host == "youtu.be":
        vid = path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(vid):
            ref = SourceRef(SourceType.VIDEO, url_hash, video_id=vid)
            if query.get("list"):
                ref.playlist_id = query["list"]
            return ref
        return SourceRef(SourceType.UNSUPPORTED, url_hash, reason="youtu.be without video id")

    seg = [s for s in path.split("/") if s]

    # Channel forms.
    if seg:
        if seg[0] == "channel" and len(seg) > 1:
            return SourceRef(SourceType.CHANNEL, url_hash, channel=seg[1])
        if seg[0].startswith("@"):
            return SourceRef(SourceType.CHANNEL, url_hash, channel=seg[0])
        if seg[0] in ("c", "user") and len(seg) > 1:
            return SourceRef(SourceType.CHANNEL, url_hash, channel=seg[1])
        # watch / shorts / embed / live / v
        if seg[0] == "watch":
            vid = query.get("v", "")
            if _VIDEO_ID_RE.match(vid):
                ref = SourceRef(SourceType.VIDEO, url_hash, video_id=vid)
                if query.get("list"):
                    ref.playlist_id = query["list"]
                return ref
            if query.get("list"):
                return SourceRef(SourceType.PLAYLIST, url_hash, playlist_id=query["list"])
            return SourceRef(SourceType.UNSUPPORTED, url_hash, reason="watch without video id")
        if seg[0] in ("shorts", "embed", "live", "v") and len(seg) > 1:
            vid = seg[1]
            if _VIDEO_ID_RE.match(vid):
                return SourceRef(SourceType.VIDEO, url_hash, video_id=vid)
            return SourceRef(SourceType.UNSUPPORTED, url_hash, reason=f"{seg[0]} without video id")
        if seg[0] == "playlist":
            if query.get("list"):
                return SourceRef(SourceType.PLAYLIST, url_hash, playlist_id=query["list"])
            return SourceRef(SourceType.UNSUPPORTED, url_hash, reason="playlist without list id")

    # Path-less URL but a list= query → playlist.
    if query.get("list"):
        return SourceRef(SourceType.PLAYLIST, url_hash, playlist_id=query["list"])

    return SourceRef(SourceType.UNSUPPORTED, url_hash, reason="ambiguous YouTube URL")


# ===========================================================================
# 3. Redaction helper (design §6, §8)
# ===========================================================================

# Each entry: (category, compiled-pattern, replacement). Replacement keeps the
# key/label for debuggability and masks only the value. Categories are surfaced
# in evidence ``redactions`` lists so reviewers can see what was scrubbed
# without ever seeing the raw value.
_VALUE_RE = r"[^\s\"',;&)}\]]+"
_REDACTORS: list[tuple[str, re.Pattern, str]] = [
    # Authorization / bearer headers.
    ("auth_header", re.compile(r"(?im)^(authorization)\s*:\s*.+$"), r"\1: ***"),
    ("auth_header", re.compile(r"(?i)(authorization|x-goog-authuser|x-goog-visitor-id)"
                               r"(\s*[:=]\s*)" + _VALUE_RE), r"\1\2***"),
    # Cookie headers and named YouTube/Google session cookies.
    ("cookie", re.compile(r"(?im)^((?:set-)?cookie)\s*:\s*.+$"), r"\1: ***"),
    ("cookie", re.compile(r"(?i)\b(SAPISID|__Secure-[A-Za-z0-9_-]+|HSID|SSID|APISID"
                          r"|SIDCC|LOGIN_INFO|SID|PSID|__Host-[A-Za-z0-9_-]+)"
                          r"=" + _VALUE_RE), r"\1=***"),
    # PO token, visitor data, Data Sync ID — fragile YouTube tokens (design §6).
    ("po_token", re.compile(r"(?i)(po_?token)(\"?\s*[:=]\s*\"?)" + _VALUE_RE), r"\1\2***"),
    ("visitor_data", re.compile(r"(?i)(visitor_?data)(\"?\s*[:=]\s*\"?)" + _VALUE_RE), r"\1\2***"),
    ("datasync_id", re.compile(r"(?i)(data_?sync_?id)(\"?\s*[:=]\s*\"?)" + _VALUE_RE), r"\1\2***"),
]

# Substrings that hint a string may contain a signed media URL. Such URLs are
# never emitted whole; callers should hash them. We still flag the category.
_SIGNED_URL_HINTS = ("googlevideo.com", "&sig=", "?sig=", "signature=", "&pot=", "?pot=")


def detect_redactions(text: str) -> list[str]:
    """Return the sorted, de-duplicated redaction categories present in ``text``.

    Used to populate the evidence ``redactions`` field without exposing values.
    """
    if not text:
        return []
    cats = set()
    for category, pattern, _ in _REDACTORS:
        if pattern.search(text):
            cats.add(category)
    low = text.lower()
    if any(h in low for h in _SIGNED_URL_HINTS):
        cats.add("signed_url")
    return sorted(cats)


def redact_acquisition_text(text: Optional[str]) -> Optional[str]:
    """Scrub cookies, auth headers, PO/visitor/Data Sync tokens, and signed
    media URLs from ``text``, then apply Hermes' generic secret redactor.

    Safe to call on any string; non-matching text passes through unchanged.
    ``None`` is returned unchanged. This is the single chokepoint every value
    bound for Discord/Kanban/logs/artifacts/prompts must pass through.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    out = text
    for _category, pattern, replacement in _REDACTORS:
        out = pattern.sub(replacement, out)
    # Hash-replace anything that looks like a signed googlevideo media URL so a
    # full signed URL can never survive in logs/evidence.
    out = re.sub(
        r"https?://[^\s\"']*googlevideo\.com[^\s\"']*",
        lambda m: hash_url(m.group(0)),
        out,
        flags=re.I,
    )
    # Generic credential coverage (sk-..., ghp_..., JWTs, URL query secrets).
    # force=True so this safety boundary never depends on the global toggle.
    out = _generic_redact(out, force=True)
    return out


def _redact_artifacts(value):
    """Recursively redact string leaves of an artifacts structure."""
    if isinstance(value, str):
        return redact_acquisition_text(value)
    if isinstance(value, dict):
        return {k: _redact_artifacts(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_artifacts(v) for v in value]
    return value


# ===========================================================================
# 4. Retry / backoff policy (design §4, §5)
# ===========================================================================


@dataclass
class RetryPolicy:
    """Bounded retry/backoff for transient/throttle classes only."""

    max_transient_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.25

    def is_retryable(self, status: AcquisitionStatus) -> bool:
        return AcquisitionStatus(status) in RETRYABLE_STATUSES

    def backoff_delay(
        self,
        attempt: int,
        *,
        retry_after: Optional[float] = None,
        rng: Optional[Callable[[float, float], float]] = None,
    ) -> float:
        """Exponential backoff with +/- ``jitter`` (design §5).

        ``attempt`` is 1-based. Honors ``retry_after`` when provided (taking the
        max of computed and server-requested delay), then clamps to ``max_delay``.
        ``rng`` is injectable for deterministic tests; defaults to ``random.uniform``.
        """
        base = self.base_delay * (2 ** max(0, attempt - 1))
        if rng is None:
            import random
            rng = random.uniform
        low = base * (1 - self.jitter)
        high = base * (1 + self.jitter)
        delay = rng(low, high)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, self.max_delay)


# ===========================================================================
# 5. Auth policy (design §6)
# ===========================================================================


@dataclass
class AuthPolicy:
    """Controls which auth modes a provider may use.

    Default-deny: only ``ANONYMOUS_PUBLIC`` is permitted. Any other mode must be
    explicitly approved with an ``approval_id`` (the EMA/Filip approval comment
    or gate id). This is what keeps cookie/oauth/browser providers off the
    default path.
    """

    mode: AuthMode = AuthMode.ANONYMOUS_PUBLIC
    approved: bool = False
    approval_id: Optional[str] = None

    def permits(self, provider_auth_mode: AuthMode) -> bool:
        """True iff a provider declaring ``provider_auth_mode`` may run."""
        mode = AuthMode(provider_auth_mode)
        if mode is AuthMode.ANONYMOUS_PUBLIC:
            return True
        return self.approved and self.mode == mode and bool(self.approval_id)


# ===========================================================================
# 6. Provider / director interface and result/evidence schema (design §3, §8)
# ===========================================================================


@dataclass
class TranscriptProvenance:
    """Transcript provenance — preserved end to end (design §8)."""

    kind: TranscriptKind = TranscriptKind.NONE
    original_language: Optional[str] = None
    requested_language: Optional[str] = None
    translated_from: Optional[str] = None
    is_asr: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value if isinstance(self.kind, TranscriptKind) else self.kind,
            "original_language": self.original_language,
            "requested_language": self.requested_language,
            "translated_from": self.translated_from,
            "is_asr": self.is_asr,
        }


@dataclass
class AcquisitionRequest:
    """A normalized acquisition request handed to providers."""

    source: SourceRef
    requested_languages: list = field(default_factory=list)
    allow_media: bool = False          # media never acquired unless explicit
    request_id: Optional[str] = None


@dataclass
class ProviderResult:
    """What an injected provider returns from :meth:`Provider.fetch`."""

    status: AcquisitionStatus
    provenance: Optional[TranscriptProvenance] = None
    artifacts: dict = field(default_factory=dict)
    redactions: list = field(default_factory=list)
    retry_after: Optional[float] = None
    duration_ms: Optional[int] = None


@runtime_checkable
class Provider(Protocol):
    """Director-style provider contract. Implementations declare support and
    fail explicitly with a typed status — never improvise credentials."""

    label: str
    phase: str
    auth_mode: AuthMode

    def supports(self, request: AcquisitionRequest) -> bool: ...

    def fetch(self, request: AcquisitionRequest) -> ProviderResult: ...


@dataclass
class ProviderAttempt:
    """One recorded provider attempt for the evidence log."""

    label: str
    phase: str
    attempt: int
    outcome: AcquisitionStatus
    retryable: bool
    duration_ms: Optional[int] = None
    redactions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "phase": self.phase,
            "attempt": self.attempt,
            "outcome": self.outcome.value if isinstance(self.outcome, AcquisitionStatus) else self.outcome,
            "retryable": self.retryable,
            "duration_ms": self.duration_ms,
            "redactions": list(self.redactions),
        }


@dataclass
class AcquisitionResult:
    """Final acquisition result. ``to_evidence`` produces a redaction-safe dict."""

    request_id: str
    source: SourceRef
    requested_languages: list
    status: AcquisitionStatus
    providers: list = field(default_factory=list)
    transcript_provenance: TranscriptProvenance = field(default_factory=TranscriptProvenance)
    artifacts: dict = field(default_factory=dict)
    next_safe_action: str = ""

    def to_evidence(self) -> dict:
        """Render the redaction-safe evidence object (design §8 shape).

        Contains only non-secret identifiers (url hash, video/playlist/channel
        id), typed outcomes, provenance, redaction categories, and the next safe
        action. All artifact string leaves pass through the redactor.
        """
        return {
            "request_id": self.request_id,
            "url_hash": self.source.url_hash,
            "source_type": self.source.source_type.value,
            "video_id": self.source.video_id,
            "playlist_id": self.source.playlist_id,
            "channel": self.source.channel,
            "requested_languages": list(self.requested_languages),
            "status": self.status.value if isinstance(self.status, AcquisitionStatus) else self.status,
            "providers": [p.to_dict() for p in self.providers],
            "transcript_provenance": self.transcript_provenance.to_dict(),
            "artifacts": _redact_artifacts(self.artifacts),
            "next_safe_action": self.next_safe_action,
        }


def _derive_request_id(url_hash: str) -> str:
    """Stable, non-secret request id derived from the url hash (no randomness)."""
    digest = url_hash.split(":", 1)[-1]
    return f"yt-{digest[:12]}"


# ===========================================================================
# 7. Orchestrator / director (design §3)
# ===========================================================================


def acquire(
    target: Union[str, AcquisitionRequest],
    providers: Sequence[Provider],
    *,
    requested_languages: Optional[list] = None,
    retry_policy: Optional[RetryPolicy] = None,
    auth_policy: Optional[AuthPolicy] = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: Optional[Callable[[float, float], float]] = None,
    clock: Callable[[], float] = time.monotonic,
    request_id: Optional[str] = None,
) -> AcquisitionResult:
    """Run the layered acquisition state machine over ``providers``.

    Sequences provider attempts in order, applying:
      * source classification (immediate ``UNSUPPORTED_URL`` for bad input);
      * the default-deny auth policy (non-anonymous providers skipped unless
        explicitly approved);
      * bounded in-place retries for transient/throttle classes;
      * cross-provider fallback for partial/no-transcript/token/DRM outcomes;
      * hard-stop short-circuit for login/captcha/geo/stale-auth gates.

    Performs no I/O itself — providers do (or, in tests, return canned results).
    Returns an :class:`AcquisitionResult` whose ``to_evidence()`` is safe to log.
    """
    retry_policy = retry_policy or RetryPolicy()
    auth_policy = auth_policy or AuthPolicy()

    request = _coerce_request(target, requested_languages, request_id)
    rid = request.request_id or _derive_request_id(request.source.url_hash)

    # Step 0: unsupported input short-circuits before any provider runs.
    if not request.source.is_supported:
        return AcquisitionResult(
            request_id=rid,
            source=request.source,
            requested_languages=request.requested_languages,
            status=AcquisitionStatus.UNSUPPORTED_URL,
            providers=[],
            transcript_provenance=TranscriptProvenance(),
            artifacts={"reason": request.source.reason} if request.source.reason else {},
            next_safe_action=next_safe_action(AcquisitionStatus.UNSUPPORTED_URL),
        )

    attempts: list[ProviderAttempt] = []
    best_status: Optional[AcquisitionStatus] = None
    best_provenance = TranscriptProvenance(
        requested_language=(request.requested_languages or [None])[0]
    )
    best_artifacts: dict = {}
    hard_stop = False

    for provider in providers:
        # Default-deny: never select a non-anonymous provider without approval.
        if not auth_policy.permits(getattr(provider, "auth_mode", AuthMode.ANONYMOUS_PUBLIC)):
            continue
        if not provider.supports(request):
            continue

        attempt_no = 0
        while True:
            attempt_no += 1
            start = clock()
            result = provider.fetch(request)
            duration_ms = result.duration_ms
            if duration_ms is None:
                duration_ms = int((clock() - start) * 1000)
            status = AcquisitionStatus(result.status)
            retryable = retry_policy.is_retryable(status)

            attempts.append(ProviderAttempt(
                label=provider.label,
                phase=getattr(provider, "phase", "acquire"),
                attempt=attempt_no,
                outcome=status,
                retryable=retryable,
                duration_ms=duration_ms,
                redactions=list(result.redactions),
            ))

            # Capture the best outcome/provenance/artifacts seen so far.
            if _is_better(status, best_status):
                best_status = status
                if result.provenance is not None:
                    best_provenance = result.provenance
                if result.artifacts:
                    best_artifacts = dict(result.artifacts)

            if status in HARD_STOP_STATUSES:
                hard_stop = True
                break

            if status is AcquisitionStatus.OK:
                # Terminal success — stop the whole acquisition.
                return _finalize(rid, request, attempts, best_provenance, best_artifacts, status)

            if retryable and attempt_no <= retry_policy.max_transient_retries:
                delay = retry_policy.backoff_delay(
                    attempt_no, retry_after=result.retry_after, rng=rng
                )
                sleeper(delay)
                continue

            # Non-retryable, or retries exhausted → try the next provider.
            break

        if hard_stop:
            break

    final_status = best_status or AcquisitionStatus.NETWORK_TRANSIENT
    return _finalize(rid, request, attempts, best_provenance, best_artifacts, final_status)


def _coerce_request(
    target: Union[str, AcquisitionRequest],
    requested_languages: Optional[list],
    request_id: Optional[str],
) -> AcquisitionRequest:
    if isinstance(target, AcquisitionRequest):
        if requested_languages is not None:
            target.requested_languages = list(requested_languages)
        if request_id is not None:
            target.request_id = request_id
        return target
    source = classify_source(target)
    return AcquisitionRequest(
        source=source,
        requested_languages=list(requested_languages or []),
        request_id=request_id,
    )


def _is_better(status: AcquisitionStatus, current: Optional[AcquisitionStatus]) -> bool:
    if current is None:
        return True
    return _FINAL_PRECEDENCE.index(status) < _FINAL_PRECEDENCE.index(current)


def _finalize(
    rid: str,
    request: AcquisitionRequest,
    attempts: list,
    provenance: TranscriptProvenance,
    artifacts: dict,
    status: AcquisitionStatus,
) -> AcquisitionResult:
    # No-hallucination guard: if no transcript was actually acquired, the
    # provenance kind must stay NONE — never invent transcript provenance.
    if status is not AcquisitionStatus.OK and provenance.kind is not TranscriptKind.NONE:
        if status in (AcquisitionStatus.PARTIAL_METADATA_ONLY,
                      AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE):
            provenance = TranscriptProvenance(
                requested_language=provenance.requested_language,
                original_language=provenance.original_language,
            )
    return AcquisitionResult(
        request_id=rid,
        source=request.source,
        requested_languages=request.requested_languages,
        status=status,
        providers=attempts,
        transcript_provenance=provenance,
        artifacts=artifacts,
        next_safe_action=next_safe_action(status),
    )
