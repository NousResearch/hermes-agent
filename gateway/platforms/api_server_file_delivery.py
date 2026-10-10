"""Downloadable file delivery for the OpenAI-compatible API server.

``MEDIA:<path>`` tags in a final response were only ever resolved for images: the sibling
``_resolve_media_to_data_urls`` returns ``None`` for any other suffix, so ``_repl`` hands back
``m.group(0)`` and the literal tag — server path included — crossed the HTTP boundary. A chat
frontend cannot read a server path and cannot render a PDF as an image, so the file never reached
the user. This module carries the second resolution pass: a non-image tag whose path is a readable,
validator-approved, in-cap regular file becomes a signed, expiring, one-shot download link served by
the API server's existing artifact transport (``gateway/browser_control_artifacts.py``).

Why a second pass instead of one combined resolver: images keep their established base64 data-URL
behavior byte-for-byte, including the "oversized/unreadable stays literal" degradation. Running the
download pass over the image pass's output makes that a property of the structure rather than of a
carefully-written branch.

TRUST MODEL — deliberate, scoped, and the load-bearing decision in this module.
The artifact download route is Bearer-gated (``API_SERVER_KEY``) because it was built for
browser-control clients, which hold that key. The consumer of a *file delivery* link is a chat
window: a browser or mobile PWA that authenticates to its own backend and never holds
``API_SERVER_KEY``. Requiring Bearer is exactly what makes the existing route useless here, so
delivery links authenticate with an HMAC signature instead::

    /v1/artifacts/download/<32-hex-id>?expires=<unix-seconds>&signature=<hex>
    signature = HMAC-SHA256(per-install secret, f"{artifact_id}\\x00{expires}")

What that buys, and why each piece is required:

* **The URL is the capability.** The artifact id and the signature are both unguessable, both are
  required, and the signature is bound to the id and the expiry — so neither can be swapped.
* **The link expires on its own.** The signature is valid only until the signed expiry (the store
  TTL), and the store independently expires the entry. A GET additionally CONSUMES it, so a link is
  one-shot even if it is copied out of the chat.
* **Every failure is a 404**, never a 403: unknown id, expired, already consumed, tampered
  signature, wrong profile. A distinguishable "exists but forbidden" would make the route an
  existence oracle.
* **The secret is per install and per profile home**, never the API key. Rotating
  ``API_SERVER_KEY`` does not mint or break links, and a leaked link grants one download of one file.
* **Nothing here is reachable unless opted in.** ``gateway.api_server.file_delivery.enabled``
  defaults to False, and the two flags are independent in both directions: enabling file delivery
  does not expose browser control, and enabling browser control does not expose this route.

The upload route's Bearer + API-key ladder is untouched by all of this.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional

from gateway.browser_control_artifacts import ArtifactError, ArtifactStore
from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE, validate_media_delivery_path

if TYPE_CHECKING:  # aiohttp is the [messaging] extra; the module must import without it.
    from aiohttp import web

logger = logging.getLogger(__name__)

#: Mirrors ``DEFAULT_ARTIFACT_TTL_SECONDS``: a delivery link rides the artifact store's own TTL, so
#: the signed expiry and the stored expiry are one number rather than two that can drift.
DEFAULT_DELIVERY_TTL_SECONDS = 300.0
#: Mirrors ``DEFAULT_MAX_ARTIFACT_BYTES``.
DEFAULT_DELIVERY_MAX_BYTES = 10 * 1024 * 1024

#: Deliverable extension -> exact MIME type. Only types a chat frontend can hand back to a user as a
#: file: text/tabular documents, the two OOXML container formats, PDF and ZIP. Deliberately narrower
#: than ``MEDIA_DELIVERY_EXTS`` (which also covers video, audio, GIS and presentations) — a new type
#: is added here on purpose, not by inheriting the messaging list.
#:
#: Every key must also be in ``MEDIA_DELIVERY_EXTS`` (asserted by the test suite): that list builds
#: ``MEDIA_TAG_CLEANUP_RE``'s extension alternation, and a tag the anchored regex cannot match is
#: never rewritten — so an extension outside it would advertise a type this transport can never
#: produce. (``.markdown`` is the trap here: only ``.md`` is in the shared list, and loosening that
#: regex would change tag handling on every messaging platform.)
DELIVERY_DOCUMENT_EXT_MIME: Dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
#: Image types are NOT deliverable through this pass (the first pass owns them) but they stay in the
#: allowlist so the configured set is a superset of the artifact transport's own default and an
#: operator narrowing it never loses the images the inline path already serves.
IMAGE_EXT_MIME: Dict[str, str] = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}
#: The conservative extended default (config ``gateway.api_server.file_delivery.allowed_mime_types``).
DEFAULT_FILE_DELIVERY_MIME_TYPES: FrozenSet[str] = frozenset(
    set(DELIVERY_DOCUMENT_EXT_MIME.values()) | set(IMAGE_EXT_MIME.values()))

#: Per-response ceiling on minted artifacts (see ``DeliveryBudget``): a blast-radius stop, not a knob.
MAX_DELIVERIES_PER_RESPONSE = 12

_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{32}$")
#: Domain separator: a signature minted for anything else in this install must not verify here.
_SIGNATURE_DOMAIN = b"hermes-file-delivery-v1"
_SECRET_FILENAME = "file-delivery-signing-secret"
_SECRET_BYTES = 32
_TAG_KEYWORD = "media:"

_ZIP_PREFIXES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
#: ZIP and both OOXML formats are the same container, so one prefix test covers all three.
_ZIP_CONTAINER_TYPES = frozenset({
    "application/zip",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
})
_PREFIX_TYPES: Dict[str, tuple] = {
    "application/pdf": (b"%PDF-",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}
_TEXTUAL_TYPES = frozenset({
    "text/plain", "text/markdown", "text/csv", "text/tab-separated-values", "application/json",
})


def content_type_matches_bytes(content_type: str, data: bytes, *, sample: int = 512) -> bool:
    """Whether ``data`` may honestly be served as ``content_type``.

    The declared type comes from the file extension and extension is not evidence, so a rename could
    otherwise turn an ELF binary into ``text/markdown`` (or a log file into a PDF) and have the
    gateway hand it to the user under a trusted label. The check is deliberately one-sided: it
    rejects clear mismatches and never requires a strict structural parse, so an unusual-but-real
    document still delivers. ``application/json`` is checked as UTF-8 text, not as valid JSON — a
    JSON-lines export or a fragment with a trailing comma is a legitimate download.
    """
    if not data:
        return False
    head = bytes(data[:max(sample, 12)])
    if content_type in _PREFIX_TYPES:
        return any(head[:len(prefix)] == prefix for prefix in _PREFIX_TYPES[content_type])
    if content_type in _ZIP_CONTAINER_TYPES:
        return any(head[:len(prefix)] == prefix for prefix in _ZIP_PREFIXES)
    if content_type == "image/webp":
        return head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if content_type in _TEXTUAL_TYPES:
        if b"\x00" in head:
            return False
        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True
    return False


@dataclass(frozen=True)
class FileDeliveryConfig:
    """Resolved ``gateway.api_server.file_delivery`` block."""
    enabled: bool = False
    public_base_url: str = ""
    max_bytes: int = DEFAULT_DELIVERY_MAX_BYTES
    ttl_seconds: float = DEFAULT_DELIVERY_TTL_SECONDS
    allowed_mime_types: FrozenSet[str] = DEFAULT_FILE_DELIVERY_MIME_TYPES

    def content_type_for(self, path: Path) -> Optional[str]:
        """Document MIME type for a path's extension, or None when this pass does not own the type."""
        return DELIVERY_DOCUMENT_EXT_MIME.get(Path(path).suffix.lower())

    def download_url(self, artifact_id: str, expires_at: int, signature: str) -> str:
        """Absolute link when ``public_base_url`` is configured, else a relative path.

        An unconfigured base URL means "the frontend already talks to this API", not "refuse to
        deliver" — a relative path is still a working link for a same-origin chat UI, and the
        alternative (dropping the file) would be the bug this module exists to fix.
        """
        route = f"/v1/artifacts/download/{artifact_id}?expires={int(expires_at)}&signature={signature}"
        base = str(self.public_base_url or "").rstrip("/")
        return f"{base}{route}" if base else route

    def format_link(self, filename: str, size_bytes: int, url: str) -> str:
        """``[report.pdf (812 KB)](…)`` — the filename is the store's sanitized display name."""
        return f"[{filename} ({format_size(size_bytes)})]({url})"


def format_size(size_bytes: int) -> str:
    """Human byte count: one decimal below 10 units (``1.5 KB``), whole numbers above (``812 KB``)."""
    try:
        value = float(size_bytes)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if value < 10 and unit != "B" else f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.0f} GB"  # unreachable; keeps the type checker honest


def file_delivery_config(config: Optional[dict] = None) -> FileDeliveryConfig:
    """Resolve the delivery block (default off). A malformed block degrades to the default."""
    if config is None:
        try:
            # Hot path (every resolved response): the read-only loader skips load_config()'s deepcopy.
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            return FileDeliveryConfig()
    gateway = config.get("gateway") if isinstance(config, dict) else None
    api_server = gateway.get("api_server") if isinstance(gateway, dict) else None
    block = api_server.get("file_delivery") if isinstance(api_server, dict) else None
    if not isinstance(block, dict):
        return FileDeliveryConfig()
    base_url = block.get("public_base_url", "")
    port = FileDeliveryConfig(
        enabled=block.get("enabled", False) is True,
        public_base_url=base_url.strip() if isinstance(base_url, str) else "",
        max_bytes=_positive_int(block.get("max_bytes"), DEFAULT_DELIVERY_MAX_BYTES),
        ttl_seconds=_positive_float(block.get("ttl_seconds"), DEFAULT_DELIVERY_TTL_SECONDS),
        allowed_mime_types=_allowlist(block.get("allowed_mime_types")))
    return port


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _allowlist(value: Any) -> FrozenSet[str]:
    """Narrow the default allowlist; unknown entries are dropped with a warning.

    Narrowing-only is the point: the vocabulary is ``DELIVERY_DOCUMENT_EXT_MIME``, because a
    configured type that no extension maps to could never be produced by this transport. Adding a
    type is a code change, so the config surface cannot quietly widen what the gateway will serve.
    """
    if not isinstance(value, (list, tuple, set, frozenset)):
        return DEFAULT_FILE_DELIVERY_MIME_TYPES
    requested = {str(item).strip().lower() for item in value if isinstance(item, str) and item.strip()}
    if not requested:
        return DEFAULT_FILE_DELIVERY_MIME_TYPES
    unknown = requested - DEFAULT_FILE_DELIVERY_MIME_TYPES
    if unknown:
        logger.warning(
            "gateway.api_server.file_delivery.allowed_mime_types: ignoring %s (not produced by "
            "this transport; see DELIVERY_DOCUMENT_EXT_MIME)", ", ".join(sorted(unknown)))
    allowed = frozenset(requested & DEFAULT_FILE_DELIVERY_MIME_TYPES)
    return allowed or DEFAULT_FILE_DELIVERY_MIME_TYPES


def signing_secret(home: Path) -> Optional[bytes]:
    """Per-install signing secret, created on first use at ``<home>/file-delivery-signing-secret``.

    Stored outside the artifact root so the store's orphan sweep never sees it, written 0600 via
    ``O_EXCL`` so two processes racing on first use cannot end up with different secrets (the loser
    reads the winner's). Returns None when the home is not writable — delivery is then skipped and
    the tag stays literal, which is the fail-closed direction.
    """
    path = Path(home) / _SECRET_FILENAME
    try:
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        generator = secrets.token_bytes(_SECRET_BYTES)
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(handle, generator.hex().encode("ascii"))
        finally:
            os.close(handle)
        return generator
    except FileExistsError:
        try:
            return bytes.fromhex(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
    except OSError:
        logger.warning("file delivery: could not create a signing secret at %s", path, exc_info=True)
        return None


def sign_download_token(secret: bytes, artifact_id: str, expires_at: int) -> str:
    """HMAC-SHA256 over (domain, artifact id, expiry), truncated to 128 bits for a usable URL."""
    message = b"\x00".join((_SIGNATURE_DOMAIN, artifact_id.encode("ascii"), str(int(expires_at)).encode("ascii")))
    return hmac.new(secret, message, "sha256").hexdigest()[:32]


def verify_download_token(secret: Optional[bytes], artifact_id: str, expires_at: Any, signature: Any, *,
                          now: Optional[float] = None) -> bool:
    """Constant-time signature + expiry check. Every rejection reason is a plain False.

    No reason is returned on purpose: the caller answers every failure with the same 404, so the
    route cannot be used to probe which artifact ids exist or which links were already consumed.
    """
    if not secret or not isinstance(signature, str) or not _SIGNATURE_RE.match(signature):
        return False
    if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.match(artifact_id):
        return False
    try:
        expires = int(expires_at)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else float(now)
    if expires <= current:
        return False
    return hmac.compare_digest(signature, sign_download_token(secret, artifact_id, expires))


class DeliveryBudget:
    """Per-response delivery memo + count cap, shared by the non-streaming and streaming lanes.

    Two bounds live here, both per response rather than per install:

    * **One artifact per path.** A reply that names the same file twice (common: once in prose, once
      as the tag) mints one artifact and repeats one link, instead of burning a store slot and a
      second expiring URL per mention.
    * **A ceiling on how many.** ``max_bytes`` bounds one file but nothing bounds their *number*, and
      a response is model-authored text: without a cap a single reply could copy an arbitrary number
      of large files into the artifact root. Files over the ceiling stay literal.

    Not configurable on purpose — it is a blast-radius stop, not a preference, and the ceiling is far
    above a reply a person would want to read.
    """

    __slots__ = ("_store", "_scope", "_config", "_secret", "_session_key", "_limit", "_resolved", "_delivered")

    def __init__(self, *, store: ArtifactStore, scope: Any, config: FileDeliveryConfig,
                 secret: Optional[bytes], session_key: str = "",
                 max_deliveries: int = MAX_DELIVERIES_PER_RESPONSE) -> None:
        self._store, self._scope, self._config = store, scope, config
        self._secret, self._session_key = secret, session_key
        self._limit = max(1, int(max_deliveries))
        self._resolved: Dict[str, Optional[str]] = {}
        self._delivered = 0

    def resolve(self, raw_path: str) -> Optional[str]:
        """Link for one tag's path, or None to leave that tag literal."""
        if raw_path not in self._resolved:
            if self._delivered >= self._limit:
                logger.info("file delivery: %d-file ceiling reached for this response; "
                            "leaving further tags literal", self._limit)
                self._resolved[raw_path] = None
            else:
                self._resolved[raw_path] = store_and_link(
                    raw_path, store=self._store, scope=self._scope, config=self._config,
                    secret=self._secret, session_key=self._session_key)
                if self._resolved[raw_path]:
                    self._delivered += 1
        return self._resolved[raw_path]


def store_and_link(raw_path: str, *, store: ArtifactStore, scope: Any, config: FileDeliveryConfig,
                   secret: Optional[bytes], session_key: str = "") -> Optional[str]:
    """Deliver one ``MEDIA:`` path as a signed download link; None means "leave the tag literal".

    Failures are silent by design: a path rejected here must degrade exactly as it did before this
    transport existed (tag left visible), never to a broken link and never to a leaked host path.
    The order matters — the shared validator is the security choke point and runs first, so a
    traversal path, a credential file or a denylisted root is never even stat'ed.
    """
    if not config.enabled or secret is None:
        return None
    safe_path = validate_media_delivery_path(raw_path, session_key=session_key)
    if not safe_path:
        return None
    path = Path(safe_path)
    content_type = config.content_type_for(path)
    if content_type is None or content_type not in config.allowed_mime_types:
        return None
    try:
        size = path.stat().st_size
        if size <= 0 or size > config.max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if not content_type_matches_bytes(content_type, data):
        logger.info("file delivery: %s is not %s; leaving the tag literal", path.name, content_type)
        return None
    try:
        receipt = store.store(data, filename=path.name, content_type=content_type, scope=scope)
    except ArtifactError as exc:
        logger.debug("file delivery: store rejected %s: %s", path.name, exc)
        return None
    expires_at = int(receipt.expires_at)
    signature = sign_download_token(secret, receipt.artifact_id, expires_at)
    return config.format_link(receipt.filename, receipt.size_bytes,
                              config.download_url(receipt.artifact_id, expires_at, signature))


def resolve_media_to_download_urls(text: str, *, store: ArtifactStore, scope: Any,
                                   config: FileDeliveryConfig, secret: Optional[bytes],
                                   session_key: str = "") -> str:
    """Rewrite the ``MEDIA:`` tags the image pass left behind into signed download links.

    A fresh :class:`DeliveryBudget` is the per-response scope: one artifact per repeated path, up to
    the per-response ceiling.
    """
    if not config.enabled or not text or "MEDIA:" not in text:
        return text
    budget = DeliveryBudget(store=store, scope=scope, config=config, secret=secret, session_key=session_key)

    def _repl(match: "re.Match[str]") -> str:
        return budget.resolve(match.group("path")) or match.group(0)

    try:
        # A leaked terminal <|eos|> glued to the last tag is not a path terminator (#111046): scan
        # without it and drop the control token only when a tag actually resolved — the same
        # treatment the image pass applies, so the two passes agree on where the text ends.
        sentinel_start = _terminal_sentinel_start(text)
        scan = text[:sentinel_start] if sentinel_start >= 0 else text
        resolved = MEDIA_TAG_CLEANUP_RE.sub(_repl, scan)
        return text if resolved == scan else resolved
    except Exception:
        logger.debug("file delivery: rewrite failed; leaving MEDIA: tags literal", exc_info=True)
        return text


def _terminal_sentinel_start(text: str) -> int:
    """Late import: this module is imported by ``api_server``, which owns the sentinel scanner."""
    try:
        from gateway.platforms.api_server import _terminal_sentinel_start as scanner
        return scanner(text)
    except Exception:
        return -1


class MediaTagStreamRewriter:
    """Resolve ``MEDIA:`` tags that straddle streaming delta boundaries.

    A tag can be split anywhere (``"…MEDIA:/data/re"`` + ``"port.pdf"``), so a per-delta regex would
    either leak the raw server path or drop the file. This holds back only the tail that could still
    become a tag — text whose tags are already terminated streams immediately, so the visible reply
    stays live instead of stalling until the turn ends — and resolves the held portion the moment a
    real terminator arrives. A tag whose only terminator is the end of the stream is resolved by
    ``flush()``.

    A match closed by end-of-buffer is held, not resolved: ``…/archive.tar`` matches at a buffer end
    and grows to ``…/archive.tar.gz`` on the next delta (``MEDIA_TAG_CLEANUP_RE`` accepts either).
    """

    __slots__ = ("_resolve", "_held")

    def __init__(self, resolve: Callable[[str], Optional[str]]) -> None:
        self._resolve = resolve
        self._held = ""

    def feed(self, delta: str) -> str:
        """Text safe to emit now; the rest is held until it resolves or the turn ends."""
        if not delta:
            return ""
        buffer = self._held + delta
        emitted: List[str] = []
        position = 0
        for match in MEDIA_TAG_CLEANUP_RE.finditer(buffer):
            if match.end() >= len(buffer):
                break
            emitted.append(buffer[position:match.start()])
            emitted.append(self._resolve(match.group("path")) or match.group(0))
            position = match.end()
        rest = buffer[position:]
        hold_at = _pending_tag_start(rest)
        emitted.append(rest[:hold_at])
        self._held = rest[hold_at:]
        return "".join(emitted)

    def flush(self) -> str:
        """Resolve everything still held, including a tag closed only by end-of-stream."""
        held, self._held = self._held, ""
        return _substitute_tags(held, self._resolve)


def _substitute_tags(text: str, resolve: Callable[[str], Optional[str]]) -> str:
    """Resolve every complete tag in a final (non-streaming) fragment."""
    if not text or "MEDIA:" not in text:
        return text

    def _repl(match: "re.Match[str]") -> str:
        return resolve(match.group("path")) or match.group(0)

    try:
        return MEDIA_TAG_CLEANUP_RE.sub(_repl, text)
    except Exception:
        return text


def _pending_tag_start(text: str) -> int:
    """Index where a possible-but-incomplete tag begins, else ``len(text)``.

    Deliberately broader than the tag regex: any ``media:`` occurrence (or a trailing fragment of the
    keyword) starts a hold. Holding a few characters too long costs a little latency; emitting them
    costs a leaked server path, so the ambiguity resolves in one direction only.
    """
    lowered = text.lower()
    index = lowered.rfind(_TAG_KEYWORD)
    if index >= 0:
        # Up to three markdown emphasis/quote markers may precede the keyword; keep them with the
        # hold when they arrived in this same buffer, so the client never renders a stray backtick.
        back = index
        while back > 0 and index - back < 3 and text[back - 1] in "`\"'*_":
            back -= 1
        return back
    for length in range(len(_TAG_KEYWORD) - 1, 0, -1):
        # Down to a single character: "MEDIA:" can be split as "M" + "EDIA:/path", and emitting that
        # lone "M" breaks the keyword for every later delta — the remainder then reads as ordinary
        # text and the raw path streams out (found by verify-file-delivery.py on 7-character chunks).
        if lowered.endswith(_TAG_KEYWORD[:length]):
            return len(text) - length
    return len(text)


#: Transport family bound into the artifact scope key. Part of the identity that is both minted and
#: verified: a signed delivery link can only ever reach an artifact minted for this family.
_DELIVERY_FAMILY = "file-delivery"


def delivery_root(profile: str) -> Path:
    """``<profile dir>/artifacts/file-delivery`` — the controlled root delivery bytes never leave.

    Same ladder as the browser-control store: the profile directory when it resolves, else the Hermes
    home. Placement under the profile's own directory is what makes "two profiles never share a
    store root" a property of the path rather than of the cache.
    """
    try:
        from hermes_cli.profiles import get_profile_dir
        return Path(get_profile_dir(profile or "default")) / "artifacts" / "file-delivery"
    except Exception:
        from hermes_state import get_hermes_home
        return Path(get_hermes_home()) / "artifacts" / "file-delivery"


class FileDeliveryMixin:
    """Adapter-bound half of the delivery transport (mixed into ``APIServerAdapter``).

    The pure half lives above — config, signing, sniffing, rewriting. This half owns what is
    per-adapter or per-request: the profile-scoped artifact store, the per-install signing secret,
    and the two entry points the request handlers call.
    """

    def _file_delivery_config(self) -> FileDeliveryConfig:
        """Read the config block per call, so ``hermes config set`` needs no gateway restart."""
        return file_delivery_config()

    def _file_delivery_principal(self, profile: str) -> str:
        """Server-derived principal for delivery artifacts.

        Derived from (profile, expected API key) exactly like the browser-control principal, under a
        distinct domain prefix so the two namespaces never collide — a signed link therefore cannot
        reach a browser-control artifact, and vice versa, even in one profile.
        """
        key = self._expected_api_key() or self._api_key or ""
        digest = hashlib.sha256(f"file-delivery\x00{profile}\x00{key}".encode("utf-8")).hexdigest()
        return f"delivery:{profile}:{digest[:32]}"

    def _file_delivery_scope(self, profile: str):
        """Artifact scope for ``profile``; the same object shape the store's scope key expects."""
        from gateway.platforms.api_server import _ArtifactScopeFacade
        return _ArtifactScopeFacade(self._file_delivery_principal(profile), transport_family=_DELIVERY_FAMILY)

    def _file_delivery_store_for(self, profile: str, config: FileDeliveryConfig) -> ArtifactStore:
        """Delivery artifact store for ``profile``, cached by (profile, caps fingerprint).

        A root of its own, separate from browser control's: the two transports have different gates,
        different MIME vocabularies and different consumers, so sharing a store would mean a change to
        one silently changes what the other will serve.

        The cache is keyed by the resolved caps as well as the profile because the store owns the only
        index of what it has minted (receipts are in memory). Pinning the first config seen would
        silently ignore an operator's later edit; rebuilding drops that index, so links minted under
        the previous caps 404 rather than serving under stale rules — the fail-closed direction.
        """
        profile_key = str(profile or "default")
        fingerprint = (config.ttl_seconds, config.max_bytes, tuple(sorted(config.allowed_mime_types)))
        cache = getattr(self, "_file_delivery_artifacts", None)
        if cache is None:
            cache = {}
            self._file_delivery_artifacts = cache
        entry = cache.get(profile_key)
        if entry is not None and entry[0] == fingerprint:
            return entry[1]
        store = ArtifactStore(delivery_root(profile_key), ttl_seconds=config.ttl_seconds,
                              max_bytes=config.max_bytes, allowed_mime_types=config.allowed_mime_types)
        store.prune_expired()
        cache[profile_key] = (fingerprint, store)
        return store

    def _file_delivery_secret(self, profile: str) -> Optional[bytes]:
        """Per-install, per-profile-home signing secret (created on first use)."""
        return signing_secret(delivery_root(profile).parent)

    def _inject_file_delivery_store(self, store: Optional[ArtifactStore], *, profile: str = "default",
                                    ttl_seconds: float = DEFAULT_DELIVERY_TTL_SECONDS,
                                    max_bytes: int = DEFAULT_DELIVERY_MAX_BYTES) -> None:
        """Inject a store (tests, diagnostics); ``None`` drops the cached one for ``profile``."""
        cache = getattr(self, "_file_delivery_artifacts", None)
        if cache is None:
            cache = {}
            self._file_delivery_artifacts = cache
        if store is None:
            cache.pop(profile, None)
        else:
            cache[profile] = ((ttl_seconds, max_bytes, tuple(sorted(store.allowed_mime_types))), store)

    # -- resolution entry points -------------------------------------------------------

    def _delivery_context(self, session_key: str = ""):
        """(store, scope, secret, config) for the request's profile, or None when not deliverable."""
        config = self._file_delivery_config()
        if not config.enabled:
            return None
        from gateway.platforms.api_server import _api_request_profile
        profile = _api_request_profile.get() or "default"
        secret = self._file_delivery_secret(profile)
        if secret is None:
            return None
        return self._file_delivery_store_for(profile, config), self._file_delivery_scope(profile), secret, config

    def _resolve_media_tags(self, text: str, *, session_key: str = "") -> str:
        """Both resolution passes for one final response.

        Images first, unchanged (``_resolve_media_to_data_urls``), then this module's pass over what
        is left. Everything that is not a deliverable file — an unreadable path, a rejected path, an
        oversize file, an extension this transport does not own, or the feature being off — leaves the
        tag exactly as that first pass left it, so the disabled path is byte-identical to the behavior
        that shipped before this transport existed.
        """
        from gateway.platforms.api_server import _resolve_media_to_data_urls
        resolved = _resolve_media_to_data_urls(text)
        if not resolved or "MEDIA:" not in resolved:
            return resolved
        try:
            context = self._delivery_context(session_key=session_key)
            if context is None:
                return resolved
            store, scope, secret, config = context
            return resolve_media_to_download_urls(
                resolved, store=store, scope=scope, config=config, secret=secret, session_key=session_key)
        except Exception:
            logger.debug("file delivery: resolution failed; leaving MEDIA: tags literal", exc_info=True)
            return resolved

    def _file_delivery_stream_rewriter(self, *, session_key: str = "") -> Optional[MediaTagStreamRewriter]:
        """Per-turn streaming rewriter, or None when delivery is off (callers then emit raw deltas).

        One :class:`DeliveryBudget` backs the whole turn, so the streaming lane shares the
        non-streaming lane's two bounds: one artifact per repeated path, and the per-response ceiling.
        """
        try:
            context = self._delivery_context(session_key=session_key)
        except Exception:
            logger.debug("file delivery: stream rewriter unavailable", exc_info=True)
            return None
        if context is None:
            return None
        store, scope, secret, config = context
        budget = DeliveryBudget(store=store, scope=scope, config=config, secret=secret,
                                session_key=session_key)
        return MediaTagStreamRewriter(budget.resolve)

    # -- signed download route --------------------------------------------------------

    def _file_delivery_signed_request(self, request: "web.Request") -> bool:
        """True when a delivery link (not a browser-control client) is addressing the route.

        The route is shared, so the discriminator has to be explicit: browser-control clients send a
        Bearer key and no ``signature``, delivery links send ``signature``/``expires`` and no key.
        """
        try:
            return self._file_delivery_config().enabled and bool(request.query.get("signature"))
        except Exception:
            return False

    async def _handle_file_delivery_download(self, request: "web.Request") -> "web.Response":
        """GET the signed one-shot delivery link: ``?expires=<unix>&signature=<hex>``.

        Every failure — unknown artifact, expired, already consumed, tampered signature, wrong
        profile, unwritable store — answers 404 with no body detail. That is deliberate: a 403 would
        confirm the artifact exists, turning the route into an existence oracle for a frontend that
        holds no key. The one non-404 is 429 from the per-profile limiter, which reports the caller's
        own request rate and leaks nothing about any artifact.
        """
        from aiohttp import web
        config = self._file_delivery_config()
        if not config.enabled:
            raise web.HTTPNotFound()
        from gateway.platforms.api_server import _api_request_profile, _error_response
        profile = _api_request_profile.get() or "default"
        if not self._artifact_limiter().allow(f"delivery-download:{profile}"):
            return _error_response("Delivery download rate limit exceeded.", 429,
                                   err_type="rate_limit_error", code="rate_limit_exceeded",
                                   headers={"Retry-After": "1"})
        artifact_id = request.match_info.get("artifact_id", "")
        try:
            store = self._file_delivery_store_for(profile, config)
            secret = self._file_delivery_secret(profile)
        except Exception:
            raise web.HTTPNotFound()
        if not verify_download_token(secret, artifact_id, request.query.get("expires"),
                                     request.query.get("signature")):
            raise web.HTTPNotFound()
        try:
            data, receipt = store.load(artifact_id, scope=self._file_delivery_scope(profile))
        except ArtifactError:
            raise web.HTTPNotFound()
        return web.Response(
            body=data, status=200, content_type=receipt.content_type,
            headers={"Content-Disposition": f'attachment; filename="{receipt.filename}"',
                     "X-Artifact-Sha256": receipt.sha256, "X-Artifact-Id": receipt.artifact_id,
                     "Cache-Control": "no-store"})
