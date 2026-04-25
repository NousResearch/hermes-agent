"""Disabled-by-default Agent-to-Agent (A2A) safety primitives.

This module does not start servers, register platform adapters, or expose network
routes. It provides the conservative core pieces needed before a live A2A adapter
can exist: peer policy loading, separate transcript storage, redaction, and
Bearer/HMAC/timestamp/nonce verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from hermes_constants import get_hermes_home


_SECRET_RE = re.compile(
    r"("  # common token shapes
    r"sk-[A-Za-z0-9_-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9_]{8,}"
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r")"
)
_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|api[_-]?key|token|secret|cookie)\s*[:=]\s*[^\s;,]+"
)
_COOKIE_RE = re.compile(r"(?i)\bCookie\s*:\s*[^\n\r]+")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_CREDENTIAL_PATH_RE = re.compile(r"(?:~|/Users/[^\s]+)?/\.credentials/[^\s)\]}>\"']+")
_MANDATORY_BLOCKED_ROOTS = tuple(DEFAULT_A2A_POLICY.blocked_roots) if 'DEFAULT_A2A_POLICY' in globals() else (
    "~/.credentials",
    "~/.ssh",
    "~/.cloudflared",
    "~/Library/Messages",
    "~/Library/Mail",
    "/etc",
    "/System",
    "/usr/bin",
    "/var/db",
)
_ALLOWED_DANGEROUS_COMMAND_MODES = {"deny", "ask", "scoped_allow"}
_MAX_NONCE_LEN = 128


@dataclass(frozen=True)
class A2APolicy:
    """Resolved policy for one A2A peer."""

    peer_id: str = ""
    display_name: str = ""
    owner: str = ""
    enabled: bool = True
    auth: dict[str, Any] = field(default_factory=dict)
    policy_profile: str = "a2a-readonly-default"
    allowed_ingress_routes: list[str] = field(default_factory=list)
    allowed_outbound_routes: list[str] = field(default_factory=list)
    read_roots: list[str] = field(default_factory=list)
    write_roots: list[str] = field(default_factory=list)
    blocked_roots: list[str] = field(default_factory=lambda: [
        "~/.credentials",
        "~/.ssh",
        "~/.cloudflared",
        "~/Library/Messages",
        "~/Library/Mail",
        "/etc",
        "/System",
        "/usr/bin",
        "/var/db",
    ])
    allowed_toolsets: list[str] = field(default_factory=list)
    external_send_allowed: bool = False
    dangerous_command_mode: str = "deny"
    credential_policy: str = "no_plaintext_file_path_only_when_explicitly_authorized"
    max_tokens_per_day: int = 20_000
    max_tool_calls_per_message: int = 5
    transcript_retention_days: int = 30


DEFAULT_A2A_POLICY = A2APolicy(enabled=True)


@dataclass(frozen=True)
class A2AConfig:
    """Top-level A2A config resolved from Hermes config.yaml."""

    enabled: bool
    peers: dict[str, dict[str, Any]]
    transcript_dir: Path


@dataclass(frozen=True)
class TranscriptAppendResult:
    path: Path
    redaction_count: int
    message: dict[str, Any]


@dataclass(frozen=True)
class A2AVerificationResult:
    ok: bool
    timestamp: int
    nonce: str


class A2AVerificationError(ValueError):
    """Raised when an A2A authentication check fails."""


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _strict_bool(value: Any, *, field_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def load_a2a_config(config_path: Path | None = None, hermes_home: Path | None = None) -> A2AConfig:
    """Load disabled-by-default A2A config from ``config.yaml``.

    Missing config means A2A is off. This function only reads local config and
    does not start any network listener.
    """

    home = hermes_home or get_hermes_home()
    path = config_path or home / "config.yaml"
    root = _load_yaml_mapping(path)
    raw = root.get("a2a") if isinstance(root, dict) else None
    cfg = raw if isinstance(raw, dict) else {}
    peers = cfg.get("peers") if isinstance(cfg.get("peers"), dict) else {}
    transcript_dir_raw = cfg.get("transcript_dir")
    if transcript_dir_raw:
        transcript_dir = Path(transcript_dir_raw).expanduser()
        if not transcript_dir.is_absolute():
            transcript_dir = home / transcript_dir
    else:
        transcript_dir = home / "a2a" / "transcripts"
    return A2AConfig(
        enabled=_strict_bool(cfg.get("enabled"), field_name="a2a.enabled", default=False),
        peers=peers,
        transcript_dir=transcript_dir,
    )


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("policy list fields must be lists")
    return [str(item) for item in value]


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _blocked_roots_with_mandatory(value: Any) -> list[str]:
    configured = _list_value(value)
    merged = list(configured)
    for root in _MANDATORY_BLOCKED_ROOTS:
        if root not in merged:
            merged.append(root)
    return merged


class A2APeerPolicyStore:
    """Load and resolve explicit peer policies from Hermes config."""

    def __init__(self, config_path: Path | None = None, hermes_home: Path | None = None):
        self.hermes_home = hermes_home or get_hermes_home()
        self.config_path = config_path or self.hermes_home / "config.yaml"

    def list_peers(self, *, validate: bool = False) -> list[A2APolicy | str]:
        """List configured peers, optionally resolving each policy eagerly."""

        config = load_a2a_config(config_path=self.config_path, hermes_home=self.hermes_home)
        peer_ids = sorted(str(peer_id) for peer_id in config.peers)
        if not validate:
            return peer_ids
        return [self.get_peer(peer_id) for peer_id in peer_ids]

    def get_peer(self, peer_id: str) -> A2APolicy:
        config = load_a2a_config(config_path=self.config_path, hermes_home=self.hermes_home)
        raw = config.peers.get(peer_id)
        if not isinstance(raw, dict):
            raise KeyError(peer_id)
        merged = {**DEFAULT_A2A_POLICY.__dict__, **raw, "peer_id": peer_id}
        auth = merged.get("auth") if isinstance(merged.get("auth"), dict) else {}
        if not auth.get("bearer_token_env") or not auth.get("hmac_secret_env"):
            raise ValueError(f"A2A peer {peer_id!r} must configure auth.bearer_token_env and auth.hmac_secret_env")
        dangerous_command_mode = str(merged.get("dangerous_command_mode") or "deny")
        if dangerous_command_mode not in _ALLOWED_DANGEROUS_COMMAND_MODES:
            raise ValueError("dangerous_command_mode must be one of deny, ask, scoped_allow")
        return A2APolicy(
            peer_id=peer_id,
            display_name=str(merged.get("display_name") or peer_id),
            owner=str(merged.get("owner") or ""),
            enabled=_strict_bool(merged.get("enabled"), field_name=f"a2a.peers.{peer_id}.enabled", default=True) and config.enabled,
            auth=dict(auth),
            policy_profile=str(merged.get("policy_profile") or DEFAULT_A2A_POLICY.policy_profile),
            allowed_ingress_routes=_list_value(merged.get("allowed_ingress_routes")),
            allowed_outbound_routes=_list_value(merged.get("allowed_outbound_routes")),
            read_roots=_list_value(merged.get("read_roots")),
            write_roots=_list_value(merged.get("write_roots")),
            blocked_roots=_blocked_roots_with_mandatory(merged.get("blocked_roots")),
            allowed_toolsets=_list_value(merged.get("allowed_toolsets")),
            external_send_allowed=_strict_bool(
                merged.get("external_send_allowed"),
                field_name=f"a2a.peers.{peer_id}.external_send_allowed",
                default=False,
            ),
            dangerous_command_mode=dangerous_command_mode,
            credential_policy=str(merged.get("credential_policy") or DEFAULT_A2A_POLICY.credential_policy),
            max_tokens_per_day=_positive_int(merged.get("max_tokens_per_day"), field_name="max_tokens_per_day"),
            max_tool_calls_per_message=_positive_int(merged.get("max_tool_calls_per_message"), field_name="max_tool_calls_per_message"),
            transcript_retention_days=_positive_int(merged.get("transcript_retention_days"), field_name="transcript_retention_days"),
        )


def redact_a2a_text(text: str) -> tuple[str, int]:
    """Redact credential-like values and credential paths from A2A transcripts."""

    count = 0

    def _secret(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "<redacted>"

    def _key_value(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}=<redacted>"

    def _cookie(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "Cookie: <redacted>"

    def _private_key(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "<private-key-redacted>"

    def _cred_path(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "<credential-path-redacted>"

    redacted = _PRIVATE_KEY_RE.sub(_private_key, text)
    redacted = _COOKIE_RE.sub(_cookie, redacted)
    redacted = _KEY_VALUE_SECRET_RE.sub(_key_value, redacted)
    redacted = _SECRET_RE.sub(_secret, redacted)
    redacted = _CREDENTIAL_PATH_RE.sub(_cred_path, redacted)
    return redacted, count


def redact_a2a_value(value: Any) -> tuple[Any, int]:
    """Recursively redact strings inside JSON-serializable metadata."""

    if isinstance(value, str):
        return redact_a2a_text(value)
    if isinstance(value, list):
        redacted_items = []
        total = 0
        for item in value:
            redacted, count = redact_a2a_value(item)
            redacted_items.append(redacted)
            total += count
        return redacted_items, total
    if isinstance(value, dict):
        redacted_dict: dict[str, Any] = {}
        total = 0
        for key, item in value.items():
            redacted_key, key_count = redact_a2a_text(str(key))
            redacted, value_count = redact_a2a_value(item)
            redacted_dict[redacted_key] = redacted
            total += key_count + value_count
        return redacted_dict, total
    return value, 0


class A2ATranscriptStore:
    """Separate JSONL transcript storage for peer-originated conversations."""

    def __init__(self, root: Path | None = None):
        self.root = root or get_hermes_home() / "a2a" / "transcripts"

    @staticmethod
    def _safe_segment(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        if not safe:
            raise ValueError("empty transcript path segment")
        return safe

    def _conversation_path(self, peer_id: str, conversation_id: str) -> Path:
        return self.root / self._safe_segment(peer_id) / f"{self._safe_segment(conversation_id)}.jsonl"

    def append_message(
        self,
        *,
        peer_id: str,
        conversation_id: str,
        direction: str,
        body: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TranscriptAppendResult:
        if direction not in {"inbound", "outbound"}:
            raise ValueError("direction must be inbound or outbound")
        redacted_body, body_redaction_count = redact_a2a_text(body)
        redacted_metadata, metadata_redaction_count = redact_a2a_value(dict(metadata or {}))
        redaction_count = body_redaction_count + metadata_redaction_count
        message = {
            "timestamp": time.time(),
            "peer_id": peer_id,
            "conversation_id": conversation_id,
            "direction": direction,
            "body": redacted_body,
            "redaction_count": redaction_count,
            "metadata": redacted_metadata,
        }
        path = self._conversation_path(peer_id, conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n")
        return TranscriptAppendResult(path=path, redaction_count=redaction_count, message=message)

    def get_conversation(self, peer_id: str, conversation_id: str) -> list[dict[str, Any]]:
        path = self._conversation_path(peer_id, conversation_id)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


class A2AAuthVerifier:
    """Verify Bearer token + HMAC signature + timestamp + nonce replay guard."""

    def __init__(
        self,
        *,
        nonce_store: set[str] | None = None,
        now: Callable[[], int | float] | None = None,
        max_skew_seconds: int = 300,
    ):
        self.nonce_store = nonce_store if nonce_store is not None else set()
        self.now = now or time.time
        self.max_skew_seconds = max_skew_seconds

    @staticmethod
    def _signature_payload(body: bytes, *, timestamp: int, nonce: str) -> bytes:
        return str(timestamp).encode() + b"." + nonce.encode() + b"." + body

    def sign_body(self, body: bytes, secret: str, *, timestamp: int, nonce: str) -> str:
        digest = hmac.new(
            secret.encode("utf-8"),
            self._signature_payload(body, timestamp=timestamp, nonce=nonce),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={digest}"

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str:
        lowered = {k.lower(): v for k, v in headers.items()}
        return lowered.get(name.lower(), "")

    def verify(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
        bearer_token_env: str,
        hmac_secret_env: str,
    ) -> A2AVerificationResult:
        expected_token = os.environ.get(bearer_token_env, "")
        secret = os.environ.get(hmac_secret_env, "")
        if not expected_token or not secret:
            raise A2AVerificationError("missing configured A2A auth secret")

        auth = self._header(headers, "Authorization")
        if not auth.startswith("Bearer "):
            raise A2AVerificationError("missing bearer token")
        token = auth.removeprefix("Bearer ")
        if not hmac.compare_digest(token, expected_token):
            raise A2AVerificationError("invalid bearer token")

        try:
            timestamp = int(self._header(headers, "X-Hermes-A2A-Timestamp"))
        except ValueError as exc:
            raise A2AVerificationError("invalid timestamp") from exc
        if abs(int(self.now()) - timestamp) > self.max_skew_seconds:
            raise A2AVerificationError("timestamp outside allowed skew")

        nonce = self._header(headers, "X-Hermes-A2A-Nonce")
        if not nonce:
            raise A2AVerificationError("missing nonce")
        if len(nonce) > _MAX_NONCE_LEN:
            raise A2AVerificationError("nonce exceeds maximum length")
        if not re.fullmatch(r"[A-Za-z0-9_:-]+", nonce):
            raise A2AVerificationError("nonce contains invalid characters")
        if nonce in self.nonce_store:
            raise A2AVerificationError("replay nonce rejected")

        supplied_signature = self._header(headers, "X-Hermes-A2A-Signature")
        expected_signature = self.sign_body(body, secret, timestamp=timestamp, nonce=nonce)
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise A2AVerificationError("invalid signature")

        self.nonce_store.add(nonce)
        return A2AVerificationResult(ok=True, timestamp=timestamp, nonce=nonce)
