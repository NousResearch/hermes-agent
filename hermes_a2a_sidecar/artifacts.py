from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from google.protobuf import json_format

from a2a.types import Message

from .config import PeerPolicy

_SCRIPT_EXTENSIONS = {
    ".bash",
    ".bat",
    ".cmd",
    ".env",
    ".fish",
    ".js",
    ".mjs",
    ".ps1",
    ".py",
    ".rb",
    ".sh",
    ".ts",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
}
_DANGEROUS_PATTERNS = [
    re.compile(r"rm\s+-rf\s+[/~$]"),
    re.compile(r"curl\b.*\|\s*(?:sh|bash)", re.I | re.S),
    re.compile(r"wget\b.*\|\s*(?:sh|bash)", re.I | re.S),
    re.compile(r"base64\s+-d\b.*\|\s*(?:sh|bash|python)", re.I | re.S),
    re.compile(r"(?:api[_-]?key|secret|token)\s*=", re.I),
]


@dataclass(slots=True)
class RemoteArtifact:
    url: str
    filename: str = ""
    media_type: str = ""
    sha256: str = ""
    issuer: str = ""
    declared_intent: str = ""
    required_permissions: list[str] = field(default_factory=list)
    source: str = "message.parts"

    @property
    def hostname(self) -> str:
        return (urlparse(self.url).hostname or "").lower()

    @property
    def extension(self) -> str:
        name = self.filename or Path(urlparse(self.url).path).name
        return Path(name).suffix.lower()

    @property
    def needs_review(self) -> bool:
        return bool(self.required_permissions) or self.extension in _SCRIPT_EXTENSIONS


@dataclass(slots=True)
class StagedArtifact:
    artifact: RemoteArtifact
    status: str
    sha256: str = ""
    local_path: str = ""
    bytes_downloaded: int = 0
    findings: list[str] = field(default_factory=list)
    error: str = ""


def _metadata_dict(obj: Any) -> dict[str, Any]:
    if not obj:
        return {}
    try:
        return json_format.MessageToDict(obj)
    except Exception:
        return {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, list):
        return [str(p).strip() for p in value if str(p).strip()]
    return [str(value).strip()] if str(value).strip() else []


def extract_remote_artifacts(message: Message | None) -> list[RemoteArtifact]:
    if message is None:
        return []
    artifacts: list[RemoteArtifact] = []
    for part in message.parts:
        if not part.url:
            continue
        meta = _metadata_dict(part.metadata)
        artifacts.append(
            RemoteArtifact(
                url=part.url,
                filename=part.filename or str(meta.get("filename") or ""),
                media_type=part.media_type or str(meta.get("media_type") or ""),
                sha256=str(meta.get("sha256") or meta.get("checksum_sha256") or ""),
                issuer=str(meta.get("issuer") or meta.get("signer") or ""),
                declared_intent=str(meta.get("declared_intent") or meta.get("intent") or ""),
                required_permissions=_as_list(meta.get("required_permissions")),
            )
        )
    msg_meta = _metadata_dict(message.metadata)
    for item in msg_meta.get("artifacts") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        artifacts.append(
            RemoteArtifact(
                url=str(item.get("url")),
                filename=str(item.get("filename") or ""),
                media_type=str(item.get("media_type") or item.get("content_type") or ""),
                sha256=str(item.get("sha256") or item.get("checksum_sha256") or ""),
                issuer=str(item.get("issuer") or item.get("signer") or ""),
                declared_intent=str(item.get("declared_intent") or item.get("intent") or ""),
                required_permissions=_as_list(item.get("required_permissions")),
                source="message.metadata.artifacts",
            )
        )
    return artifacts


def validate_artifact_sources(
    artifacts: list[RemoteArtifact], policy: PeerPolicy
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    allowed = {d.lower() for d in policy.allowed_artifact_domains}
    for artifact in artifacts:
        parsed = urlparse(artifact.url)
        if parsed.scheme not in {"http", "https"}:
            problems.append(f"unsupported artifact URL scheme for {artifact.url!r}")
        if allowed and artifact.hostname not in allowed:
            problems.append(
                f"artifact host {artifact.hostname!r} is not in allowlist for peer {policy.id}"
            )
    return not problems, problems


def _safe_filename(artifact: RemoteArtifact, index: int) -> str:
    raw = artifact.filename or Path(urlparse(artifact.url).path).name or f"artifact-{index}"
    raw = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("._")
    return raw or f"artifact-{index}"


def _scan_bytes(data: bytes) -> list[str]:
    findings: list[str] = []
    sample = data[:1_000_000]
    try:
        text = sample.decode("utf-8", errors="ignore")
    except Exception:
        return findings
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            findings.append(f"matched pattern {pattern.pattern}")
    if text.startswith("#!"):
        findings.append("has shebang; treat as executable script")
    return findings


def stage_artifacts(
    artifacts: list[RemoteArtifact],
    *,
    policy: PeerPolicy,
    root: Path,
    correlation_id: str,
) -> list[StagedArtifact]:
    """Download/checksum/scan remote artifact handles without executing them."""

    staged: list[StagedArtifact] = []
    ok, problems = validate_artifact_sources(artifacts, policy)
    if not ok:
        for artifact in artifacts:
            staged.append(StagedArtifact(artifact=artifact, status="rejected", findings=problems))
        return staged
    if not artifacts:
        return staged
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / re.sub(r"[^A-Za-z0-9._-]", "_", correlation_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, artifact in enumerate(artifacts, start=1):
        if not policy.download_artifacts:
            staged.append(StagedArtifact(artifact=artifact, status="not_downloaded"))
            continue
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                with client.stream("GET", artifact.url) as response:
                    response.raise_for_status()
                    hasher = hashlib.sha256()
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        total += len(chunk)
                        if policy.max_artifact_bytes > 0 and total > policy.max_artifact_bytes:
                            raise ValueError(
                                f"artifact exceeds max_artifact_bytes={policy.max_artifact_bytes}"
                            )
                        hasher.update(chunk)
                        chunks.append(chunk)
            data = b"".join(chunks)
            digest = hasher.hexdigest()
            findings = _scan_bytes(data)
            if artifact.sha256 and artifact.sha256.lower() != digest:
                staged.append(
                    StagedArtifact(
                        artifact=artifact,
                        status="checksum_mismatch",
                        sha256=digest,
                        bytes_downloaded=len(data),
                        findings=findings,
                        error="declared sha256 did not match downloaded artifact",
                    )
                )
                continue
            target = target_dir / _safe_filename(artifact, index)
            target.write_bytes(data)
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
            staged.append(
                StagedArtifact(
                    artifact=artifact,
                    status="staged",
                    sha256=digest,
                    local_path=str(target),
                    bytes_downloaded=len(data),
                    findings=findings,
                )
            )
        except Exception as exc:
            staged.append(StagedArtifact(artifact=artifact, status="error", error=str(exc)))
    return staged


def artifacts_need_review(artifacts: list[RemoteArtifact], staged: list[StagedArtifact]) -> bool:
    if any(a.needs_review for a in artifacts):
        return True
    if any(item.status != "staged" for item in staged):
        return True
    if any(item.findings for item in staged):
        return True
    return False
