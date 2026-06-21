from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_default_hermes_root

DEFAULT_SKILLS = [
    "delegate_task_to_sasha_hermes",
    "delegate_engineering_task",
    "delegate_research_task",
    "request_summary",
    "submit_artifact_for_review",
]

HIGH_IMPACT_DEFAULTS = [
    "code_execution",
    "external_posting",
    "file_write",
    "script_execution",
    "config_change",
]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _list(value: Any, *, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return list(default or [])


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(slots=True)
class PeerPolicy:
    """Per-peer A2A authorization and routing policy.

    ``token_sha256`` is preferred over raw tokens in config.yaml. For local dev,
    ``token_env`` can name an environment variable whose value is hashed at
    runtime and never logged.
    """

    id: str
    token_sha256: str = ""
    token_env: str = ""
    trusted_subject_header: str = ""
    trusted_subject: str = ""
    allowed_skills: list[str] = field(default_factory=lambda: list(DEFAULT_SKILLS))
    default_skill: str = "delegate_task_to_sasha_hermes"
    default_assignee: str = "engineer"
    board: str = ""
    tenant: str = "a2a"
    priority: int = 0
    max_payload_bytes: int = 200_000
    max_artifact_bytes: int = 10 * 1024 * 1024
    allowed_artifact_domains: list[str] = field(default_factory=list)
    requires_human_review_for: list[str] = field(
        default_factory=lambda: list(HIGH_IMPACT_DEFAULTS)
    )
    download_artifacts: bool = True

    @classmethod
    def from_mapping(cls, peer_id: str, raw: dict[str, Any]) -> "PeerPolicy":
        token_env = str(raw.get("token_env") or raw.get("bearer_token_env") or "").strip()
        token_sha = str(raw.get("token_sha256") or raw.get("bearer_token_sha256") or "").strip()
        # Escape hatch for tests/prototypes only. Prefer token_sha256 or token_env.
        raw_token = str(raw.get("token") or raw.get("bearer_token") or "").strip()
        if raw_token and not token_sha:
            token_sha = _sha256(raw_token)
        if token_env and not token_sha:
            token = os.environ.get(token_env, "")
            if token:
                token_sha = _sha256(token)
        return cls(
            id=peer_id,
            token_sha256=token_sha,
            token_env=token_env,
            trusted_subject_header=str(raw.get("trusted_subject_header") or "").strip(),
            trusted_subject=str(raw.get("trusted_subject") or "").strip(),
            allowed_skills=_list(raw.get("allowed_skills"), default=DEFAULT_SKILLS),
            default_skill=str(raw.get("default_skill") or "delegate_task_to_sasha_hermes").strip(),
            default_assignee=str(raw.get("default_assignee") or raw.get("assignee") or "engineer").strip(),
            board=str(raw.get("board") or "").strip(),
            tenant=str(raw.get("tenant") or "a2a").strip(),
            priority=int(raw.get("priority") or 0),
            max_payload_bytes=int(raw.get("max_payload_bytes") or 200_000),
            max_artifact_bytes=int(raw.get("max_artifact_bytes") or 10 * 1024 * 1024),
            allowed_artifact_domains=_list(raw.get("allowed_artifact_domains")),
            requires_human_review_for=_list(
                raw.get("requires_human_review_for"), default=HIGH_IMPACT_DEFAULTS
            ),
            download_artifacts=_bool(raw.get("download_artifacts"), True),
        )

    def resolved_token_sha256(self) -> str:
        if self.token_env:
            token = os.environ.get(self.token_env, "")
            if token:
                return _sha256(token)
        return self.token_sha256


@dataclass(slots=True)
class SidecarConfig:
    """Runtime settings for the Hermes A2A sidecar."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    public_url: str = "http://127.0.0.1:8765"
    rpc_path: str = "/a2a"
    agent_name: str = "Sasha Hermes Agent"
    description: str = (
        "A narrow Agent2Agent facade for delegating reviewed tasks to Sasha's Hermes runtime."
    )
    version: str = "0.1.0"
    provider_name: str = "Hermes Agent"
    provider_url: str = "https://hermes-agent.nousresearch.com/docs"
    board: str = ""
    audit_db_path: Path = field(
        default_factory=lambda: get_default_hermes_root() / "a2a" / "sidecar.db"
    )
    artifact_root: Path = field(
        default_factory=lambda: get_default_hermes_root() / "a2a" / "artifacts"
    )
    peers: dict[str, PeerPolicy] = field(default_factory=dict)
    allow_insecure_local: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SidecarConfig":
        peers_raw = raw.get("peers") or {}
        peers: dict[str, PeerPolicy] = {}
        if isinstance(peers_raw, dict):
            for peer_id, peer_raw in peers_raw.items():
                if isinstance(peer_raw, dict):
                    peers[str(peer_id)] = PeerPolicy.from_mapping(str(peer_id), peer_raw)
        elif isinstance(peers_raw, list):
            for item in peers_raw:
                if isinstance(item, dict):
                    peer_id = str(item.get("id") or item.get("name") or "").strip()
                    if peer_id:
                        peers[peer_id] = PeerPolicy.from_mapping(peer_id, item)

        env_token = os.environ.get("HERMES_A2A_TOKEN", "")
        if env_token and "env" not in peers:
            peers["env"] = PeerPolicy(
                id="env",
                token_env="HERMES_A2A_TOKEN",
                token_sha256=_sha256(env_token),
                allowed_skills=_list(raw.get("allowed_skills"), default=DEFAULT_SKILLS),
                default_assignee=str(raw.get("default_assignee") or "engineer"),
                board=str(raw.get("board") or ""),
            )

        root = get_default_hermes_root()
        audit_db_raw = str(raw.get("audit_db_path") or "").strip()
        artifact_root_raw = str(raw.get("artifact_root") or "").strip()
        return cls(
            enabled=_bool(raw.get("enabled"), False),
            host=str(raw.get("host") or "127.0.0.1"),
            port=int(raw.get("port") or 8765),
            public_url=str(raw.get("public_url") or "http://127.0.0.1:8765").rstrip("/"),
            rpc_path=str(raw.get("rpc_path") or "/a2a"),
            agent_name=str(raw.get("agent_name") or "Sasha Hermes Agent"),
            description=str(raw.get("description") or cls.description),
            version=str(raw.get("version") or "0.1.0"),
            provider_name=str(raw.get("provider_name") or "Hermes Agent"),
            provider_url=str(raw.get("provider_url") or "https://hermes-agent.nousresearch.com/docs"),
            board=str(raw.get("board") or ""),
            audit_db_path=(Path(audit_db_raw).expanduser() if audit_db_raw else root / "a2a" / "sidecar.db"),
            artifact_root=(Path(artifact_root_raw).expanduser() if artifact_root_raw else root / "a2a" / "artifacts"),
            peers=peers,
            allow_insecure_local=_bool(raw.get("allow_insecure_local"), False),
        )

    @property
    def rpc_url(self) -> str:
        path = self.rpc_path if self.rpc_path.startswith("/") else f"/{self.rpc_path}"
        return f"{self.public_url}{path}"


def load_sidecar_config(overrides: dict[str, Any] | None = None) -> SidecarConfig:
    """Load ``a2a:`` settings from Hermes config.yaml and apply CLI overrides."""

    try:
        from hermes_cli.config import load_config

        base = dict(load_config().get("a2a") or {})
    except Exception:
        base = {}
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
    return SidecarConfig.from_mapping(base)
