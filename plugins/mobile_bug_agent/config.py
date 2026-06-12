from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class SlackConfig:
    allowed_channels: tuple[str, ...] = ()
    approver_user_ids: tuple[str, ...] = ()
    bot_user_ids: tuple[str, ...] = ()
    download_attachments: bool = True


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int = 8
    timeout_minutes: int = 30
    no_progress_limit: int = 2
    create_linear: bool = True
    require_fix_approval: bool = True
    max_thread_messages: int = 40
    max_attachment_bytes: int = 15_000_000


@dataclass(frozen=True)
class LinearConfig:
    team_id: str = ""
    project_id: str = ""
    label_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepoConfig:
    url: str = ""
    local_name: str = "mobile-app"
    default_branch: str = "main"
    branch_prefix: str = "monica"


@dataclass(frozen=True)
class VerificationConfig:
    commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofConfig:
    enabled: bool = False
    required_for_done: bool = False
    platform_order: tuple[str, ...] = ("ios", "android")
    artifact_dir: str = "proof"
    commands: tuple[str, ...] = ()
    dev_client_scheme: str = ""
    ios_simulator_udid: str = ""
    ios_bundle_id: str = ""
    android_serial: str = ""
    android_avd: str = ""
    android_package: str = ""
    timeout_minutes: int = 10


@dataclass(frozen=True)
class RuntimeConfig:
    home_subdir: str = "agents/monica"
    worker_session_prefix: str = "monica"
    skip_memory: bool = True


@dataclass(frozen=True)
class WorkerConfig:
    backend: str = "codex_cli"
    codex_command: str = "codex"
    codex_model: str = ""
    codex_profile: str = ""
    codex_sandbox: str = "workspace-write"
    codex_approval_policy: str = "never"
    timeout_minutes: int = 45


@dataclass(frozen=True)
class MonicaConfig:
    enabled: bool = False
    rollout_mode: str = "dry_run"
    dry_run: bool = True
    slack: SlackConfig = field(default_factory=SlackConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    linear: LinearConfig = field(default_factory=LinearConfig)
    repo: RepoConfig = field(default_factory=RepoConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    proof: ProofConfig = field(default_factory=ProofConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return ()


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def config_from_mapping(data: Mapping[str, Any] | None) -> MonicaConfig:
    raw = data or {}
    slack = raw.get("slack") if isinstance(raw.get("slack"), Mapping) else {}
    loop = raw.get("loop") if isinstance(raw.get("loop"), Mapping) else {}
    linear = raw.get("linear") if isinstance(raw.get("linear"), Mapping) else {}
    repo = raw.get("repo") if isinstance(raw.get("repo"), Mapping) else {}
    verification = raw.get("verification") if isinstance(raw.get("verification"), Mapping) else {}
    proof = raw.get("proof") if isinstance(raw.get("proof"), Mapping) else {}
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), Mapping) else {}
    worker = raw.get("worker") if isinstance(raw.get("worker"), Mapping) else {}

    raw_rollout_mode = str(raw.get("rollout_mode") or "").strip()
    raw_dry_run = _as_bool(raw.get("dry_run"), True)
    rollout_mode = raw_rollout_mode or ("dry_run" if raw_dry_run else "linear_only")
    dry_run = rollout_mode == "dry_run" if raw_rollout_mode else raw_dry_run

    return MonicaConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        rollout_mode=rollout_mode,
        dry_run=dry_run,
        slack=SlackConfig(
            allowed_channels=_as_tuple(slack.get("allowed_channels")),
            approver_user_ids=_as_tuple(slack.get("approver_user_ids")),
            bot_user_ids=_as_tuple(slack.get("bot_user_ids")),
            download_attachments=_as_bool(slack.get("download_attachments"), True),
        ),
        loop=LoopConfig(
            max_iterations=_as_int(loop.get("max_iterations"), 8),
            timeout_minutes=_as_int(loop.get("timeout_minutes"), 30),
            no_progress_limit=_as_int(loop.get("no_progress_limit"), 2),
            create_linear=_as_bool(loop.get("create_linear"), True),
            require_fix_approval=_as_bool(loop.get("require_fix_approval"), True),
            max_thread_messages=_as_int(loop.get("max_thread_messages"), 40),
            max_attachment_bytes=_as_int(loop.get("max_attachment_bytes"), 15_000_000),
        ),
        linear=LinearConfig(
            team_id=str(linear.get("team_id") or "").strip(),
            project_id=str(linear.get("project_id") or "").strip(),
            label_ids=_as_tuple(linear.get("label_ids")),
        ),
        repo=RepoConfig(
            url=str(repo.get("url") or "").strip(),
            local_name=str(repo.get("local_name") or "mobile-app").strip() or "mobile-app",
            default_branch=str(repo.get("default_branch") or "main").strip() or "main",
            branch_prefix=str(repo.get("branch_prefix") or "monica").strip() or "monica",
        ),
        verification=VerificationConfig(
            commands=_as_tuple(verification.get("commands")),
        ),
        proof=ProofConfig(
            enabled=_as_bool(proof.get("enabled"), False),
            required_for_done=_as_bool(proof.get("required_for_done"), False),
            platform_order=_as_tuple(proof.get("platform_order")) or ("ios", "android"),
            artifact_dir=str(proof.get("artifact_dir") or "proof").strip() or "proof",
            commands=_as_tuple(proof.get("commands")),
            dev_client_scheme=str(proof.get("dev_client_scheme") or "").strip(),
            ios_simulator_udid=str(proof.get("ios_simulator_udid") or "").strip(),
            ios_bundle_id=str(proof.get("ios_bundle_id") or "").strip(),
            android_serial=str(proof.get("android_serial") or "").strip(),
            android_avd=str(proof.get("android_avd") or "").strip(),
            android_package=str(proof.get("android_package") or "").strip(),
            timeout_minutes=_as_int(proof.get("timeout_minutes"), 10),
        ),
        runtime=RuntimeConfig(
            home_subdir=(
                str(runtime.get("home_subdir") or "agents/monica").strip()
                or "agents/monica"
            ),
            worker_session_prefix=(
                str(runtime.get("worker_session_prefix") or "monica").strip() or "monica"
            ),
            skip_memory=_as_bool(runtime.get("skip_memory"), True),
        ),
        worker=WorkerConfig(
            backend=str(worker.get("backend") or "codex_cli").strip() or "codex_cli",
            codex_command=str(worker.get("codex_command") or "codex").strip() or "codex",
            codex_model=str(worker.get("codex_model") or "").strip(),
            codex_profile=str(worker.get("codex_profile") or "").strip(),
            codex_sandbox=(
                str(worker.get("codex_sandbox") or "workspace-write").strip()
                or "workspace-write"
            ),
            codex_approval_policy=(
                str(worker.get("codex_approval_policy") or "never").strip() or "never"
            ),
            timeout_minutes=_as_int(worker.get("timeout_minutes"), 45),
        ),
    )


def load_monica_config() -> MonicaConfig:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return MonicaConfig()

    raw = config.get("mobile_bug_agent") if isinstance(config, Mapping) else None
    return config_from_mapping(raw if isinstance(raw, Mapping) else None)


def runtime_root(config: MonicaConfig) -> Path:
    raw = config.runtime.home_subdir.strip() or "agents/monica"
    if _path_mentions_chandler(raw):
        raise ValueError(
            "mobile_bug_agent.runtime.home_subdir must not point at a Chandler runtime path."
        )
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    home = get_hermes_home()
    root = (home / path).resolve()
    home_resolved = home.resolve()
    if root != home_resolved and home_resolved not in root.parents:
        raise ValueError("mobile_bug_agent.runtime.home_subdir must stay inside HERMES_HOME.")
    return root


def _path_mentions_chandler(value: str) -> bool:
    return any(
        "chandler" in part.lower()
        for part in str(value or "").replace("\\", "/").split("/")
        if part
    )
