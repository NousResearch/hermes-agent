from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import MonicaConfig, runtime_root
from .repo_manager import (
    RepoManagerError,
    safe_branch_prefix,
    safe_default_branch,
    safe_repo_local_name,
)
from .secrets import MONICA_SLACK_APP_TOKEN, MONICA_SLACK_BOT_TOKEN

_SLACK_CHANNEL_ID_RE = re.compile(r"^[CDG][A-Z0-9]{2,}$")
_SLACK_USER_ID_RE = re.compile(r"^[UW][A-Z0-9_]{2,}$")


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ReadinessReport:
    rollout_mode: str
    runtime_root_value: str
    warnings: tuple[ReadinessIssue, ...]
    failures: tuple[ReadinessIssue, ...]

    @property
    def ready(self) -> bool:
        return not self.failures

    def first_approval_failure(self) -> str:
        if not self.failures:
            return ""
        by_code = {issue.code: issue.message for issue in self.failures}
        for code in _APPROVAL_FAILURE_PRIORITY:
            message = by_code.get(code)
            if message:
                return message
        return self.failures[0].message


def check_monica_readiness(
    *,
    config: MonicaConfig,
    environ: dict[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
    module_available: Callable[[str], bool] | None = None,
    command_succeeds: Callable[[tuple[str, ...]], bool] | None = None,
) -> ReadinessReport:
    env = environ if environ is not None else os.environ
    resolve = which or shutil.which
    has_module = module_available or _module_available
    command_ok = command_succeeds or _command_succeeds
    verify_commands = command_succeeds is not None or which is None
    failures: list[ReadinessIssue] = []
    warnings: list[ReadinessIssue] = []
    runtime_root_value = ""

    def require(code: str, condition: bool, message: str) -> None:
        if not condition:
            failures.append(ReadinessIssue(code, message))

    def warn(code: str, condition: bool, message: str) -> None:
        if not condition:
            warnings.append(ReadinessIssue(code, message))

    require("config_enabled", config.enabled, "mobile_bug_agent.enabled is false")
    try:
        runtime_root_value = str(runtime_root(config))
    except ValueError as exc:
        failures.append(ReadinessIssue("runtime_root", str(exc)))
    require(
        "runtime_chandler",
        not _runtime_path_mentions_chandler(config.runtime.home_subdir),
        "mobile_bug_agent.runtime.home_subdir must not point at a Chandler runtime path",
    )
    require(
        "slack_bot_token",
        _has_secret(env, MONICA_SLACK_BOT_TOKEN),
        f"{MONICA_SLACK_BOT_TOKEN} is missing",
    )
    require(
        "slack_sdk",
        bool(has_module("slack_sdk")),
        "slack-sdk Python package is not installed; install `hermes-agent[slack]` or `hermes-agent[messaging]`",
    )
    warn(
        "slack_app_token",
        _has_secret(env, MONICA_SLACK_APP_TOKEN),
        f"{MONICA_SLACK_APP_TOKEN} is missing; Monica Slack Socket Mode may not start",
    )
    warn(
        "slack_bot_user_ids_empty_warning",
        bool(config.slack.bot_user_ids),
        "slack.bot_user_ids is empty; Monica will rely only on app_mention events",
    )
    bot_id_values = [
        user_id for user_id in config.slack.bot_user_ids if user_id.upper().startswith("B")
    ]
    require(
        "slack_bot_id_value",
        not bot_id_values,
        (
            "slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
            f"not bot_id values like {bot_id_values[0]}"
        ) if bot_id_values else "",
    )
    handle_style_bot_ids = [
        user_id for user_id in config.slack.bot_user_ids if str(user_id).strip().startswith("@")
    ]
    require(
        "slack_bot_handle_value",
        not handle_style_bot_ids,
        (
            "slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
            f"not handles like {handle_style_bot_ids[0]}"
        ) if handle_style_bot_ids else "",
    )
    invalid_bot_user_ids = [
        user_id
        for user_id in config.slack.bot_user_ids
        if not str(user_id).strip().startswith("@")
        and not str(user_id).upper().startswith("B")
        and not _is_slack_user_id(user_id)
    ]
    require(
        "slack_bot_user_id_invalid",
        not invalid_bot_user_ids,
        (
            "slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
            f"not invalid values like {invalid_bot_user_ids[0]}"
        ) if invalid_bot_user_ids else "",
    )
    invalid_channels = [
        channel for channel in config.slack.allowed_channels if not _is_slack_channel_id(channel)
    ]
    require(
        "slack_allowed_channel_invalid",
        not invalid_channels,
        (
            "slack.allowed_channels must contain Slack channel IDs like C123 or G123, "
            f"not names like {invalid_channels[0]}"
        ) if invalid_channels else "",
    )

    side_effect_rollout = config.rollout_mode in {"linear_only", "local_fix_only", "approved_pr"}
    if side_effect_rollout:
        require(
            "slack_bot_user_ids_empty",
            bool(config.slack.bot_user_ids),
            "slack.bot_user_ids is empty; configure Monica's Slack mention user ID",
        )
        require(
            "slack_allowed_channels_empty",
            bool(config.slack.allowed_channels),
            "slack.allowed_channels is empty; configure the Slack channels Monica may act in",
        )
    else:
        warn(
            "slack_allowed_channels_empty_warning",
            bool(config.slack.allowed_channels),
            "slack.allowed_channels is empty; Monica can run in any channel where the Slack app is present",
        )

    scopes = _scope_set(env.get("SLACK_BOT_SCOPES") or env.get("SLACK_SCOPES"))
    if scopes:
        require(
            "slack_scope_app_mentions",
            "app_mentions:read" in scopes,
            "SLACK_BOT_SCOPES is missing required scope: app_mentions:read",
        )
        require(
            "slack_scope_chat_write",
            "chat:write" in scopes,
            "SLACK_BOT_SCOPES is missing required scope: chat:write",
        )
        require(
            "slack_scope_history",
            bool(scopes & {"channels:history", "groups:history", "im:history", "mpim:history"}),
            "SLACK_BOT_SCOPES needs at least one thread history scope: channels:history, groups:history, im:history, or mpim:history",
        )
        if config.slack.download_attachments:
            require(
                "slack_scope_files_read",
                "files:read" in scopes,
                "SLACK_BOT_SCOPES is missing required scope: files:read",
            )

    if config.rollout_mode not in {"dry_run", "linear_only", "local_fix_only", "approved_pr"}:
        failures.append(ReadinessIssue("rollout_mode", f"unknown rollout_mode: {config.rollout_mode}"))

    if config.rollout_mode in {"linear_only", "local_fix_only", "approved_pr"}:
        require(
            "loop_create_linear",
            config.loop.create_linear,
            f"loop.create_linear must be true in {config.rollout_mode} mode",
        )

    if config.rollout_mode in {"linear_only", "local_fix_only", "approved_pr"} and config.loop.create_linear:
        require("linear_api_key", _has_secret(env, "LINEAR_API_KEY"), "LINEAR_API_KEY is missing")
        require("linear_team_id", bool(config.linear.team_id), "linear.team_id is missing")

    if config.rollout_mode in {"local_fix_only", "approved_pr"}:
        require(
            "loop_require_fix_approval",
            config.loop.require_fix_approval,
            f"loop.require_fix_approval must be true in {config.rollout_mode} mode",
        )
        require(
            "worker_session_prefix",
            "monica" in config.runtime.worker_session_prefix.lower(),
            "runtime.worker_session_prefix must include `monica` to keep worker sessions segregated",
        )
        require("repo_url", bool(config.repo.url), "repo.url is missing")
        try:
            safe_repo_local_name(config.repo.local_name)
        except RepoManagerError as exc:
            failures.append(ReadinessIssue("repo_local_name", str(exc)))
        try:
            safe_branch_prefix(config.repo.branch_prefix)
        except RepoManagerError as exc:
            failures.append(ReadinessIssue("repo_branch_prefix", str(exc)))
        try:
            safe_default_branch(config.repo.default_branch)
        except RepoManagerError as exc:
            failures.append(ReadinessIssue("repo_default_branch", str(exc)))
        require("verification_commands", bool(config.verification.commands), "verification.commands is empty")
        require("git_executable", resolve("git") is not None, "git executable was not found")
        if config.rollout_mode == "approved_pr":
            require("gh_executable", resolve("gh") is not None, "gh executable was not found")
        if config.worker.backend == "codex_cli":
            require(
                "codex_executable",
                resolve(config.worker.codex_command) is not None,
                f"{config.worker.codex_command} executable was not found",
            )
            require(
                "codex_approval_policy",
                config.worker.codex_approval_policy == "never",
                f"worker.codex_approval_policy must be `never` for {config.rollout_mode} codex_cli runs",
            )
            require(
                "codex_sandbox",
                config.worker.codex_sandbox == "workspace-write",
                f"worker.codex_sandbox must be `workspace-write` for {config.rollout_mode} codex_cli runs",
            )
        elif config.worker.backend != "internal_agent":
            failures.append(ReadinessIssue("worker_backend", f"unknown worker.backend: {config.worker.backend}"))
        if config.rollout_mode == "approved_pr":
            warn(
                "github_token",
                _has_secret(env, "GITHUB_TOKEN"),
                "GITHUB_TOKEN is missing; this is okay only if gh is already authenticated",
            )
            require(
                "github_auth",
                _has_secret(env, "GITHUB_TOKEN")
                or not verify_commands
                or command_ok(("gh", "auth", "status")),
                "GITHUB_TOKEN is missing and `gh auth status` failed; Monica cannot open draft PRs",
            )
        if config.proof.enabled or config.proof.required_for_done:
            warn(
                "proof_commands_empty",
                bool(config.proof.commands),
                "proof.commands is empty; Monica will block at proof after verification",
            )
            proof_platforms = {platform.lower() for platform in config.proof.platform_order}
            uses_builtin_simulator_proof = _uses_builtin_simulator_proof(config.proof.commands)
            if "ios" in proof_platforms:
                if uses_builtin_simulator_proof:
                    require(
                        "proof_ios_dev_client_scheme",
                        _proof_setting_configured(
                            config.proof.dev_client_scheme,
                            config.proof.commands,
                            cli_flag="--dev-client-scheme",
                            env_var="MONICA_DEV_CLIENT_SCHEME",
                        ),
                        (
                            "iOS simulator proof uses Monica's built-in Expo dev-client harness; "
                            "set proof.dev_client_scheme or pass --dev-client-scheme/"
                            "MONICA_DEV_CLIENT_SCHEME in proof.commands"
                        ),
                    )
                    require(
                        "proof_ios_bundle_id",
                        _proof_setting_configured(
                            config.proof.ios_bundle_id,
                            config.proof.commands,
                            cli_flag="--ios-bundle-id",
                            env_var="MONICA_IOS_BUNDLE_ID",
                        ),
                        (
                            "iOS simulator proof uses Monica's built-in Expo dev-client harness; "
                            "set proof.ios_bundle_id or pass --ios-bundle-id/"
                            "MONICA_IOS_BUNDLE_ID in proof.commands"
                        ),
                    )
                ios_tools_present = resolve("xcrun") is not None and resolve("xcodebuild") is not None
                ios_ready = ios_tools_present and (
                    not verify_commands
                    or (
                        command_ok(("xcrun", "--find", "simctl"))
                        and command_ok(("xcodebuild", "-version"))
                    )
                )
                warn(
                    "ios_proof_tooling",
                    ios_ready,
                    (
                        "iOS proof is configured but xcrun/simctl or xcodebuild was not found; "
                        "Monica will block at proof until a simulator is available"
                    ),
                )
            if "android" in proof_platforms:
                warn(
                    "android_proof_tooling",
                    resolve("adb") is not None and resolve("emulator") is not None,
                    (
                        "Android proof is configured but adb or emulator was not found; "
                        "Monica will block at proof until an emulator is available"
                    ),
                )
        require(
            "slack_approvers_empty",
            bool(config.slack.approver_user_ids),
            "slack.approver_user_ids is empty; configure at least one Monica code approver",
        )
        handle_style_approvers = [
            user_id
            for user_id in config.slack.approver_user_ids
            if str(user_id).strip().startswith("@")
        ]
        require(
            "slack_approver_handle_value",
            not handle_style_approvers,
            (
                "slack.approver_user_ids must contain Slack user IDs like U123, "
                f"not handles like {handle_style_approvers[0]}"
            ) if handle_style_approvers else "",
        )
        invalid_approvers = [
            user_id
            for user_id in config.slack.approver_user_ids
            if not str(user_id).strip().startswith("@")
            and not _is_slack_user_id(user_id)
        ]
        require(
            "slack_approver_invalid",
            not invalid_approvers,
            (
                "slack.approver_user_ids must contain Slack user IDs like U123, "
                f"not invalid values like {invalid_approvers[0]}"
            ) if invalid_approvers else "",
        )

    return ReadinessReport(
        rollout_mode=config.rollout_mode,
        runtime_root_value=runtime_root_value,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )


_APPROVAL_FAILURE_PRIORITY = (
    "config_enabled",
    "rollout_mode",
    "loop_create_linear",
    "loop_require_fix_approval",
    "worker_session_prefix",
    "runtime_root",
    "runtime_chandler",
    "linear_api_key",
    "linear_team_id",
    "repo_url",
    "repo_local_name",
    "repo_branch_prefix",
    "repo_default_branch",
    "verification_commands",
    "slack_approvers_empty",
    "slack_approver_handle_value",
    "slack_approver_invalid",
    "git_executable",
    "gh_executable",
    "github_auth",
    "worker_backend",
    "codex_executable",
    "codex_approval_policy",
    "codex_sandbox",
    "slack_bot_token",
    "slack_sdk",
    "slack_bot_user_ids_empty",
    "slack_bot_id_value",
    "slack_bot_handle_value",
    "slack_bot_user_id_invalid",
    "slack_allowed_channels_empty",
    "slack_allowed_channel_invalid",
    "slack_scope_app_mentions",
    "slack_scope_chat_write",
    "slack_scope_history",
    "slack_scope_files_read",
)


def _has_secret(env: dict[str, str], key: str) -> bool:
    return bool(str(env.get(key) or "").strip())


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _command_succeeds(command: tuple[str, ...]) -> bool:
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _uses_builtin_simulator_proof(commands: tuple[str, ...]) -> bool:
    return any("plugins.mobile_bug_agent.simulator_proof" in command for command in commands)


def _proof_setting_configured(
    config_value: str | None,
    commands: tuple[str, ...],
    *,
    cli_flag: str,
    env_var: str,
) -> bool:
    if str(config_value or "").strip():
        return True
    assignment_re = re.compile(rf"(?:^|[\s;])(?:export\s+)?{re.escape(env_var)}\s*=")
    return any(cli_flag in command or assignment_re.search(command) for command in commands)


def _scope_set(value: object) -> set[str]:
    return {
        part.strip()
        for part in str(value or "").replace("\n", ",").replace(" ", ",").split(",")
        if part.strip()
    }


def _is_slack_channel_id(value: str) -> bool:
    return bool(_SLACK_CHANNEL_ID_RE.fullmatch(str(value or "").strip()))


def _is_slack_user_id(value: str) -> bool:
    return bool(_SLACK_USER_ID_RE.fullmatch(str(value or "").strip()))


def _runtime_path_mentions_chandler(value: str) -> bool:
    return any(
        "chandler" in part.lower()
        for part in re.split(r"[\\/]+", str(value or ""))
        if part
    )
