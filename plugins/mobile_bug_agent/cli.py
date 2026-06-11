from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Any

from .config import MonicaConfig, load_monica_config, runtime_root
from .linear_client import LinearClient, LinearClientError
from .loop import MonicaLoop
from .readiness import check_monica_readiness, _is_slack_channel_id, _is_slack_user_id
from .repo_manager import (
    RepoManagerError,
    safe_branch_prefix,
    safe_default_branch,
    safe_repo_local_name,
)
from .secrets import MONICA_SLACK_BOT_TOKEN, monica_slack_bot_token
from .skills import DefaultMonicaSkills
from .slack_client import SlackClientError, SlackThreadClient
from .state import MonicaState

RETRYABLE_STATUSES = {"blocked", "failed", "needs_clarification", "proof_blocked"}
_SIMULATION_THREAD_SEQUENCE = 0


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="mobile_bug_agent_action")

    status_p = subs.add_parser("status", help="List recent Monica runs")
    status_p.add_argument("--limit", type=int, default=20)
    status_p.add_argument("--json", action="store_true", help="Print recent runs as JSON")

    doctor_p = subs.add_parser("doctor", help="Check Monica rollout readiness")
    doctor_p.add_argument(
        "--rollout-mode",
        choices=("dry_run", "linear_only", "local_fix_only", "approved_pr"),
        default="",
        help="Preflight a rollout mode without changing config",
    )
    doctor_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable readiness JSON",
    )

    setup_plan_p = subs.add_parser(
        "setup-plan",
        help="Print ordered Monica rollout setup steps",
    )
    setup_plan_p.set_defaults(mobile_bug_agent_action="setup_plan")
    setup_plan_p.add_argument(
        "--rollout-mode",
        choices=("dry_run", "linear_only", "local_fix_only", "approved_pr"),
        default="",
        help="Plan setup for a rollout mode without changing config",
    )
    setup_plan_p.add_argument("--json", action="store_true", help="Print setup plan as JSON")

    manifest_p = subs.add_parser(
        "slack-manifest",
        help="Print a Slack App Manifest for Monica setup",
    )
    manifest_p.set_defaults(mobile_bug_agent_action="slack_manifest")
    manifest_p.add_argument("--app-name", default="Monica")
    manifest_p.add_argument("--bot-display-name", default="monica")

    slack_metadata_p = subs.add_parser(
        "slack-metadata",
        help="List Slack bot identity and channel IDs for Monica setup",
    )
    slack_metadata_p.set_defaults(mobile_bug_agent_action="slack_metadata")
    slack_metadata_p.add_argument(
        "--json",
        action="store_true",
        help="Print Slack metadata as JSON",
    )

    linear_metadata_p = subs.add_parser(
        "linear-metadata",
        help="List Linear team/project/label IDs for Monica setup",
    )
    linear_metadata_p.set_defaults(mobile_bug_agent_action="linear_metadata")
    linear_metadata_p.add_argument(
        "--json",
        action="store_true",
        help="Print Linear metadata as JSON",
    )

    configure_dry_run_p = subs.add_parser(
        "configure-dry-run",
        help="Persist non-secret Monica settings for dry-run Slack rollout",
    )
    configure_dry_run_p.set_defaults(mobile_bug_agent_action="configure_dry_run")
    configure_dry_run_p.add_argument(
        "--bot-user-id",
        dest="bot_user_ids",
        action="append",
        default=[],
        help="Optional Monica Slack mention user ID, e.g. U012ABCDEF",
    )
    configure_dry_run_p.add_argument(
        "--channel-id",
        dest="channel_ids",
        action="append",
        default=[],
        help="Optional private dry-run Slack channel ID, e.g. C012ABCDEF or G012ABCDEF",
    )

    configure_linear_p = subs.add_parser(
        "configure-linear-only",
        help="Persist non-secret Monica settings for Linear-only rollout",
    )
    configure_linear_p.set_defaults(mobile_bug_agent_action="configure_linear_only")
    configure_linear_p.add_argument(
        "--bot-user-id",
        dest="bot_user_ids",
        action="append",
        required=True,
        help="Monica Slack mention user ID, e.g. U012ABCDEF",
    )
    configure_linear_p.add_argument(
        "--channel-id",
        dest="channel_ids",
        action="append",
        required=True,
        help="Allowed Slack channel ID, e.g. C012ABCDEF or G012ABCDEF",
    )
    configure_linear_p.add_argument(
        "--linear-team-id",
        required=True,
        help="Linear team ID for mobile app issues",
    )
    configure_linear_p.add_argument(
        "--linear-project-id",
        default="",
        help="Optional Linear project ID",
    )
    configure_linear_p.add_argument(
        "--linear-label-id",
        dest="linear_label_ids",
        action="append",
        default=[],
        help="Optional Linear label ID; repeat for multiple labels",
    )

    configure_pr_p = subs.add_parser(
        "configure-approved-pr",
        help="Persist non-secret Monica settings for approved draft PR rollout",
    )
    configure_pr_p.set_defaults(mobile_bug_agent_action="configure_approved_pr")
    configure_pr_p.add_argument(
        "--approver-user-id",
        dest="approver_user_ids",
        action="append",
        required=True,
        help="Slack user ID allowed to approve Monica code work, e.g. U012ABCDEF",
    )
    configure_pr_p.add_argument(
        "--repo-url",
        required=True,
        help="React Native app repository URL",
    )
    configure_pr_p.add_argument(
        "--verification-command",
        dest="verification_commands",
        action="append",
        required=True,
        help="Command Monica must run before opening a draft PR; repeat for multiple commands",
    )
    configure_pr_p.add_argument("--repo-local-name", default="mobile-app")
    configure_pr_p.add_argument("--default-branch", default="main")
    configure_pr_p.add_argument("--branch-prefix", default="monica")

    configure_local_fix_p = subs.add_parser(
        "configure-local-fix-only",
        help="Persist non-secret Monica settings for local fix rollout without push or PR",
    )
    configure_local_fix_p.set_defaults(mobile_bug_agent_action="configure_local_fix_only")
    configure_local_fix_p.add_argument(
        "--approver-user-id",
        dest="approver_user_ids",
        action="append",
        required=True,
        help="Slack user ID allowed to approve Monica code work, e.g. U012ABCDEF",
    )
    configure_local_fix_p.add_argument(
        "--repo-url",
        required=True,
        help="React Native app repository URL",
    )
    configure_local_fix_p.add_argument(
        "--verification-command",
        dest="verification_commands",
        action="append",
        required=True,
        help="Command Monica must run before marking a local fix ready; repeat for multiple commands",
    )
    configure_local_fix_p.add_argument("--repo-local-name", default="mobile-app")
    configure_local_fix_p.add_argument("--default-branch", default="main")
    configure_local_fix_p.add_argument("--branch-prefix", default="monica")

    show_p = subs.add_parser("show", help="Show one Monica run")
    show_p.add_argument("run_id")
    show_p.add_argument("--json", action="store_true", help="Print the run as JSON")

    retry_p = subs.add_parser("retry", help="Retry a Monica run from its last safe gate")
    retry_p.add_argument("run_id")
    retry_p.add_argument("--json", action="store_true", help="Print retry result as JSON")

    approve_p = subs.add_parser("approve", help="Approve a Monica run locally and continue the fix loop")
    approve_p.add_argument("run_id")
    approve_p.add_argument("--user-id", default="local-operator")
    approve_p.add_argument("--json", action="store_true", help="Print approval result as JSON")

    simulate_p = subs.add_parser("simulate", help="Run a local Monica message simulation")
    simulate_p.add_argument("text", nargs="+")
    simulate_p.add_argument("--channel-id", default="LOCAL_MONICA_SIM")
    simulate_p.add_argument("--user-id", default="local-operator")
    simulate_p.add_argument("--thread-ts", default="")
    simulate_p.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="Allow simulate to run real Linear/PR side effects when rollout_mode is not dry_run",
    )
    simulate_p.add_argument("--json", action="store_true", help="Print simulation result as JSON")

    subparser.set_defaults(func=mobile_bug_agent_command)


def mobile_bug_agent_command(args: argparse.Namespace) -> int:
    action = getattr(args, "mobile_bug_agent_action", None)
    config = load_monica_config()
    if action == "doctor":
        return run_doctor_command(
            config=config,
            target_rollout_mode=str(getattr(args, "rollout_mode", "") or ""),
            json_output=bool(getattr(args, "json", False)),
        )
    if action == "setup_plan":
        return run_setup_plan_command(
            config=config,
            target_rollout_mode=str(getattr(args, "rollout_mode", "") or ""),
            json_output=bool(getattr(args, "json", False)),
        )
    if action == "slack_manifest":
        return run_slack_manifest_command(
            config=config,
            app_name=str(getattr(args, "app_name", "") or ""),
            bot_display_name=str(getattr(args, "bot_display_name", "") or ""),
        )
    if action == "slack_metadata":
        return run_slack_metadata_command(json_output=bool(getattr(args, "json", False)))
    if action == "linear_metadata":
        return run_linear_metadata_command(json_output=bool(getattr(args, "json", False)))
    if action == "configure_dry_run":
        return run_configure_dry_run_command(
            bot_user_ids=tuple(getattr(args, "bot_user_ids", ()) or ()),
            channel_ids=tuple(getattr(args, "channel_ids", ()) or ()),
        )
    if action == "configure_linear_only":
        return run_configure_linear_only_command(
            bot_user_ids=tuple(getattr(args, "bot_user_ids", ()) or ()),
            channel_ids=tuple(getattr(args, "channel_ids", ()) or ()),
            linear_team_id=str(getattr(args, "linear_team_id", "") or ""),
            linear_project_id=str(getattr(args, "linear_project_id", "") or ""),
            linear_label_ids=tuple(getattr(args, "linear_label_ids", ()) or ()),
        )
    if action == "configure_approved_pr":
        return run_configure_approved_pr_command(
            approver_user_ids=tuple(getattr(args, "approver_user_ids", ()) or ()),
            repo_url=str(getattr(args, "repo_url", "") or ""),
            verification_commands=tuple(getattr(args, "verification_commands", ()) or ()),
            repo_local_name=str(getattr(args, "repo_local_name", "") or ""),
            default_branch=str(getattr(args, "default_branch", "") or ""),
            branch_prefix=str(getattr(args, "branch_prefix", "") or ""),
        )
    if action == "configure_local_fix_only":
        return run_configure_local_fix_only_command(
            approver_user_ids=tuple(getattr(args, "approver_user_ids", ()) or ()),
            repo_url=str(getattr(args, "repo_url", "") or ""),
            verification_commands=tuple(getattr(args, "verification_commands", ()) or ()),
            repo_local_name=str(getattr(args, "repo_local_name", "") or ""),
            default_branch=str(getattr(args, "default_branch", "") or ""),
            branch_prefix=str(getattr(args, "branch_prefix", "") or ""),
        )

    try:
        state = _open_state(config)
    except ValueError as exc:
        print(f"Monica runtime is not configured correctly: {exc}")
        return 1
    if action == "status":
        return run_status_command(
            state=state,
            limit=int(getattr(args, "limit", 20)),
            json_output=bool(getattr(args, "json", False)),
        )
    if action == "show":
        return run_show_command(
            state=state,
            run_id=str(args.run_id),
            json_output=bool(getattr(args, "json", False)),
        )
    if action == "retry":
        return run_retry_command(
            state=state,
            run_id=str(args.run_id),
            config=config,
            json_output=bool(getattr(args, "json", False)),
        )
    if action == "approve":
        return run_approve_command(
            state=state,
            run_id=str(args.run_id),
            user_id=str(getattr(args, "user_id", "") or "local-operator"),
            config=config,
            json_output=bool(getattr(args, "json", False)),
        )
    if action == "simulate":
        return run_simulate_command(
            config=config,
            state=state,
            text=" ".join(getattr(args, "text", []) or []),
            channel_id=str(getattr(args, "channel_id", "") or "LOCAL_MONICA_SIM"),
            user_id=str(getattr(args, "user_id", "") or "local-operator"),
            thread_ts=str(getattr(args, "thread_ts", "") or ""),
            allow_side_effects=bool(getattr(args, "allow_side_effects", False)),
            json_output=bool(getattr(args, "json", False)),
        )
    print(
        "Usage: hermes mobile-bug-agent "
        "{status|doctor|setup-plan|configure-dry-run|configure-linear-only|configure-local-fix-only|"
        "configure-approved-pr|"
        "show|retry|approve|simulate}"
    )
    return 2


def run_configure_linear_only_command(
    *,
    bot_user_ids: tuple[str, ...],
    channel_ids: tuple[str, ...],
    linear_team_id: str,
    linear_project_id: str = "",
    linear_label_ids: tuple[str, ...] = (),
    load_config_fn: Any | None = None,
    save_config_fn: Any | None = None,
) -> int:
    bot_ids = _clean_sequence(bot_user_ids)
    channels = _clean_sequence(channel_ids)
    team_id = str(linear_team_id or "").strip()
    project_id = str(linear_project_id or "").strip()
    label_ids = _clean_sequence(linear_label_ids)

    errors = _linear_only_config_errors(
        bot_user_ids=bot_ids,
        channel_ids=channels,
        linear_team_id=team_id,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Monica linear_only configuration was not saved.")
        return 1

    if load_config_fn is None or save_config_fn is None:
        from hermes_cli.config import load_config, save_config

        load = load_config if load_config_fn is None else load_config_fn
        save = save_config if save_config_fn is None else save_config_fn
    else:
        load = load_config_fn
        save = save_config_fn

    config = load()
    if not isinstance(config, dict):
        config = {}
    _enable_plugin(config)

    monica_cfg = _ensure_dict(config, "mobile_bug_agent")
    monica_cfg["enabled"] = True
    monica_cfg["rollout_mode"] = "linear_only"
    monica_cfg["dry_run"] = False

    slack_cfg = _ensure_dict(monica_cfg, "slack")
    slack_cfg["bot_user_ids"] = list(bot_ids)
    slack_cfg["allowed_channels"] = list(channels)

    loop_cfg = _ensure_dict(monica_cfg, "loop")
    loop_cfg["create_linear"] = True

    linear_cfg = _ensure_dict(monica_cfg, "linear")
    linear_cfg["team_id"] = team_id
    linear_cfg["project_id"] = project_id
    linear_cfg["label_ids"] = list(label_ids)

    save(config)
    print("Configured Monica for linear_only rollout.")
    print(
        "Secrets were not written; keep MONICA_SLACK_BOT_TOKEN and LINEAR_API_KEY in the Monica profile .env."
    )
    print("Run `hermes mobile-bug-agent doctor --rollout-mode linear_only` before inviting Monica.")
    return 0


def run_configure_dry_run_command(
    *,
    bot_user_ids: tuple[str, ...] = (),
    channel_ids: tuple[str, ...] = (),
    load_config_fn: Any | None = None,
    save_config_fn: Any | None = None,
) -> int:
    bot_ids = _clean_sequence(bot_user_ids)
    channels = _clean_sequence(channel_ids)

    errors = _slack_scope_config_errors(
        bot_user_ids=bot_ids,
        channel_ids=channels,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Monica dry_run configuration was not saved.")
        return 1

    if load_config_fn is None or save_config_fn is None:
        from hermes_cli.config import load_config, save_config

        load = load_config if load_config_fn is None else load_config_fn
        save = save_config if save_config_fn is None else save_config_fn
    else:
        load = load_config_fn
        save = save_config_fn

    config = load()
    if not isinstance(config, dict):
        config = {}
    _enable_plugin(config)

    monica_cfg = _ensure_dict(config, "mobile_bug_agent")
    monica_cfg["enabled"] = True
    monica_cfg["rollout_mode"] = "dry_run"
    monica_cfg["dry_run"] = True

    if bot_ids or channels:
        slack_cfg = _ensure_dict(monica_cfg, "slack")
        if bot_ids:
            slack_cfg["bot_user_ids"] = list(bot_ids)
        if channels:
            slack_cfg["allowed_channels"] = list(channels)

    loop_cfg = _ensure_dict(monica_cfg, "loop")
    loop_cfg["create_linear"] = True
    loop_cfg["require_fix_approval"] = True

    runtime_cfg = _ensure_dict(monica_cfg, "runtime")
    runtime_cfg["home_subdir"] = "agents/monica"
    runtime_cfg["worker_session_prefix"] = "monica"
    runtime_cfg["skip_memory"] = True

    save(config)
    print("Configured Monica for dry_run rollout.")
    print("Secrets were not written; keep MONICA_SLACK_BOT_TOKEN in the Monica profile .env.")
    print("Run `hermes mobile-bug-agent doctor --rollout-mode dry_run` before inviting Monica.")
    return 0


def run_slack_manifest_command(
    *,
    config: MonicaConfig,
    app_name: str = "Monica",
    bot_display_name: str = "monica",
) -> int:
    manifest = _slack_app_manifest(
        config=config,
        app_name=app_name,
        bot_display_name=bot_display_name,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def run_linear_metadata_command(
    *,
    api_key: str | None = None,
    json_output: bool = False,
    client_factory: Any = LinearClient,
) -> int:
    key = str(api_key if api_key is not None else os.environ.get("LINEAR_API_KEY", "")).strip()
    if not key:
        message = "LINEAR_API_KEY is missing; keep it in ~/.hermes/.env before listing Linear IDs."
        if json_output:
            _print_linear_metadata_error_json("linear_api_key_missing", message)
            return 1
        print(message)
        return 1
    try:
        metadata = client_factory(api_key=key).list_workspace_metadata()
    except LinearClientError as exc:
        message = str(exc)
        if json_output:
            _print_linear_metadata_error_json("linear_request_failed", message)
            return 1
        print(f"Could not list Linear metadata: {message}")
        return 1
    if json_output:
        print(json.dumps(_linear_metadata_payload(metadata), sort_keys=True))
        return 0
    _print_linear_metadata_text(metadata)
    return 0


def _linear_metadata_payload(metadata: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "teams": [
            {
                "id": str(getattr(team, "id", "") or ""),
                "key": str(getattr(team, "key", "") or ""),
                "name": str(getattr(team, "name", "") or ""),
            }
            for team in getattr(metadata, "teams", ()) or ()
        ],
        "projects": [
            {
                "id": str(getattr(project, "id", "") or ""),
                "name": str(getattr(project, "name", "") or ""),
                "state": str(getattr(project, "state", "") or ""),
            }
            for project in getattr(metadata, "projects", ()) or ()
        ],
        "labels": [
            {
                "id": str(getattr(label, "id", "") or ""),
                "name": str(getattr(label, "name", "") or ""),
                "color": str(getattr(label, "color", "") or ""),
            }
            for label in getattr(metadata, "labels", ()) or ()
        ],
    }


def _print_linear_metadata_text(metadata: Any) -> None:
    print("Linear metadata for Monica setup")
    print("")
    print("Teams:")
    teams = getattr(metadata, "teams", ()) or ()
    if teams:
        for team in teams:
            key = str(getattr(team, "key", "") or "")
            key_suffix = f" [{key}]" if key else ""
            print(f"- {getattr(team, 'name', '')}{key_suffix}: {getattr(team, 'id', '')}")
    else:
        print("- none returned")
    print("")
    print("Projects:")
    projects = getattr(metadata, "projects", ()) or ()
    if projects:
        for project in projects:
            state = str(getattr(project, "state", "") or "")
            state_suffix = f" ({state})" if state else ""
            print(f"- {getattr(project, 'name', '')}{state_suffix}: {getattr(project, 'id', '')}")
    else:
        print("- none returned")
    print("")
    print("Labels:")
    labels = getattr(metadata, "labels", ()) or ()
    if labels:
        for label in labels:
            print(f"- {getattr(label, 'name', '')}: {getattr(label, 'id', '')}")
    else:
        print("- none returned")
    print("")
    print(
        "Use the chosen team ID with `hermes mobile-bug-agent configure-linear-only "
        "--linear-team-id <id>`."
    )


def _print_linear_metadata_error_json(code: str, message: str) -> None:
    print(
        json.dumps(
            {
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            },
            sort_keys=True,
        )
    )


def run_slack_metadata_command(
    *,
    token: str | None = None,
    json_output: bool = False,
    client_factory: Any = SlackThreadClient.from_token,
) -> int:
    bot_token = str(token if token is not None else monica_slack_bot_token()).strip()
    if not bot_token:
        message = (
            f"{MONICA_SLACK_BOT_TOKEN} is missing; keep it in the Monica profile .env "
            "before listing Slack IDs."
        )
        if json_output:
            _print_slack_metadata_error_json("slack_bot_token_missing", message)
            return 1
        print(message)
        return 1
    try:
        metadata = client_factory(token=bot_token).list_workspace_metadata()
    except SlackClientError as exc:
        message = str(exc)
        if json_output:
            _print_slack_metadata_error_json("slack_request_failed", message)
            return 1
        print(f"Could not list Slack metadata: {message}")
        return 1
    if json_output:
        print(json.dumps(_slack_metadata_payload(metadata), sort_keys=True))
        return 0
    _print_slack_metadata_text(metadata)
    return 0


def _slack_metadata_payload(metadata: Any) -> dict[str, Any]:
    auth = getattr(metadata, "auth", None)
    return {
        "ok": True,
        "auth": {
            "bot_user_id": str(getattr(auth, "bot_user_id", "") or ""),
            "bot_id": str(getattr(auth, "bot_id", "") or ""),
            "team_id": str(getattr(auth, "team_id", "") or ""),
            "team_name": str(getattr(auth, "team_name", "") or ""),
            "team_url": str(getattr(auth, "team_url", "") or ""),
        },
        "channels": [
            {
                "id": str(getattr(channel, "id", "") or ""),
                "name": str(getattr(channel, "name", "") or ""),
                "is_private": bool(getattr(channel, "is_private", False)),
                "is_member": bool(getattr(channel, "is_member", False)),
                "is_archived": bool(getattr(channel, "is_archived", False)),
            }
            for channel in getattr(metadata, "channels", ()) or ()
        ],
    }


def _print_slack_metadata_text(metadata: Any) -> None:
    auth = getattr(metadata, "auth", None)
    print("Slack metadata for Monica setup")
    print("")
    print(f"Workspace: {getattr(auth, 'team_name', '') or '(unknown)'}")
    print(f"Team ID: {getattr(auth, 'team_id', '') or '(unknown)'}")
    print(f"Bot user ID: {getattr(auth, 'bot_user_id', '') or '(unknown)'}")
    bot_id = str(getattr(auth, "bot_id", "") or "")
    if bot_id:
        print(f"Bot ID: {bot_id} (do not use this for slack.bot_user_ids)")
    print("")
    print("Channels visible to Monica:")
    channels = getattr(metadata, "channels", ()) or ()
    if channels:
        for channel in channels:
            privacy = "private" if getattr(channel, "is_private", False) else "public"
            membership = "joined" if getattr(channel, "is_member", False) else "not joined"
            print(
                f"- #{getattr(channel, 'name', '')} ({privacy}, {membership}): "
                f"{getattr(channel, 'id', '')}"
            )
    else:
        print("- none returned")
    print("")
    print(
        "Use the bot user ID and selected channel IDs with "
        "`hermes mobile-bug-agent configure-linear-only --bot-user-id <U...> --channel-id <C...>`."
    )


def _print_slack_metadata_error_json(code: str, message: str) -> None:
    print(
        json.dumps(
            {
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            },
            sort_keys=True,
        )
    )


def _slack_app_manifest(
    *,
    config: MonicaConfig,
    app_name: str,
    bot_display_name: str,
) -> dict[str, Any]:
    scopes = [
        "app_mentions:read",
        "channels:history",
        "groups:history",
        "im:history",
        "im:read",
        "chat:write",
    ]
    if config.slack.download_attachments:
        scopes.append("files:read")
    return {
        "display_information": {
            "name": str(app_name or "").strip() or "Monica",
        },
        "features": {
            "app_home": {
                "home_tab_enabled": False,
                "messages_tab_enabled": True,
                "messages_tab_read_only_enabled": False,
            },
            "bot_user": {
                "display_name": str(bot_display_name or "").strip() or "monica",
                "always_online": True,
            },
        },
        "oauth_config": {
            "scopes": {
                "bot": scopes,
            },
        },
        "settings": {
            "event_subscriptions": {
                "bot_events": ["app_mention", "message.im"],
            },
            "interactivity": {
                "is_enabled": False,
            },
            "org_deploy_enabled": False,
            "socket_mode_enabled": True,
            "token_rotation_enabled": False,
        },
    }


def run_configure_approved_pr_command(
    *,
    approver_user_ids: tuple[str, ...],
    repo_url: str,
    verification_commands: tuple[str, ...],
    repo_local_name: str = "mobile-app",
    default_branch: str = "main",
    branch_prefix: str = "monica",
    load_config_fn: Any | None = None,
    save_config_fn: Any | None = None,
) -> int:
    approvers = _clean_sequence(approver_user_ids)
    commands = _clean_sequence(verification_commands)
    repo_url_value = str(repo_url or "").strip()
    local_name = str(repo_local_name or "mobile-app").strip() or "mobile-app"
    branch = str(default_branch or "main").strip() or "main"
    prefix = str(branch_prefix or "monica").strip() or "monica"

    errors = _approved_pr_config_errors(
        approver_user_ids=approvers,
        repo_url=repo_url_value,
        verification_commands=commands,
        repo_local_name=local_name,
        default_branch=branch,
        branch_prefix=prefix,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Monica approved_pr configuration was not saved.")
        return 1

    if load_config_fn is None or save_config_fn is None:
        from hermes_cli.config import load_config, save_config

        load = load_config if load_config_fn is None else load_config_fn
        save = save_config if save_config_fn is None else save_config_fn
    else:
        load = load_config_fn
        save = save_config_fn

    config = load()
    if not isinstance(config, dict):
        config = {}
    _enable_plugin(config)
    monica_cfg = _ensure_dict(config, "mobile_bug_agent")
    monica_cfg["enabled"] = True
    monica_cfg["rollout_mode"] = "approved_pr"
    monica_cfg["dry_run"] = False

    slack_cfg = _ensure_dict(monica_cfg, "slack")
    slack_cfg["approver_user_ids"] = list(approvers)

    loop_cfg = _ensure_dict(monica_cfg, "loop")
    loop_cfg["create_linear"] = True
    loop_cfg["require_fix_approval"] = True

    repo_cfg = _ensure_dict(monica_cfg, "repo")
    repo_cfg["url"] = repo_url_value
    repo_cfg["local_name"] = local_name
    repo_cfg["default_branch"] = branch
    repo_cfg["branch_prefix"] = prefix

    verification_cfg = _ensure_dict(monica_cfg, "verification")
    verification_cfg["commands"] = list(commands)

    runtime_cfg = _ensure_dict(monica_cfg, "runtime")
    runtime_cfg["home_subdir"] = "agents/monica"
    runtime_cfg["worker_session_prefix"] = "monica"
    runtime_cfg["skip_memory"] = True

    worker_cfg = _ensure_dict(monica_cfg, "worker")
    worker_cfg["backend"] = "codex_cli"
    worker_cfg["codex_command"] = str(worker_cfg.get("codex_command") or "codex").strip() or "codex"
    worker_cfg["codex_sandbox"] = "workspace-write"
    worker_cfg["codex_approval_policy"] = "never"
    if not worker_cfg.get("timeout_minutes"):
        worker_cfg["timeout_minutes"] = 45

    save(config)
    print("Configured Monica for approved_pr rollout.")
    print(
        "Secrets were not written; keep MONICA_SLACK_BOT_TOKEN, LINEAR_API_KEY, and GITHUB_TOKEN/gh auth outside config."
    )
    print("Run `hermes mobile-bug-agent doctor --rollout-mode approved_pr` before approving fixes.")
    return 0


def run_configure_local_fix_only_command(
    *,
    approver_user_ids: tuple[str, ...],
    repo_url: str,
    verification_commands: tuple[str, ...],
    repo_local_name: str = "mobile-app",
    default_branch: str = "main",
    branch_prefix: str = "monica",
    load_config_fn: Any | None = None,
    save_config_fn: Any | None = None,
) -> int:
    return _run_configure_code_rollout_command(
        rollout_mode="local_fix_only",
        saved_label="local_fix_only",
        doctor_mode="local_fix_only",
        approver_user_ids=approver_user_ids,
        repo_url=repo_url,
        verification_commands=verification_commands,
        repo_local_name=repo_local_name,
        default_branch=default_branch,
        branch_prefix=branch_prefix,
        load_config_fn=load_config_fn,
        save_config_fn=save_config_fn,
    )


def _run_configure_code_rollout_command(
    *,
    rollout_mode: str,
    saved_label: str,
    doctor_mode: str,
    approver_user_ids: tuple[str, ...],
    repo_url: str,
    verification_commands: tuple[str, ...],
    repo_local_name: str = "mobile-app",
    default_branch: str = "main",
    branch_prefix: str = "monica",
    load_config_fn: Any | None = None,
    save_config_fn: Any | None = None,
) -> int:
    approvers = _clean_sequence(approver_user_ids)
    commands = _clean_sequence(verification_commands)
    repo_url_value = str(repo_url or "").strip()
    local_name = str(repo_local_name or "mobile-app").strip() or "mobile-app"
    branch = str(default_branch or "main").strip() or "main"
    prefix = str(branch_prefix or "monica").strip() or "monica"

    errors = _approved_pr_config_errors(
        approver_user_ids=approvers,
        repo_url=repo_url_value,
        verification_commands=commands,
        repo_local_name=local_name,
        default_branch=branch,
        branch_prefix=prefix,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Monica {saved_label} configuration was not saved.")
        return 1

    if load_config_fn is None or save_config_fn is None:
        from hermes_cli.config import load_config, save_config

        load = load_config if load_config_fn is None else load_config_fn
        save = save_config if save_config_fn is None else save_config_fn
    else:
        load = load_config_fn
        save = save_config_fn

    config = load()
    if not isinstance(config, dict):
        config = {}
    _enable_plugin(config)
    monica_cfg = _ensure_dict(config, "mobile_bug_agent")
    monica_cfg["enabled"] = True
    monica_cfg["rollout_mode"] = rollout_mode
    monica_cfg["dry_run"] = False

    slack_cfg = _ensure_dict(monica_cfg, "slack")
    slack_cfg["approver_user_ids"] = list(approvers)

    loop_cfg = _ensure_dict(monica_cfg, "loop")
    loop_cfg["create_linear"] = True
    loop_cfg["require_fix_approval"] = True

    repo_cfg = _ensure_dict(monica_cfg, "repo")
    repo_cfg["url"] = repo_url_value
    repo_cfg["local_name"] = local_name
    repo_cfg["default_branch"] = branch
    repo_cfg["branch_prefix"] = prefix

    verification_cfg = _ensure_dict(monica_cfg, "verification")
    verification_cfg["commands"] = list(commands)

    runtime_cfg = _ensure_dict(monica_cfg, "runtime")
    runtime_cfg["home_subdir"] = "agents/monica"
    runtime_cfg["worker_session_prefix"] = "monica"
    runtime_cfg["skip_memory"] = True

    worker_cfg = _ensure_dict(monica_cfg, "worker")
    worker_cfg["backend"] = "codex_cli"
    worker_cfg["codex_command"] = str(worker_cfg.get("codex_command") or "codex").strip() or "codex"
    worker_cfg["codex_sandbox"] = "workspace-write"
    worker_cfg["codex_approval_policy"] = "never"
    if not worker_cfg.get("timeout_minutes"):
        worker_cfg["timeout_minutes"] = 45

    save(config)
    print(f"Configured Monica for {saved_label} rollout.")
    print(
        "Secrets were not written; keep MONICA_SLACK_BOT_TOKEN and LINEAR_API_KEY outside config."
    )
    print(f"Run `hermes mobile-bug-agent doctor --rollout-mode {doctor_mode}` before approving fixes.")
    return 0


def _linear_only_config_errors(
    *,
    bot_user_ids: tuple[str, ...],
    channel_ids: tuple[str, ...],
    linear_team_id: str,
) -> list[str]:
    errors: list[str] = []
    if not bot_user_ids:
        errors.append("at least one --bot-user-id is required")
    if not channel_ids:
        errors.append("at least one --channel-id is required")
    if not str(linear_team_id or "").strip():
        errors.append("--linear-team-id is required")
    errors.extend(_slack_scope_config_errors(bot_user_ids=bot_user_ids, channel_ids=channel_ids))
    return errors


def _slack_scope_config_errors(
    *,
    bot_user_ids: tuple[str, ...],
    channel_ids: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    for user_id in bot_user_ids:
        if str(user_id).startswith("@"):
            errors.append(
                f"slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, not handles like {user_id}"
            )
        elif str(user_id).upper().startswith("B"):
            errors.append(
                f"slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, not bot_id values like {user_id}"
            )
        elif not _is_slack_user_id(user_id):
            errors.append(
                f"slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, not invalid values like {user_id}"
            )
    for channel_id in channel_ids:
        if str(channel_id).startswith("#"):
            errors.append(
                f"slack.allowed_channels must contain Slack channel IDs like C123 or G123, not names like {channel_id}"
            )
        elif not _is_slack_channel_id(channel_id):
            errors.append(
                f"slack.allowed_channels must contain Slack channel IDs like C123 or G123, not invalid values like {channel_id}"
            )
    return errors


def _approved_pr_config_errors(
    *,
    approver_user_ids: tuple[str, ...],
    repo_url: str,
    verification_commands: tuple[str, ...],
    repo_local_name: str,
    default_branch: str,
    branch_prefix: str,
) -> list[str]:
    errors: list[str] = []
    if not approver_user_ids:
        errors.append("at least one --approver-user-id is required")
    if not str(repo_url or "").strip():
        errors.append("--repo-url is required")
    if not verification_commands:
        errors.append("at least one --verification-command is required")
    for user_id in approver_user_ids:
        if str(user_id).startswith("@"):
            errors.append(
                f"slack.approver_user_ids must contain Slack user IDs like U123, not handles like {user_id}"
            )
        elif not _is_slack_user_id(user_id):
            errors.append(
                f"slack.approver_user_ids must contain Slack user IDs like U123, not invalid values like {user_id}"
            )
    for validator, value in (
        (safe_repo_local_name, repo_local_name),
        (safe_default_branch, default_branch),
        (safe_branch_prefix, branch_prefix),
    ):
        try:
            validator(value)
        except RepoManagerError as exc:
            errors.append(str(exc))
    return errors


def _clean_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]
    return tuple(str(item).strip() for item in raw_values if str(item).strip())


def _enable_plugin(config: dict[str, Any]) -> None:
    plugins_cfg = _ensure_dict(config, "plugins")
    enabled = set(_clean_sequence(plugins_cfg.get("enabled", ())))
    disabled = set(_clean_sequence(plugins_cfg.get("disabled", ())))
    enabled.add("mobile-bug-agent")
    disabled.discard("mobile-bug-agent")
    plugins_cfg["enabled"] = sorted(enabled)
    if disabled:
        plugins_cfg["disabled"] = sorted(disabled)
    elif "disabled" in plugins_cfg:
        plugins_cfg["disabled"] = []


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def run_status_command(*, state: MonicaState, limit: int = 20, json_output: bool = False) -> int:
    runs = state.list_runs(limit=limit)
    if json_output:
        print(
            json.dumps(
                {
                    "count": len(runs),
                    "runs": [_run_payload(run, include_raw_event=False) for run in runs],
                },
                sort_keys=True,
            )
        )
        return 0
    if not runs:
        print("No Monica runs found.")
        return 0
    for run in runs:
        parts = [
            run.id,
            run.status,
            f"slack:{run.channel_id}/{run.thread_ts}",
            f"linear:{run.linear_identifier or '-'}",
            f"branch:{run.branch_name or '-'}",
            f"url:{run.pr_url or run.linear_url or '-'}",
            _one_line(run.request_text),
        ]
        print(" | ".join(parts))
    return 0


def run_show_command(*, state: MonicaState, run_id: str, json_output: bool = False) -> int:
    run = state.get_run(run_id)
    if run is None:
        print(f"Monica run not found: {run_id}")
        return 1
    if json_output:
        print(json.dumps(_run_payload(run, include_raw_event=True), sort_keys=True))
        return 0
    for key, value in run.__dict__.items():
        print(f"{key}: {value}")
    return 0


def _run_payload(run: Any, *, include_raw_event: bool) -> dict[str, Any]:
    payload = {
        "id": run.id,
        "platform": run.platform,
        "status": run.status,
        "intent": run.intent,
        "request_text": run.request_text,
        "slack": {
            "channel_id": run.channel_id,
            "thread_ts": run.thread_ts,
            "message_ts": run.message_ts,
            "user_id": run.user_id,
        },
        "linear": {
            "identifier": run.linear_identifier,
            "issue_id": run.linear_issue_id,
            "url": run.linear_url,
        },
        "branch_name": run.branch_name,
        "pr_url": run.pr_url,
        "failure_reason": run.failure_reason,
        "approved_by_user_id": run.approved_by_user_id,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }
    if include_raw_event:
        payload["raw_event"] = run.raw_event or {}
    return payload


def run_doctor_command(
    *,
    config: MonicaConfig,
    target_rollout_mode: str = "",
    json_output: bool = False,
    environ: dict[str, str] | None = None,
    which: Any | None = None,
    module_available: Any | None = None,
) -> int:
    effective_config = _config_for_doctor_target(
        config=config,
        target_rollout_mode=target_rollout_mode,
    )
    report = check_monica_readiness(
        config=effective_config,
        environ=environ,
        which=which,
        module_available=module_available,
    )

    if json_output:
        print(json.dumps(_readiness_report_payload(report), sort_keys=True))
        return 0 if report.ready else 1

    print(f"Monica rollout mode: {report.rollout_mode}")
    if report.runtime_root_value:
        print(f"Monica runtime root: {report.runtime_root_value}")
    for warning in report.warnings:
        print(f"WARN: {warning.message}")
    for failure in report.failures:
        print(f"FAIL: {failure.message}")
    if report.failures:
        print("Monica doctor: not ready")
        return 1
    print("Monica doctor: ready")
    return 0


def run_setup_plan_command(
    *,
    config: MonicaConfig,
    target_rollout_mode: str = "",
    json_output: bool = False,
    environ: dict[str, str] | None = None,
    which: Any | None = None,
    module_available: Any | None = None,
) -> int:
    effective_config = _config_for_doctor_target(
        config=config,
        target_rollout_mode=target_rollout_mode,
    )
    report = check_monica_readiness(
        config=effective_config,
        environ=environ,
        which=which,
        module_available=module_available,
    )
    steps = _setup_plan_steps(report)
    if json_output:
        payload = _readiness_report_payload(report)
        payload["steps"] = steps
        print(json.dumps(payload, sort_keys=True))
        return 0 if report.ready else 1

    print(f"Monica setup plan: {report.rollout_mode}")
    if report.ready:
        print("Monica is ready for this rollout mode.")
        if steps:
            print("")
            print("Recommended proof setup steps:")
            for index, step in enumerate(steps, start=1):
                print(f"{index}. {step['title']}")
                print(f"   {step['command']}")
                print(f"   {step['why']}")
        return 0
    print("")
    print("Blocking readiness failures:")
    for failure in report.failures:
        print(f"- {failure.message}")
    print("")
    print("Next steps:")
    for index, step in enumerate(steps, start=1):
        print(f"{index}. {step['title']}")
        print(f"   {step['command']}")
        print(f"   {step['why']}")
    return 1


def _setup_plan_steps(report: Any) -> list[dict[str, str]]:
    failure_codes = {str(issue.code) for issue in getattr(report, "failures", ())}
    warning_codes = {str(issue.code) for issue in getattr(report, "warnings", ())}
    rollout_mode = str(getattr(report, "rollout_mode", "") or "")
    steps: list[dict[str, str]] = []
    proof_warning_codes = {
        "proof_commands_empty",
        "ios_proof_tooling",
        "android_proof_tooling",
    }
    if not failure_codes and not (warning_codes & proof_warning_codes):
        return []
    if rollout_mode == "dry_run" and "config_enabled" in failure_codes:
        _append_setup_step(
            steps,
            step_id="configure_dry_run",
            title="Persist Monica's dry-run rollout settings",
            command="hermes mobile-bug-agent configure-dry-run",
            why="This enables the Monica plugin and keeps her in no-side-effect dry-run mode.",
        )
    if failure_codes & {
        "slack_bot_token",
        "slack_sdk",
        "slack_scope_app_mentions",
        "slack_scope_chat_write",
        "slack_scope_history",
        "slack_scope_files_read",
    }:
        _append_setup_step(
            steps,
            step_id="create_slack_app",
            title="Create or update the Monica Slack app",
            command="hermes mobile-bug-agent slack-manifest",
            why="Import the manifest in Slack, install the app, and invite Monica to the target channels.",
        )
    if "slack_bot_token" in failure_codes:
        _append_setup_step(
            steps,
            step_id="set_slack_bot_token",
            title="Store Monica's Slack bot token",
            command="Add MONICA_SLACK_BOT_TOKEN=xoxb-... to the Monica profile .env",
            why="Monica needs the bot token to read tagged threads and reply in Slack.",
        )
    if failure_codes & {
        "slack_bot_user_ids_empty",
        "slack_bot_id_value",
        "slack_bot_handle_value",
        "slack_bot_user_id_invalid",
        "slack_allowed_channels_empty",
        "slack_allowed_channel_invalid",
    }:
        _append_setup_step(
            steps,
            step_id="discover_slack_ids",
            title="Discover Monica's Slack user ID and channel IDs",
            command="hermes mobile-bug-agent slack-metadata --json",
            why="Use auth.bot_user_id for --bot-user-id and selected C.../G... channel IDs for --channel-id.",
        )
    if "linear_api_key" in failure_codes:
        _append_setup_step(
            steps,
            step_id="set_linear_api_key",
            title="Store the Linear API key",
            command="Add LINEAR_API_KEY=lin_api_... to ~/.hermes/.env",
            why="Monica needs Linear access to create or update mobile bug tickets.",
        )
    if failure_codes & {"linear_api_key", "linear_team_id"}:
        _append_setup_step(
            steps,
            step_id="discover_linear_ids",
            title="Discover Linear team, project, and label IDs",
            command="hermes mobile-bug-agent linear-metadata --json",
            why="Use the selected team ID, and optionally project/label IDs, in Monica's rollout config.",
        )
    if rollout_mode in {"linear_only", "local_fix_only", "approved_pr"} and failure_codes & {
        "config_enabled",
        "slack_bot_user_ids_empty",
        "slack_bot_id_value",
        "slack_bot_handle_value",
        "slack_bot_user_id_invalid",
        "slack_allowed_channels_empty",
        "slack_allowed_channel_invalid",
        "linear_team_id",
        "loop_create_linear",
    }:
        _append_setup_step(
            steps,
            step_id="configure_linear_only",
            title="Persist Monica's Linear-only rollout settings",
            command=(
                "hermes mobile-bug-agent configure-linear-only "
                "--bot-user-id <U...> --channel-id <C...> --linear-team-id <team-id>"
            ),
            why="This enables Monica only in explicitly allowed Slack channels and lets her file Linear issues.",
        )
    if rollout_mode == "local_fix_only" and failure_codes & {
        "repo_url",
        "verification_commands",
        "slack_approvers_empty",
        "slack_approver_handle_value",
        "slack_approver_invalid",
        "repo_local_name",
        "repo_branch_prefix",
        "repo_default_branch",
        "loop_require_fix_approval",
        "worker_session_prefix",
    }:
        _append_setup_step(
            steps,
            step_id="configure_local_fix_only",
            title="Persist local code-fix rollout settings",
            command=(
                "hermes mobile-bug-agent configure-local-fix-only "
                "--approver-user-id <U...> --repo-url <repo-url> "
                "--verification-command '<command>'"
            ),
            why="This configures the React Native repo, approver allowlist, and verification gate without enabling push or PR creation.",
        )
    if rollout_mode == "approved_pr" and failure_codes & {
        "repo_url",
        "verification_commands",
        "slack_approvers_empty",
        "slack_approver_handle_value",
        "slack_approver_invalid",
        "repo_local_name",
        "repo_branch_prefix",
        "repo_default_branch",
        "loop_require_fix_approval",
        "worker_session_prefix",
    }:
        _append_setup_step(
            steps,
            step_id="configure_approved_pr",
            title="Persist approved draft-PR rollout settings",
            command=(
                "hermes mobile-bug-agent configure-approved-pr "
                "--approver-user-id <U...> --repo-url <repo-url> "
                "--verification-command '<command>'"
            ),
            why="This configures the React Native repo, approver allowlist, and verification gate.",
        )
    if rollout_mode in {"local_fix_only", "approved_pr"} and failure_codes & {
        "git_executable",
        "codex_executable",
        "codex_approval_policy",
        "codex_sandbox",
    }:
        _append_setup_step(
            steps,
            step_id="prepare_code_tools",
            title="Prepare local code-fix tooling",
            command="git --version && codex --version",
            why="Code rollout modes need Git and a non-interactive Codex CLI worker.",
        )
    if rollout_mode == "approved_pr" and failure_codes & {
        "gh_executable",
        "github_token",
    }:
        _append_setup_step(
            steps,
            step_id="prepare_pr_tools",
            title="Prepare draft-PR tooling",
            command="gh auth status",
            why="Approved-PR mode needs GitHub CLI auth for pushing branches and opening draft PRs.",
        )
    if warning_codes & {"proof_commands_empty"}:
        _append_setup_step(
            steps,
            step_id="configure_simulator_proof",
            title="Configure Monica simulator proof command",
            command=(
                "Set mobile_bug_agent.proof.commands to "
                "`uv run --project \"$MONICA_HERMES_AGENT_ROOT\" python -m "
                "plugins.mobile_bug_agent.simulator_proof --timeout-seconds 600`"
            ),
            why="Stage D/E/F cannot complete until proof commands write screenshots or recordings to MONICA_PROOF_DIR.",
        )
    if warning_codes & {"ios_proof_tooling"}:
        _append_setup_step(
            steps,
            step_id="prepare_ios_simulator",
            title="Prepare iOS simulator proof tooling",
            command="xcrun --find simctl && xcodebuild -version",
            why=(
                "Install full Xcode, open it once, accept the license, select "
                "/Applications/Xcode.app/Contents/Developer, and install an iOS simulator runtime."
            ),
        )
    if warning_codes & {"android_proof_tooling"}:
        _append_setup_step(
            steps,
            step_id="prepare_android_emulator",
            title="Prepare Android emulator proof tooling",
            command="adb version && emulator -list-avds",
            why=(
                "Install Android SDK platform-tools and emulator, create an AVD, "
                "and make adb/emulator available on PATH."
            ),
        )
    _append_setup_step(
        steps,
        step_id="rerun_doctor",
        title="Re-run the rollout readiness check",
        command=f"hermes mobile-bug-agent doctor --rollout-mode {rollout_mode or 'linear_only'} --json",
        why="Only continue to a live Slack test once Monica reports ready.",
    )
    return steps


def _append_setup_step(
    steps: list[dict[str, str]],
    *,
    step_id: str,
    title: str,
    command: str,
    why: str,
) -> None:
    if any(step["id"] == step_id for step in steps):
        return
    steps.append(
        {
            "id": step_id,
            "title": title,
            "command": command,
            "why": why,
        }
    )


def _readiness_report_payload(report: Any) -> dict[str, Any]:
    return {
        "ready": report.ready,
        "rollout_mode": report.rollout_mode,
        "runtime_root": report.runtime_root_value,
        "warnings": [_readiness_issue_payload(issue) for issue in report.warnings],
        "failures": [_readiness_issue_payload(issue) for issue in report.failures],
    }


def _readiness_issue_payload(issue: Any) -> dict[str, str]:
    return {
        "code": str(issue.code),
        "message": str(issue.message),
    }


def _config_for_doctor_target(
    *,
    config: MonicaConfig,
    target_rollout_mode: str = "",
) -> MonicaConfig:
    target = str(target_rollout_mode or "").strip()
    if not target:
        return config
    if target not in {"dry_run", "linear_only", "local_fix_only", "approved_pr"}:
        return replace(config, rollout_mode=target)
    return replace(config, rollout_mode=target, dry_run=(target == "dry_run"))


def run_retry_command(
    *,
    state: MonicaState,
    run_id: str,
    config: MonicaConfig | None = None,
    readiness_checker: Any | None = None,
    json_output: bool = False,
) -> int:
    run = state.get_run(run_id)
    if run is None:
        message = f"Monica run not found: {run_id}"
        if json_output:
            _print_operator_error_json(action="retry", code="not_found", message=message)
            return 1
        print(message)
        return 1
    if run.status not in RETRYABLE_STATUSES:
        message = f"Monica run {run.id} is not retryable from status `{run.status}`."
        if json_output:
            _print_operator_error_json(
                action="retry",
                code="not_retryable",
                message=message,
                run=run,
            )
            return 1
        print(message)
        return 1
    if _is_cancelled_run(run):
        message = (
            f"Monica run {run.id} was cancelled from Slack and is not retryable from the CLI. "
            "Tag Monica again in the Slack thread with new context to start a fresh pass."
        )
        if json_output:
            _print_operator_error_json(
                action="retry",
                code="cancelled",
                message=message,
                run=run,
            )
            return 1
        print(message)
        return 1
    if str(run.pr_url or "").strip():
        message = (
            f"Monica run {run.id} already has a draft PR: {run.pr_url}. "
            "Open a new tagged Slack request for additional work instead of retrying this run."
        )
        if json_output:
            _print_operator_error_json(
                action="retry",
                code="pr_exists",
                message=message,
                run=run,
            )
            return 1
        print(message)
        return 1
    next_status = "approved" if run.approved_by_user_id else "queued"
    if (
        next_status == "approved"
        and config is not None
        and not _is_allowed_local_approver(config=config, user_id=run.approved_by_user_id)
    ):
        message = (
            f"Monica run {run.id} cannot retry from approval because stored approver "
            f"{run.approved_by_user_id} is not configured as a Monica code approver."
        )
        if json_output:
            _print_operator_error_json(
                action="retry",
                code="approver_not_allowed",
                message=message,
                run=run,
            )
            return 1
        print(message)
        return 1
    if next_status == "approved" and config is not None and config.rollout_mode in {"local_fix_only", "approved_pr"}:
        checker = readiness_checker or (lambda: run_doctor_command(config=config))
        try:
            readiness_exit_code = int(checker())
        except Exception as exc:
            message = f"Monica readiness check failed: {exc}"
            if json_output:
                _print_operator_error_json(
                    action="retry",
                    code="readiness_exception",
                    message=message,
                    run=run,
                )
                return 1
            print(message)
            return 1
        if readiness_exit_code != 0:
            message = "Monica readiness check failed; not retrying the approved Monica run."
            if json_output:
                _print_operator_error_json(
                    action="retry",
                    code="readiness_failed",
                    message=message,
                    run=run,
                )
                return 1
            print(message)
            return 1
    state.update_run(
        run.id,
        status=next_status,
        failure_reason="",
        branch_name="",
        pr_url="",
    )
    _run_loop(run.id, state=state)
    if json_output:
        _print_operator_success_json(action="retry", run=state.get_run(run.id) or run)
        return 0
    print(f"Retried Monica run {run.id}.")
    return 0


def run_approve_command(
    *,
    state: MonicaState,
    run_id: str,
    user_id: str,
    config: MonicaConfig | None = None,
    readiness_checker: Any | None = None,
    json_output: bool = False,
) -> int:
    run = state.get_run(run_id)
    if run is None:
        message = f"Monica run not found: {run_id}"
        if json_output:
            _print_operator_error_json(action="approve", code="not_found", message=message)
            return 1
        print(message)
        return 1
    if run.status != "awaiting_fix_approval":
        message = (
            f"Monica run {run.id} is not awaiting fix approval; current status is `{run.status}`."
        )
        if json_output:
            _print_operator_error_json(
                action="approve",
                code="not_awaiting_approval",
                message=message,
                run=run,
            )
            return 1
        print(message)
        return 1
    if config is not None and not _is_allowed_local_approver(config=config, user_id=user_id):
        message = f"{user_id} is not configured as a Monica code approver."
        if json_output:
            _print_operator_error_json(
                action="approve",
                code="approver_not_allowed",
                message=message,
                run=run,
            )
            return 1
        print(message)
        return 1
    if config is not None and config.rollout_mode in {"local_fix_only", "approved_pr"}:
        checker = readiness_checker or (lambda: run_doctor_command(config=config))
        try:
            readiness_exit_code = int(checker())
        except Exception as exc:
            message = f"Monica readiness check failed: {exc}"
            if json_output:
                _print_operator_error_json(
                    action="approve",
                    code="readiness_exception",
                    message=message,
                    run=run,
                )
                return 1
            print(message)
            return 1
        if readiness_exit_code != 0:
            message = "Monica readiness check failed; not approving the Monica run."
            if json_output:
                _print_operator_error_json(
                    action="approve",
                    code="readiness_failed",
                    message=message,
                    run=run,
                )
                return 1
            print(message)
            return 1
    approve_once = getattr(state, "approve_fix_once", None)
    if callable(approve_once):
        approved, changed = approve_once(run.id, approved_by_user_id=user_id)
    else:
        approved = state.approve_fix(run.id, approved_by_user_id=user_id)
        changed = True
    if not changed:
        message = f"Monica run {run.id} is already approved or no longer awaiting fix approval."
        if json_output:
            _print_operator_error_json(
                action="approve",
                code="already_approved_or_changed",
                message=message,
                run=state.get_run(run.id) or run,
            )
            return 1
        print(message)
        return 1
    _run_loop(approved.id, state=state)
    if json_output:
        _print_operator_success_json(action="approve", run=state.get_run(approved.id) or approved)
        return 0
    print(f"Approved Monica run {run.id}.")
    return 0


def run_simulate_command(
    *,
    config: MonicaConfig,
    state: MonicaState,
    text: str,
    channel_id: str,
    user_id: str,
    thread_ts: str = "",
    allow_side_effects: bool = False,
    json_output: bool = False,
    loop_runner: Any | None = None,
    readiness_checker: Any | None = None,
) -> int:
    if not config.enabled:
        if json_output:
            _print_simulation_error_json(
                code="config_disabled",
                message="Monica is disabled: set mobile_bug_agent.enabled: true before simulating.",
                config=config,
                allow_side_effects=allow_side_effects,
            )
            return 1
        print("Monica is disabled: set mobile_bug_agent.enabled: true before simulating.")
        return 1
    clean_text = " ".join(str(text or "").split())
    if not clean_text:
        if json_output:
            _print_simulation_error_json(
                code="empty_text",
                message="Simulation text is required.",
                config=config,
                allow_side_effects=allow_side_effects,
            )
            return 2
        print("Simulation text is required.")
        return 2
    if config.rollout_mode != "dry_run" and not allow_side_effects:
        message = (
            "Simulation is in a side-effect rollout mode "
            f"(`{config.rollout_mode}`). Re-run with --allow-side-effects only if you "
            "intend to create/update Linear issues or start a code-fix loop."
        )
        if json_output:
            _print_simulation_error_json(
                code="side_effects_not_allowed",
                message=message,
                config=config,
                allow_side_effects=allow_side_effects,
            )
            return 1
        print(message)
        return 1
    if config.rollout_mode in {"linear_only", "local_fix_only", "approved_pr"} and allow_side_effects:
        allowed_channels = set(config.slack.allowed_channels)
        if not allowed_channels:
            message = (
                "Simulation side effects require mobile_bug_agent.slack.allowed_channels. "
                "Configure the exact Slack channel IDs Monica may act in."
            )
            if json_output:
                _print_simulation_error_json(
                    code="allowed_channels_empty",
                    message=message,
                    config=config,
                    allow_side_effects=allow_side_effects,
                )
                return 1
            print(message)
            return 1
        if not config.slack.bot_user_ids:
            message = (
                "Simulation side effects require mobile_bug_agent.slack.bot_user_ids. "
                "Configure Monica's Slack mention user ID before exercising real side effects."
            )
            if json_output:
                _print_simulation_error_json(
                    code="bot_user_ids_empty",
                    message=message,
                    config=config,
                    allow_side_effects=allow_side_effects,
                )
                return 1
            print(message)
            return 1
        if channel_id not in allowed_channels:
            message = (
                f"Simulation channel {channel_id} is not in "
                "mobile_bug_agent.slack.allowed_channels."
            )
            if json_output:
                _print_simulation_error_json(
                    code="channel_not_allowed",
                    message=message,
                    config=config,
                    allow_side_effects=allow_side_effects,
                )
                return 1
            print(message)
            return 1
        if config.rollout_mode in {"local_fix_only", "approved_pr"} and not config.slack.approver_user_ids:
            message = (
                "Simulation code-fix side effects require "
                "mobile_bug_agent.slack.approver_user_ids. Configure at least one "
                "Monica code approver."
            )
            if json_output:
                _print_simulation_error_json(
                    code="approver_user_ids_empty",
                    message=message,
                    config=config,
                    allow_side_effects=allow_side_effects,
                )
                return 1
            print(message)
            return 1
        checker = readiness_checker or (lambda: run_doctor_command(config=config))
        try:
            readiness_exit_code = int(checker())
        except Exception as exc:
            message = f"Monica readiness check failed: {exc}"
            if json_output:
                _print_simulation_error_json(
                    code="readiness_exception",
                    message=message,
                    config=config,
                    allow_side_effects=allow_side_effects,
                )
                return 1
            print(message)
            return 1
        if readiness_exit_code != 0:
            if json_output:
                _print_simulation_error_json(
                    code="readiness_failed",
                    message="Monica readiness check failed; not creating a simulated side-effect run.",
                    config=config,
                    allow_side_effects=allow_side_effects,
                )
                return 1
            print("Monica readiness check failed; not creating a simulated side-effect run.")
            return 1

    now_value = time.time()
    now = f"{now_value:.6f}"
    thread = thread_ts.strip() or _next_simulated_thread_ts(now_value)
    run, created = state.create_run_once(
        platform="slack",
        channel_id=channel_id,
        thread_ts=thread,
        message_ts=now,
        user_id=user_id,
        request_text=clean_text,
        raw_event={
            "monica_simulated": True,
            "type": "app_mention",
            "text": f"<@MONICA_SIM> {clean_text}",
            "channel": channel_id,
            "user": user_id,
            "thread_ts": thread,
            "ts": now,
            "permalink": f"local://monica/simulated/{thread}",
            "files": [],
        },
    )
    if not created:
        if json_output:
            print(
                json.dumps(
                    {
                        "created": False,
                        "rollout_mode": config.rollout_mode,
                        "allow_side_effects": allow_side_effects,
                        "error": {
                            "code": "duplicate_thread",
                            "message": (
                                f"Monica simulation already exists for channel `{channel_id}` "
                                f"thread `{thread}`."
                            ),
                        },
                        "run": _run_payload(run, include_raw_event=True),
                    },
                    sort_keys=True,
                )
            )
            return 1
        print(
            f"Monica simulation already exists for channel `{channel_id}` thread `{thread}`: {run.id}. "
            "Use a different --thread-ts to start a new local simulation."
        )
        return 1
    runner = loop_runner or _run_loop
    runner(run.id, state=state)
    updated = state.get_run(run.id) or run
    if json_output:
        print(
            json.dumps(
                {
                    "created": True,
                    "rollout_mode": config.rollout_mode,
                    "allow_side_effects": allow_side_effects,
                    "run": _run_payload(updated, include_raw_event=True),
                },
                sort_keys=True,
            )
        )
        return 0
    print(f"Simulated Monica run {run.id}: {updated.status}")
    if updated.linear_identifier:
        print(f"Linear: {updated.linear_identifier}")
    if updated.linear_url:
        print(f"Linear URL: {updated.linear_url}")
    if updated.pr_url:
        print(f"PR: {updated.pr_url}")
    if updated.failure_reason:
        print(f"Reason: {updated.failure_reason}")
    return 0


def _print_simulation_error_json(
    *,
    code: str,
    message: str,
    config: MonicaConfig,
    allow_side_effects: bool,
) -> None:
    print(
        json.dumps(
            {
                "created": False,
                "rollout_mode": config.rollout_mode,
                "allow_side_effects": allow_side_effects,
                "error": {
                    "code": code,
                    "message": message,
                },
            },
            sort_keys=True,
        )
    )


def _print_operator_success_json(*, action: str, run: Any) -> None:
    print(
        json.dumps(
                {
                    "ok": True,
                    "action": action,
                    "run": _run_payload(run, include_raw_event=False),
                },
                sort_keys=True,
            )
    )


def _print_operator_error_json(
    *,
    action: str,
    code: str,
    message: str,
    run: Any | None = None,
) -> None:
    payload: dict[str, Any] = {
        "ok": False,
        "action": action,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if run is not None:
        payload["run"] = _run_payload(run, include_raw_event=False)
    print(json.dumps(payload, sort_keys=True))


def _run_loop(run_id: str, *, state: MonicaState) -> None:
    config = load_monica_config()
    skills = DefaultMonicaSkills(config=config, state=state)
    MonicaLoop(config=config, state=state, skills=skills).run(run_id)


def _open_state(config: MonicaConfig | None = None) -> MonicaState:
    return MonicaState.open(runtime_root(config or load_monica_config()) / "state.sqlite")


def _one_line(value: Any, limit: int = 100) -> str:
    return " ".join(str(value).split())[:limit]


def _next_simulated_thread_ts(now: float) -> str:
    global _SIMULATION_THREAD_SEQUENCE
    seconds = int(now)
    micros = int((now - seconds) * 1_000_000)
    _SIMULATION_THREAD_SEQUENCE = (_SIMULATION_THREAD_SEQUENCE + 1) % 1_000_000
    micros = (micros + _SIMULATION_THREAD_SEQUENCE) % 1_000_000
    return f"{seconds}.{micros:06d}"


def _is_allowed_local_approver(*, config: MonicaConfig, user_id: str) -> bool:
    return str(user_id or "").strip() in set(config.slack.approver_user_ids)


def _is_cancelled_run(run: Any) -> bool:
    return str(getattr(run, "failure_reason", "") or "").startswith("cancelled by ")
