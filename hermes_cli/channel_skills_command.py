"""``hermes channel-skills`` command implementation."""

from __future__ import annotations

import json
import sys
from typing import Any

from gateway.channel_skill_admin import ChannelSkillAdmin
from gateway.channel_skill_generation import ChannelSkillRegistry


def _registry() -> ChannelSkillRegistry:
    return ChannelSkillRegistry()


def _config_extra() -> dict[str, Any]:
    from hermes_cli.config import load_config
    return load_config().get("discord", {})


def channel_skills_command(args: Any) -> int:
    action = getattr(args, "action", None) or "list"
    if getattr(args, "root_json", False):
        args.json = True
    try:
        admin = ChannelSkillAdmin(_registry(), config_extra=_config_extra())
        if action == "list":
            result = admin.list(
                status=getattr(args, "status", None),
                include_static=getattr(args, "all", False),
                source=getattr(args, "source", None),
                effective=getattr(args, "effective", False),
            )
            if getattr(args, "json", False):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_list(result)
            return 0
        if action == "effective":
            result = admin.effective(args.channel_id, getattr(args, "parent_id", None))
            print(json.dumps(result, ensure_ascii=False, indent=2) if getattr(args, "json", False) else _format_effective(result))
            return 0
        if action == "validate":
            result = admin.validate()
            print(json.dumps(result, ensure_ascii=False, indent=2) if getattr(args, "json", False) else _format_validation(result))
            return 0 if result["valid"] else 1
        if action in {"show", "inspect"}:
            result = admin.inspect(args.channel_id) if action == "inspect" else admin.show(args.channel_id)
            if getattr(args, "json", False) or action == "inspect":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_show(result)
            return 0
        if action == "remove":
            if getattr(args, "delete_skill", False) and not getattr(args, "yes", False):
                print("Refusing --delete-skill without --yes.", file=sys.stderr)
                return 2
            result = admin.remove(args.channel_id, delete_skill=getattr(args, "delete_skill", False))
            print(json.dumps(result, ensure_ascii=False) if getattr(args, "json", False) else "Channel skill removed.")
            return 0
        if action == "regenerate":
            result = admin.regenerate(
                args.channel_id,
                force=getattr(args, "force", False),
                dry_run=getattr(args, "dry_run", False),
                backup=getattr(args, "backup", False),
            )
            if getattr(args, "json", False) or getattr(args, "dry_run", False):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("Channel skill regenerated.")
            return 0
        if action == "repair":
            result = admin.repair(apply=getattr(args, "apply", False))
            print(json.dumps(result, ensure_ascii=False, indent=2) if getattr(args, "json", False) else _format_repair(result))
            return 0 if not result["issues"] else 1
        if action in {"bind", "unbind", "policy"}:
            return _config_command(args)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Unknown channel-skills action: {action}", file=sys.stderr)
    return 2


def _config_command(args: Any) -> int:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    discord = config.setdefault("discord", {})
    action = args.action
    if action == "bind":
        bindings = [item for item in discord.get("channel_skill_bindings", [])
                    if isinstance(item, dict) and str(item.get("id")) != str(args.channel_id)]
        binding = {"id": str(args.channel_id)}
        if getattr(args, "profile", None):
            binding["profile"] = args.profile
        else:
            binding["skills"] = list(args.skill)
        bindings.append(binding)
        discord["channel_skill_bindings"] = bindings
        save_config(config)
        print(f"Static binding saved for channel {args.channel_id}.")
        return 0
    if action == "unbind":
        before = discord.get("channel_skill_bindings", [])
        discord["channel_skill_bindings"] = [
            item for item in before
            if not (isinstance(item, dict) and str(item.get("id")) == str(args.channel_id))
        ]
        save_config(config)
        print(f"Static binding removed for channel {args.channel_id}.")
        return 0
    policy = discord.setdefault("channel_skill_auto_generate", {})
    if args.policy_action == "show":
        print(json.dumps(policy, ensure_ascii=False, indent=2) if getattr(args, "json", False) else
              f"enabled: {bool(policy.get('enabled', False))}")
        return 0
    policy["enabled"] = args.policy_action == "enable"
    save_config(config)
    print(f"Automatic channel skill generation {'enabled' if policy['enabled'] else 'disabled'}.")
    return 0


def _print_list(items: list[dict[str, Any]]) -> None:
    if not items:
        print("No channel skills found.")
        return
    print(f"{'CHANNEL ID':<24} {'PLATFORM':<10} {'STATUS':<16} PROFILE")
    for item in items:
        print(f"{item['channel_id']:<24} {str(item.get('platform') or '-'):<10} "
              f"{item['status']:<16} {item.get('profile') or '-'}")


def _print_show(item: dict[str, Any]) -> None:
    record = item["record"]
    file_info = item["file"]
    print(f"Channel:       {item['channel_id']}")
    print(f"Platform:      {item.get('platform') or '-'}")
    print(f"Status:        {item['status']}")
    print(f"Profile:       {item.get('profile') or '-'}")
    print(f"Skill file:    {file_info.get('path') or '-'}")
    print(f"Exists:        {'yes' if file_info.get('exists') else 'no'}")
    print(f"Frontmatter:   {file_info.get('frontmatter')}")
    print(f"Name:          {file_info.get('name')}")
    print(f"Channel ID:    {file_info.get('channel_id')}")
    if record.get("resolved_skills"):
        print("Resolved skills:")
        for skill in record["resolved_skills"]:
            print(f"  - {skill}")




def _format_effective(result: dict[str, Any]) -> str:
    lines = [f"Channel: {result['channel_id']}", f"Source: {result['source']}", "Effective skills:"]
    lines.extend(f"  - {skill}" for skill in result.get("effective_skills", []))
    if result.get("profile_chain"):
        lines.append("Profile chain: " + " -> ".join(result["profile_chain"]))
    if result.get("inherited_from_parent"):
        lines.append("Inherited from parent: yes")
    return "\n".join(lines)


def _format_validation(result: dict[str, Any]) -> str:
    if result["valid"]:
        return "Channel skill configuration is valid."
    lines = ["Channel skill validation issues:"]
    for issue in result["issues"]:
        detail = issue.get("skill") or issue.get("message") or issue["type"]
        lines.append(f"  - {issue['channel_id']}: {issue['type']} ({detail})")
    return "\n".join(lines)


def _format_repair(result: dict[str, Any]) -> str:
    if not result["issues"]:
        return "No repair issues found."
    lines = ["Repair issues:"]
    for issue in result["issues"]:
        lines.append(f"  - {issue['channel_id']}: {issue['status']} ({issue['action']})")
    return "\n".join(lines)
