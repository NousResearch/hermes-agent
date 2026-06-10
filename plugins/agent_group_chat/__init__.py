"""
Agent Group Chat plugin for Hermes.

Routes @profile mentions in shared gateway chats to named Hermes profiles and
posts their replies back into the same platform room. The transport is the
Hermes CLI itself, so the feature is agent-harness-agnostic: any profile/harness
that can be invoked through `hermes -p <profile> -z <prompt>` can participate.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MENTION_RE = re.compile(r"(^|\s)@([A-Za-z][A-Za-z0-9_.-]{1,63})\b")
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_REPLY_CHARS = 12000
DEFAULT_MAX_CONCURRENT = 4
_SHARED_CHAT_TYPES = {"group", "supergroup", "channel", "thread", "forum", "topic"}

_ACTIVE_TASKS: set[asyncio.Task] = set()
_SEMAPHORE: Optional[asyncio.Semaphore] = None


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hermes_root() -> Path:
    return Path(os.environ.get("HERMES_ROOT") or Path.home() / ".hermes").expanduser()


def _current_profile_name() -> str:
    env_profile = os.environ.get("HERMES_PROFILE") or os.environ.get("HERMES_PROFILE_NAME")
    if env_profile:
        return env_profile.strip().lower() or "default"
    home = Path(os.environ.get("HERMES_HOME") or _hermes_root()).expanduser().resolve()
    try:
        root = _hermes_root().resolve()
        profiles_dir = root / "profiles"
        if home.parent == profiles_dir:
            return home.name.lower()
    except Exception:
        pass
    return "default"


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        logger.debug("agent_group_chat: failed to load config: %s", exc)
        return {}


def _plugin_config() -> Dict[str, Any]:
    cfg = _load_config()
    agc = cfg.get("agent_group_chat")
    if isinstance(agc, dict):
        return agc
    gateway = cfg.get("gateway")
    if isinstance(gateway, dict) and isinstance(gateway.get("agent_group_chat"), dict):
        return gateway["agent_group_chat"]
    return {}


def _read_profile_description(profile_dir: Path) -> str:
    path = profile_dir / "profile.yaml"
    if not path.exists():
        return ""
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return str(data.get("description") or "").strip()
    except Exception:
        return ""
    return ""


def _coerce_aliases(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def _profile_aliases(name: str, cfg: Dict[str, Any]) -> List[str]:
    aliases = {name.lower()}
    raw_agents = cfg.get("agents")
    if isinstance(raw_agents, dict):
        entry = raw_agents.get(name) or raw_agents.get(name.lower())
        if isinstance(entry, dict):
            for alias in _coerce_aliases(entry.get("aliases")):
                alias = str(alias).strip().lower().lstrip("@")
                if alias:
                    aliases.add(alias)
    return sorted(aliases)


def _discover_agents(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return alias -> agent mapping for all configured/current profiles."""
    root = _hermes_root()
    profiles_dir = root / "profiles"
    current = _current_profile_name()
    agents: Dict[str, Dict[str, Any]] = {}

    configured = cfg.get("agents")
    configured_names: set[str] = set()
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(value, dict) and value.get("enabled") is False:
                continue
            configured_names.add(str(key).strip().lower())

    names: set[str] = set(configured_names)
    if profiles_dir.exists():
        for child in profiles_dir.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                names.add(child.name.lower())

    include_default = _truthy(cfg.get("include_default"), False)
    if include_default:
        names.add("default")

    explicit_only = _truthy(cfg.get("explicit_agents_only"), False)
    if explicit_only:
        names = configured_names

    for name in sorted(names):
        if not name or name == current:
            continue
        if name == "default":
            profile_dir = root
        else:
            profile_dir = profiles_dir / name
            if not profile_dir.exists():
                continue
        entry = configured.get(name, {}) if isinstance(configured, dict) else {}
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            entry = {}
        display_name = str(entry.get("name") or name).strip() or name
        agent = {
            "profile": name,
            "name": display_name,
            "description": str(entry.get("description") or _read_profile_description(profile_dir)),
            "profile_dir": str(profile_dir),
        }
        for alias in _profile_aliases(name, cfg):
            agents[alias] = agent
        # Display-name alias too, for @HankCodingAgentBot-style or cased names.
        display_alias = display_name.lower().lstrip("@")
        if display_alias:
            agents[display_alias] = agent
    return agents


def _extract_mentions(text: str) -> List[str]:
    seen = set()
    out = []
    for match in MENTION_RE.finditer(text or ""):
        mention = match.group(2).strip().lower()
        if mention and mention not in seen:
            seen.add(mention)
            out.append(mention)
    return out


def _source_dict(source: Any) -> Dict[str, Any]:
    if hasattr(source, "to_dict"):
        try:
            return source.to_dict()
        except Exception:
            pass
    out = {}
    for key in (
        "platform", "chat_id", "chat_name", "chat_type", "user_id", "user_name",
        "thread_id", "chat_topic", "guild_id", "parent_chat_id", "message_id",
    ):
        value = getattr(source, key, None)
        if value is not None:
            out[key] = getattr(value, "value", value)
    return out


def _build_prompt(agent: Dict[str, Any], event: Any, cfg: Dict[str, Any], mentions: List[str]) -> str:
    source = getattr(event, "source", None)
    source_data = _source_dict(source)
    sender = source_data.get("user_name") or source_data.get("user_id") or "someone in the group"
    room = source_data.get("chat_name") or source_data.get("chat_id") or "the group chat"
    text = event.text or ""
    addressed = ", ".join(f"@{m}" for m in mentions)
    description = agent.get("description") or ""
    desc_block = f"\nYour profile description: {description}" if description else ""
    return (
        "You were mentioned in a shared multi-agent group chat. Reply as yourself, "
        "not as the routing system. Keep the reply useful and addressed to the room. "
        "If you need another agent, mention them with @Name in your reply.\n\n"
        f"Your agent/profile name: {agent['name']} ({agent['profile']}).{desc_block}\n"
        f"Mention(s) detected: {addressed}\n"
        f"Room: {room}\n"
        f"Sender: {sender}\n"
        f"Source metadata: {json.dumps(source_data, ensure_ascii=False)}\n\n"
        "Original group message:\n"
        f"{text}"
    )


def _hermes_command(profile: str, prompt: str, cfg: Dict[str, Any]) -> List[str]:
    configured = cfg.get("hermes_command")
    if isinstance(configured, str) and configured.strip():
        base = shlex.split(configured)
    elif isinstance(configured, list) and configured:
        base = [str(x) for x in configured]
    else:
        hermes_bin = shutil.which("hermes")
        base = [hermes_bin] if hermes_bin else [sys.executable, "-m", "hermes_cli.main"]

    cmd = [*base]
    if profile and profile != "default":
        cmd.extend(["-p", profile])
    cmd.extend(["-z", prompt])
    for extra in cfg.get("extra_args") or []:
        cmd.append(str(extra))
    return cmd


async def _run_profile(agent: Dict[str, Any], event: Any, gateway: Any, cfg: Dict[str, Any], mentions: List[str]) -> None:
    global _SEMAPHORE
    max_concurrent = int(cfg.get("max_concurrent", DEFAULT_MAX_CONCURRENT) or DEFAULT_MAX_CONCURRENT)
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(max(1, max_concurrent))

    async with _SEMAPHORE:
        source = event.source
        adapter = gateway.adapters.get(source.platform)
        if adapter is None:
            logger.warning("agent_group_chat: no adapter for platform %s", source.platform)
            return

        timeout = int(cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS) or DEFAULT_TIMEOUT_SECONDS)
        max_reply_chars = int(cfg.get("max_reply_chars", DEFAULT_MAX_REPLY_CHARS) or DEFAULT_MAX_REPLY_CHARS)
        prompt = _build_prompt(agent, event, cfg, mentions)
        cmd = _hermes_command(agent["profile"], prompt, cfg)
        env = os.environ.copy()
        env.setdefault("HERMES_PROFILE", agent["profile"])
        env.setdefault("HERMES_PROFILE_NAME", agent["profile"])
        env["HERMES_AGENT_GROUP_CHAT_INVOCATION"] = "1"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path.cwd()),
                env=env,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            reply = f"@{agent['name']} timed out after {timeout}s."
        except Exception as exc:
            logger.exception("agent_group_chat: failed invoking %s", agent["profile"])
            reply = f"@{agent['name']} failed to start: {exc}"
        else:
            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                detail = stderr or stdout or f"exit code {proc.returncode}"
                reply = f"@{agent['name']} failed: {detail[-1000:]}"
            else:
                reply = stdout or f"@{agent['name']} returned an empty reply."

        if len(reply) > max_reply_chars:
            reply = reply[:max_reply_chars].rstrip() + "\n… [truncated]"

        prefix = cfg.get("reply_prefix")
        if prefix is None:
            prefix = f"**{agent['name']}**: "
        elif isinstance(prefix, str):
            prefix = prefix.format(agent=agent["name"], profile=agent["profile"])
        content = f"{prefix}{reply}" if prefix else reply

        metadata = {}
        thread_id = getattr(source, "thread_id", None)
        if thread_id:
            metadata["thread_id"] = thread_id
        message_id = getattr(event, "message_id", None) or getattr(source, "message_id", None)
        if message_id:
            metadata["reply_to_message_id"] = message_id

        try:
            await adapter.send(source.chat_id, content, metadata=metadata)
        except TypeError:
            await adapter.send(source.chat_id, content)
        except Exception:
            logger.exception("agent_group_chat: failed delivering reply from %s", agent["profile"])


def _schedule(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _ACTIVE_TASKS.add(task)
    task.add_done_callback(_ACTIVE_TASKS.discard)


def pre_gateway_dispatch(event: Any, gateway: Any, session_store: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
    cfg = _plugin_config()
    if not _truthy(cfg.get("enabled"), True):
        return None
    if os.environ.get("HERMES_AGENT_GROUP_CHAT_INVOCATION") == "1":
        return None
    if getattr(event, "internal", False):
        return None
    text = event.text or ""
    if not text or "@" not in text:
        return None
    if text.lstrip().startswith("/"):
        return None
    source = getattr(event, "source", None)
    if source is None:
        return None
    if getattr(source, "is_bot", False):
        return None
    chat_type = str(getattr(source, "chat_type", "") or "").lower()
    shared_only = _truthy(cfg.get("shared_chats_only"), True)
    if shared_only and chat_type not in _SHARED_CHAT_TYPES:
        return None

    agents_by_alias = _discover_agents(cfg)
    mentions = _extract_mentions(text)
    targets: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
    for mention in mentions:
        agent = agents_by_alias.get(mention)
        if not agent:
            continue
        profile = agent["profile"]
        bucket = targets.setdefault(profile, (agent, []))[1]
        bucket.append(mention)

    if not targets:
        return None

    for agent, matched_mentions in targets.values():
        _schedule(_run_profile(agent, event, gateway, cfg, matched_mentions))

    # If a group message only mentions other agents, do not also run the current
    # gateway profile. This is the important "@Hank replies, Vera doesn't barge
    # in" behavior. Set also_run_current=true to disable the skip.
    if not _truthy(cfg.get("also_run_current"), False):
        return {"action": "skip", "reason": "agent_group_chat_routed"}
    return {"action": "allow"}
