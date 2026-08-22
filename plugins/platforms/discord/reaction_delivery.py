"""Discord reaction manifest delivery helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from agent.secret_scope import get_secret
from gateway.platforms.base import BasePlatformAdapter

try:
    from .reaction_requests import ReactionError, add_reaction_request
except ImportError:
    from reaction_requests import ReactionError, add_reaction_request  # type: ignore

try:
    from .reaction_manifest import (
        choose_reaction_message_index,
        extract_reaction_manifest,
        manifest_actions,
        manifest_discord_messages,
    )
except ImportError:
    from reaction_manifest import (  # type: ignore
        choose_reaction_message_index,
        extract_reaction_manifest,
        manifest_actions,
        manifest_discord_messages,
    )


StandaloneSender = Callable[..., Awaitable[dict[str, Any]]]
CaptionSplit = Callable[..., tuple[str | None, str]]


async def add_manifest_reaction_with_retry(
    pconfig: Any,
    chat_id: str,
    message_id: str,
    emoji: str,
    *,
    max_attempts: int = 3,
) -> str | None:
    """Best-effort REST reaction add for out-of-process Discord sends."""
    try:
        import aiohttp
    except ImportError:
        return "Discord reaction add skipped: aiohttp not installed"

    token = (getattr(pconfig, "token", None) or "").strip()
    if not token:
        token = (get_secret("DISCORD_BOT_TOKEN", "") or "").strip()
    if not token:
        return "Discord reaction add skipped: DISCORD_BOT_TOKEN is not set"

    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

        proxy_url = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
        session_kwargs, request_kwargs = proxy_kwargs_for_aiohttp(proxy_url)
    except Exception:
        session_kwargs, request_kwargs = {}, {}

    try:
        request = add_reaction_request(chat_id, message_id, emoji)
    except ReactionError as exc:
        return f"Discord reaction add skipped for message {message_id} emoji {emoji}: {exc}"
    url = f"https://discord.com/api/v10{request['path']}"
    headers = {"Authorization": f"Bot {token}"}

    for attempt in range(max_attempts):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                **session_kwargs,
            ) as session:
                async with session.put(url, headers=headers, **request_kwargs) as resp:
                    if resp.status in {200, 204}:
                        return None
                    if resp.status == 429 and attempt < max_attempts - 1:
                        retry_after = None
                        try:
                            body = await resp.json()
                            retry_after = float(body.get("retry_after"))
                        except Exception:
                            for header in ("Retry-After", "X-RateLimit-Reset-After"):
                                try:
                                    retry_after = float(resp.headers.get(header))
                                    break
                                except Exception:
                                    continue
                        if retry_after is None:
                            retry_after = 1.0
                        await asyncio.sleep(max(0.0, min(retry_after, 10.0)))
                        continue
                    try:
                        body = await resp.text()
                    except Exception:
                        body = ""
                    detail = f": {body[:200]}" if body else ""
                    return (
                        "Discord reaction add failed "
                        f"for message {message_id} emoji {emoji} ({resp.status}){detail}"
                    )
        except Exception as exc:
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.5)
                continue
            return (
                "Discord reaction add failed "
                f"for message {message_id} emoji {emoji}: {type(exc).__name__}: {exc}"
            )
    return f"Discord reaction add failed for message {message_id} emoji {emoji}"


async def attach_manifest_actions_standalone(
    pconfig: Any,
    chat_id: str,
    actions: list[dict[str, Any]],
    sent_messages: list[dict[str, Any]],
) -> list[str]:
    warnings = []
    for action in actions:
        index = choose_reaction_message_index(action, sent_messages)
        if index is None:
            continue
        sent = sent_messages[index]
        message_id = sent.get("message_id")
        emoji = action.get("emoji")
        if not message_id or not isinstance(emoji, str) or not emoji.strip():
            continue
        warning = await add_manifest_reaction_with_retry(
            pconfig,
            chat_id,
            str(message_id),
            emoji,
        )
        if warning:
            warnings.append(warning)
    return warnings


async def attach_manifest_actions_live(
    adapter: Any,
    actions: list[dict[str, Any]],
    sent_messages: list[dict[str, Any]],
) -> list[str]:
    """Best-effort outbound reaction shortcuts for sent Discord messages."""
    warnings: list[str] = []
    for action in actions:
        index = choose_reaction_message_index(action, sent_messages)
        if index is None:
            continue
        message = sent_messages[index].get("message")
        emoji = action.get("emoji")
        if not message or not isinstance(emoji, str) or not emoji.strip():
            continue
        if not await adapter._add_reaction(message, emoji):
            message_id = getattr(message, "id", sent_messages[index].get("message_id", "unknown"))
            warnings.append(f"Failed to add Discord reaction {emoji} to message {message_id}")
    return warnings


def merge_warnings(result: Any, warnings: list[str]) -> Any:
    if warnings and isinstance(result, dict):
        existing = result.get("warnings")
        if isinstance(existing, list):
            result["warnings"] = [*existing, *warnings]
        elif existing:
            result["warnings"] = [existing, *warnings]
        else:
            result["warnings"] = warnings
    return result


async def send_standalone_discord_with_manifest(
    *,
    pconfig: Any,
    chat_id: str,
    message: str,
    thread_id: str | None,
    media_files: list[tuple[str, bool]],
    max_len: int | None,
    default_caption_limit: int,
    caption_split: CaptionSplit,
    standalone_sender_fn: StandaloneSender,
) -> dict[str, Any]:
    """Send Discord text/media through the standalone sender with manifests."""
    send_content, discord_manifest = extract_reaction_manifest(message)
    discord_manifest_messages = manifest_discord_messages(discord_manifest)

    if discord_manifest_messages and not media_files:
        last_result = None
        sent_messages = []
        warnings = []
        for item in discord_manifest_messages:
            item_chunks = BasePlatformAdapter.truncate_message(
                item["content"],
                max_len or default_caption_limit,
            )
            item_sent = []
            for chunk in item_chunks:
                if not chunk.strip():
                    continue
                result = await standalone_sender_fn(
                    pconfig,
                    chat_id,
                    chunk,
                    thread_id=thread_id,
                    media_files=[],
                )
                if isinstance(result, dict) and result.get("error"):
                    return result
                last_result = result
                message_id = result.get("message_id") if isinstance(result, dict) else None
                if message_id:
                    record = {"content": chunk, "message_id": message_id}
                    item_sent.append(record)
                    sent_messages.append(record)
            warnings.extend(
                await attach_manifest_actions_standalone(
                    pconfig,
                    chat_id,
                    item.get("actions", []),
                    item_sent,
                )
            )
        warnings.extend(
            await attach_manifest_actions_standalone(
                pconfig,
                chat_id,
                manifest_actions(discord_manifest),
                sent_messages,
            )
        )
        if last_result is None:
            last_result = {"error": "Discord reaction manifest did not contain deliverable text"}
        return merge_warnings(last_result, warnings)

    caption, _ = caption_split(
        send_content,
        media_files,
        max_caption_len=(max_len or default_caption_limit),
    )
    if caption is not None:
        result = await standalone_sender_fn(
            pconfig,
            chat_id,
            "",
            thread_id=thread_id,
            media_files=media_files,
            caption=caption,
        )
        if isinstance(result, dict) and result.get("error"):
            return result
        sent_messages = []
        message_id = result.get("message_id") if isinstance(result, dict) else None
        if message_id:
            sent_messages.append({"content": caption, "message_id": message_id})
        warnings = await attach_manifest_actions_standalone(
            pconfig,
            chat_id,
            manifest_actions(discord_manifest),
            sent_messages,
        )
        return merge_warnings(result, warnings)

    chunks = BasePlatformAdapter.truncate_message(send_content, max_len) if max_len else [send_content]
    last_result = None
    sent_messages = []
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        result = await standalone_sender_fn(
            pconfig,
            chat_id,
            chunk,
            thread_id=thread_id,
            media_files=media_files if is_last else [],
        )
        if isinstance(result, dict) and result.get("error"):
            return result
        last_result = result
        message_id = result.get("message_id") if isinstance(result, dict) else None
        if message_id:
            sent_messages.append({"content": chunk, "message_id": message_id})
    warnings = await attach_manifest_actions_standalone(
        pconfig,
        chat_id,
        manifest_actions(discord_manifest),
        sent_messages,
    )
    return merge_warnings(last_result, warnings)
