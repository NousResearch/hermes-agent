"""Uniform Channel Capabilities block for the gateway session-context prompt.

Issue: NousResearch/hermes-agent#45122 (Phase 1).

The session-context prompt (``gateway.session.build_session_context_prompt``)
already injects a ``**Source:**`` line, ``**Channel Topic:**``, Matrix extras,
``_PLATFORM_NOTES`` (behavioral notes from agent/prompt_builder.py PLATFORM_HINTS
+ gateway/session.py _STATIC_PLATFORM_NOTES), and the connected-platforms /
home-channels / delivery-options blocks.  What it does NOT yet inject is
uniform, structured per-channel **behavioral** guidance the model can act on:

  * interaction mode  (sync / async / autonomous)
  * response style   (conversational, brief / document-style allowed / ...)
  * length / format limits
  * media-delivery summary
  * capability actions the platform supports

This module owns that block.  The rendered output is keyed by
``src.platform.value`` (and optionally ``src.chat_type``); both already
appear in ``gateway.run_agent_cache._ephemeral_change_key`` (the inputs the
prompt renderer is pinned on), so the block is automatically pin-safe — a
hit on the change key returns the verbatim bytes; a miss re-renders once.
NO env var, config flag, adapter state, or time is read; the renderer is
a pure function of ``(platform_value, chat_type)``.

The block lives in a new sibling (not appended to ``gateway.session``)
per the facade rule in the root AGENTS.md: ``gateway.session`` is the
public facade; new topical behavior lives in ``gateway/<topic>.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from gateway.config import Platform


@dataclass(frozen=True)
class ChannelCapabilities:
    """Behavioral guidance for one channel (platform value).

    Fields are authored static text.  Empty fields are omitted from the
    rendered block — the renderer never invents or interpolates untrusted
    values into this text.
    """

    # "sync" | "async" | "autonomous" — interaction mode.  Short
    # parenthetical suffixes are allowed (e.g. "sync (user is present and
    # can respond in real time)").  Test asserts the field starts with one
    # of the three canonical tokens.
    mode: str
    # Response style guidance (e.g. "conversational, keep messages brief",
    # "well-structured document-style allowed").
    style: str
    # Length / format constraints or "" for none (e.g. "~1600 characters",
    # "keep messages short; long responses may be truncated").
    limits: str
    # Short media-delivery summary or "".  Do NOT duplicate the per-platform
    # MEDIA: syntax detail that PLATFORM_HINTS already carries; a one-liner
    # at most.
    media: str = ""
    # A few short capability phrases (e.g. ("file delivery via MEDIA:",
    # "reactions")).  Empty tuple = none claimed.
    actions: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Registry: platform value -> behavioral guidance.
#
# Every entry MUST be backed by a real Platform member or a bundled plugin
# platform name (Platform._missing_ accepts both).  TestEveryPlatformRendersABlock
# + TestRegistryKeysAreRealPlatforms pin the contract.
#
# Behavioral guidance is derived from the existing PLATFORM_HINTS text
# (agent/prompt_builder.py) and _STATIC_PLATFORM_NOTES text (gateway/session.py)
# — the new block must NEVER contradict those.
# ---------------------------------------------------------------------------

CHANNEL_CAPABILITIES: dict[str, ChannelCapabilities] = {
    # ---- 24 built-in Platform members (gateway/config.py) ----
    Platform.LOCAL.value: ChannelCapabilities(
        mode="sync (operator is at the terminal; responses stream live)",
        style="plain text — markdown does not render (terminal / TUI / desktop framing varies)",
        limits="no hard cap; prefer concise output the operator can read at a glance",
        media="images/audio/video inline via MEDIA: where the runtime supports it (desktop yes; cli/tui no — state absolute paths)",
        actions=("MEDIA: file delivery on supported runtimes",),
    ),
    Platform.TELEGRAM.value: ChannelCapabilities(
        mode="sync (user is present and can respond in real time)",
        style="conversational, prefer bullets or labeled lines for structured data",
        limits="messages may be long; Telegram renders long replies natively",
        media="images, videos, audio (voice bubble with [[audio_as_voice]]), documents via MEDIA:",
        actions=("MEDIA: file delivery",),  # no agent-facing reactions/keyboards: adapter.py reply_markup is gateway-internal picker UI
    ),
    Platform.DISCORD.value: ChannelCapabilities(
        mode="sync (user is present in the channel)",
        style="conversational; bullets or labeled lines (no tables)",
        limits="Discord's 2000-char message limit; long replies may need to be split",
        media="images, audio, documents via MEDIA:; image URLs via ![alt](url) as attachments",
        actions=("MEDIA: file delivery", "threads"),  # reactions are tool-gated (_discord_platform_notes)
    ),
    Platform.WHATSAPP.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="long replies supported; auto-converted markdown (*bold*, _italic_, ~strike~)",
        media="images (.jpg/.png/.webp), videos (.mp4/.mov), audio, documents via MEDIA:",
        actions=("MEDIA: file delivery",),
    ),
    Platform.WHATSAPP_CLOUD.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="long replies supported; Meta 24h free-form window restriction (error 131047) matters only for delayed sends",
        media="images (.jpg/.png), videos (.mp4), audio as voice/audio, documents via MEDIA:",
        actions=("MEDIA: file delivery",),
    ),
    Platform.SLACK.value: ChannelCapabilities(
        mode="sync (user is present in the channel)",
        style="conversational; bullets or labeled lines (no tables)",
        limits="Slack message limits vary by workspace; standard markdown auto-converted to Slack format",
        media="images, audio, documents via MEDIA:; image URLs via ![alt](url) as attachments",
        actions=("MEDIA: file delivery", "threads"),  # reactions are tool-gated (_slack_platform_notes)
    ),
    Platform.SIGNAL.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="standard markdown auto-converts; bullets render as •",
        media="images, documents via MEDIA:; image URLs via ![alt](url) as photos",
        actions=("MEDIA: file delivery",),
    ),
    Platform.MATTERMOST.value: ChannelCapabilities(
        mode="sync (user is present in the channel)",
        style="conversational; markdown including tables supported",
        limits="long replies supported",
        media="images, audio, video, documents via MEDIA:; ![alt](url) inline previews",
        actions=("MEDIA: file delivery",),
    ),
    Platform.MATRIX.value: ChannelCapabilities(
        mode="sync (user is present in the room)",
        style="conversational; bullets or labeled lines (no tables — Element X collapses them)",
        limits="long replies supported; markdown converts to HTML (avoid ||spoilers||, ~~strike~~, checkboxes)",
        media="images, audio (.ogg/.mp3) as voice, video (.mp4) inline, documents via MEDIA:",
        actions=("MEDIA: file delivery", "reply threads via reply_to"),  # send(reply_to=...) is agent-facing; _send_reaction is gateway-internal
    ),
    Platform.HOMEASSISTANT.value: ChannelCapabilities(
        mode="autonomous (no human in a chat sense — service-triggered)",
        style="structured, machine-friendly output (this surfaces as Home Assistant data, not a chat)",
        limits="compact, deterministic output",
        media="media support depends on the trigger; no MEDIA: by default",
        actions=(),
    ),
    Platform.EMAIL.value: ChannelCapabilities(
        mode="async (response sits in the recipient's inbox until they read it)",
        style="clear, well-structured prose — plain text only (no markdown)",
        limits="long replies expected; subject line preserved for threading",
        media="file attachments via MEDIA:",
        actions=("MEDIA: file delivery",),
    ),
    Platform.SMS.value: ChannelCapabilities(
        mode="async (user may not be at the device; replies arrive as text messages)",
        style="plain text only — no markdown, no formatting",
        limits="limited to ~1600 characters; keep messages brief and direct",
        media="no native media channel; state URLs or paths in plain text",
        actions=(),
    ),
    Platform.DINGTALK.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="markdown supported; long replies supported",
        media="images via MEDIA: (adapter send_image/send_image_file); no document/attachment send",  # media refs in adapter are inbound-only
        actions=("MEDIA: image delivery",),  # images only: adapter send_image/send_image_file; no document send
    ),
    Platform.API_SERVER.value: ChannelCapabilities(
        mode="sync (or async depending on endpoint — treat as a programmatic API call)",
        style="plain text — assume the rendering layer is unknown (no markdown)",
        limits="keep responses brief and natural; the caller parses the result",
        media="image MEDIA: tags inline as base64 data URLs on chat/completions/responses endpoints; other files are not intercepted — state plain file paths",
        actions=(),
    ),
    Platform.WEBHOOK.value: ChannelCapabilities(
        mode="async — an external system POSTed an event; no human is in a chat",
        style="structured, machine-friendly (the consumer parses the response)",
        limits="no rendering constraints; keep responses to what the consumer expects",
        media="media support depends on the consumer",
        actions=(),
    ),
    Platform.MSGRAPH_WEBHOOK.value: ChannelCapabilities(
        mode="async — Microsoft Graph subscription notification; no human in a chat",
        style="structured, machine-friendly (the Graph webhook handler parses it)",
        limits="no rendering constraints",
        media="media support depends on the downstream handler",
        actions=(),
    ),
    Platform.FEISHU.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief; markdown supported (bold, italic, code, links)",
        limits="long replies supported",
        media="images, audio as native voice (non-Opus transcoded; without ffmpeg falls back to file attachments), documents via MEDIA:",
        actions=("MEDIA: file delivery",),  # reactions are processing-status markers set by the gateway (adapter.py:180-186), not agent-callable
    ),
    Platform.WECOM.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="markdown supported",
        media="images (.jpg/.png/.webp, ≤10 MB), documents (≤20 MB), videos (.mp4) via MEDIA:; voice messages must be AMR — other audio sends as file attachments",
        actions=("MEDIA: file delivery",),
    ),
    Platform.WECOM_CALLBACK.value: ChannelCapabilities(
        mode="async — WeCom callback event; no human in a chat sense",
        style="structured, machine-friendly (WeCom handler parses the response)",
        limits="no rendering constraints; match the response shape WeCom expects",
        media="media support depends on the downstream handler",
        actions=(),
    ),
    Platform.WEIXIN.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, compact, chat-friendly",
        limits="markdown supported; keep the message compact",
        media="images, videos, documents via MEDIA:; image URLs via ![alt](url) downloaded and sent as native media",
        actions=("MEDIA: file delivery",),
    ),
    Platform.BLUEBUBBLES.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="plain text — iMessage does not render markdown; texts, not essays",
        limits="each block between blank lines is delivered as its own bubble; one idea per bubble, 1–3 sentences each",
        media="images (.jpg/.png/.heic) as photos, other files as attachments via MEDIA:",
        actions=("MEDIA: file delivery",),
    ),
    Platform.QQBOT.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational; markdown and emoji supported",
        limits="long replies supported",
        media="images as native photos, documents via MEDIA:",
        actions=("MEDIA: file delivery",),
    ),
    Platform.YUANBAO.value: ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational; markdown renders (code blocks, tables, bold/italic)",
        limits="long replies supported",
        media="images (.jpg/.png/.webp/.gif), documents (≤50 MB) via MEDIA:; stickers via yb_search_sticker / yb_send_sticker (never substitute Unicode emoji or PNG stickers)",
        actions=("MEDIA: file delivery", "yb_send_dm for private messages", "sticker tools"),
    ),
    Platform.RELAY.value: ChannelCapabilities(
        mode="async (the relay adapter fronts an external connector; responses are relayed back)",
        style="depends on the downstream connector — keep it minimal and predictable",
        limits="depends on the downstream connector",
        media="depends on the downstream connector",
        actions=(),
    ),

    # ---- Bundled plugin platforms (plugins/platforms/<name>/) ----
    # Bundled plugins that already register a ``platform_hint`` typically have
    # rich guidance in PLATFORM_HINTS or their adapter; mirror that here.
    "a2a": ChannelCapabilities(
        mode="sync (agent-to-agent protocol; peer agent is the counterpart)",
        style="structured payload the peer agent will parse",
        limits="no rendering constraints; match the A2A contract",
        media="depends on the A2A peer",
        actions=(),
    ),
    "buzz": ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="markdown supported",
        media="images, documents via MEDIA:",
        actions=("MEDIA: file delivery",),
    ),
    "google_chat": ChannelCapabilities(
        mode="sync (user is present in the space)",
        style="conversational; markdown supported",
        limits="long replies supported",
        media="images, documents via MEDIA:",
        actions=("MEDIA: file delivery", "threads"),
    ),
    "irc": ChannelCapabilities(
        mode="sync (user is present in the channel)",
        style="plain text — IRC does not render markdown",
        limits="long replies supported but consider the channel's culture; no tables",
        media="no native media; state URLs or paths in plain text",
        actions=(),
    ),
    "line": ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="markdown rendering depends on the LINE client; prefer plain text",
        media="images as native photos, audio, documents via MEDIA:",
        actions=("MEDIA: file delivery",),
    ),
    "ntfy": ChannelCapabilities(
        mode="async — ntfy.sh push notification; no live user in a chat",
        style="structured, machine-friendly (the subscriber parses the notification)",
        limits="keep notifications compact",
        media="no attachment primitive — state URLs in plain text (adapter.py send: 'signature parity only')",  # ntfy/adapter.py:352-353
        actions=(),
    ),
    "photon": ChannelCapabilities(
        mode="async — Photon-XMPP message; recipient reads asynchronously",
        style="conversational, brief",
        limits="depends on the recipient client",
        media="images, voice, video via MEDIA: (adapter send_image/send_voice/send_video)",
        actions=("MEDIA: file delivery", "reactions via send_message action='react'"),  # only adapter implementing agent-facing add_reaction (adapter.py:1264)
    ),
    "raft": ChannelCapabilities(
        mode="sync (or async depending on the Raft surface; a Raft peer or operator is the counterpart)",
        style="structured, machine-friendly",
        limits="match the Raft protocol contract",
        media="media support depends on the surface",
        actions=(),
    ),
    "simplex": ChannelCapabilities(
        mode="sync (user is present in the chat)",
        style="conversational, brief",
        limits="markdown support depends on the client; prefer plain text",
        media="images, documents via MEDIA:; image URLs via ![alt](url)",
        actions=("MEDIA: file delivery",),
    ),
    "teams": ChannelCapabilities(
        mode="sync (user is present in the chat or channel)",
        style="conversational; markdown supported (Teams Adaptive Card rendering for rich content)",
        limits="long replies supported",
        media="images, documents via MEDIA:",
        actions=("MEDIA: file delivery", "threads"),  # thread_id send param: teams/adapter.py:184; no reaction support in adapter
    ),
}


def channel_capabilities_block(platform_value: str, chat_type: str = "dm") -> str:
    """Render the ``**Channel Capabilities:**`` block for a platform, or ``""``.

    The renderer is a pure function of ``(platform_value, chat_type)``.  It
    never reads env vars, config, adapter state, or time — the same inputs
    are already hashed in ``gateway.run_agent_cache._ephemeral_change_key``
    (``src.platform.value`` and ``src.chat_type``), so the block is
    automatically pin-safe.

    ``chat_type`` is currently unused: entries do not vary by chat type
    today.  It is kept in the signature for render-input parity with the
    change key and reserved for a future chat-type-specific entry.

    Shape::

        **Channel Capabilities:**
        - mode: sync (user is present and can respond in real time)
        - style: conversational, keep messages brief
        - limits: ~1600 characters
        - media: images, audio, video, documents
        - actions: file delivery via MEDIA:, reactions

    Empty fields are omitted; the heading line is always present when the
    block renders.  Multi-line (``\\n``-joined) text is returned; the caller
    appends it to the prompt's ``lines`` list and joins with ``\\n``.
    """
    caps = CHANNEL_CAPABILITIES.get(platform_value)
    if caps is None:
        return ""
    lines = ["**Channel Capabilities:**"]
    if caps.mode:
        lines.append(f"- mode: {caps.mode}")
    if caps.style:
        lines.append(f"- style: {caps.style}")
    if caps.limits:
        lines.append(f"- limits: {caps.limits}")
    if caps.media:
        lines.append(f"- media: {caps.media}")
    if caps.actions:
        lines.append(f"- actions: {', '.join(caps.actions)}")
    return "\n".join(lines)
