"""Mention detection and attachment provenance for inbound Slack events.

Finds the ``<@UID>`` tokens a message actually addresses across every carrier
Slack uses (flat ``text``, Block Kit ``blocks``, legacy ``attachments``) while
skipping content that is displayed rather than spoken: quotes, code, forwarded
shares and automatic link unfurls.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret

logger = logging.getLogger(__name__)

# A user-mention token as it appears in a mrkdwn/plain_text string, optionally
# carrying a display label (``<@U123|alice>``). Only the ID is captured, so a
# labelled mention normalizes to the bare ``<@U123>`` the gates compare against.
# The ID class is deliberately permissive: the gates substring-compare against
# whatever ``auth.test`` reported, so a narrower charset here would silently drop
# the very mention this is meant to recover. Group mentions use ``<!...>`` and are
# handled by the adapter's _SLACK_SPECIAL_MENTION_RE, so ``<@`` is always a user mention.
_SLACK_USER_MENTION_RE = re.compile(r"<@([^>|\s]+)(?:\|[^>]*)?>")

# mrkdwn blockquote markers. Slack escapes a literal ``>`` in message text to
# ``&gt;``, so both forms reach us. Quoting via mrkdwn is a carrier the
# structured ``rich_text_quote`` node check cannot see.
_SLACK_BLOCKQUOTE_PREFIXES = (">", "&gt;")

# Rich-text containers whose contents are *displayed*, not spoken. Quoted text
# carries someone else's words; code/preformatted text carries a token being
# shown to the reader (a log line, a payload dump, an example). A ``<@UID>``
# inside either is not an address, so neither may summon the bot — the same
# contract #52390 established for ``rich_text_quote``, applied to every carrier
# that shares its "verbatim content" nature.
_SLACK_VERBATIM_RICH_TEXT_TYPES = ("rich_text_quote", "rich_text_preformatted")

# An mrkdwn inline code span (``` `<@U123>` ```). Only single-line spans are
# matched, so a lone stray backtick in prose strips nothing. Triple-backtick
# runs are consumed by the fence handling in _extract_mention_tokens.
_SLACK_MRKDWN_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

_SLACK_AUTHORED_URL_RE = re.compile(
    r"(?<![<\w])https?://[^\s<>]+|<(?P<mrkdwn>https?://[^>|]+)(?:\|[^>]*)?>",
    re.IGNORECASE,
)


def _extract_mention_tokens(value: str) -> list:
    """Return the ``<@UID>`` tokens authored in a mrkdwn string.

    Labelled mentions (``<@U123|alice>``) normalize to the bare ``<@U123>`` the
    gates compare against.

    Two kinds of region are skipped, because a token inside them is being shown
    rather than addressed to anyone:

    * Lines opening with a mrkdwn blockquote marker — quoted/forwarded content,
      the same contract the structured ``rich_text_quote`` check enforces
      applied to the carrier that check cannot see.
    * Fenced (``` ``` ```) and inline (`` ` ``) code spans. Slack does not
      linkify mrkdwn inside code, so a token there notifies nobody; a bot that
      dumps a payload or relays a log line must not summon us with it.
    """
    tokens: list = []
    in_fence = False
    for line in value.split("\n"):
        if line.lstrip().startswith(_SLACK_BLOCKQUOTE_PREFIXES):
            continue

        visible: list[str] = []
        cursor = 0
        while cursor < len(line):
            fence = line.find("```", cursor)
            if fence < 0:
                if not in_fence:
                    visible.append(line[cursor:])
                break
            if not in_fence:
                visible.append(line[cursor:fence])
            in_fence = not in_fence
            cursor = fence + 3

        scanned = _SLACK_MRKDWN_INLINE_CODE_RE.sub(" ", "".join(visible))
        tokens.extend(f"<@{uid}>" for uid in _SLACK_USER_MENTION_RE.findall(scanned))
    return tokens


def _strip_slack_user_mention(value: str, user_id: str) -> str:
    """Remove bare or labelled mentions of ``user_id`` from Slack text."""
    if not value or not user_id:
        return value
    pattern = re.compile(rf"<@{re.escape(user_id)}(?:\|[^>]*)?>")
    return pattern.sub("", value)


def _collect_slack_block_mentions(blocks: list) -> list:
    """``<@UID>`` mentions authored in non-quoted Block Kit text (flat ``text`` omits block-only
    mentions); ``rich_text_quote`` is ignored so quoted/forwarded text can't summon the bot.

    Slack's flat top-level ``text`` field does NOT contain mentions that were authored only inside Block Kit
    ``blocks`` (e.g. a ``rich_text_section`` with a ``user`` element). This walker recovers those mentions
    so the gates can see Block-Kit-only mentions instead of silently dropping them (#52387).

    Two carriers exist and both are recovered. The WYSIWYG composer emits a
    structured ``user`` element, while an app building blocks by hand writes the
    raw ``<@UID>`` token into a ``section``/``header``/``context`` block's
    ``text`` or ``fields`` string, where there is no ``user`` node to find.

    Mentions inside verbatim content are deliberately ignored, so text the
    author is *displaying* rather than speaking cannot trick the bot into
    responding (matches the existing channel-routing contract). That covers
    ``rich_text_quote`` (quoted/forwarded content) and ``rich_text_preformatted``
    plus ``style.code`` elements (a token shown as code — a log line, a payload
    dump). The flag propagates down the subtree, so a mention nested any depth
    below a verbatim node stays ignored.
    """
    mentions: list = []

    def _is_code_styled(node: dict) -> bool:
        # ``style`` is a dict on inline elements but a plain string on
        # ``rich_text_list`` ("bullet"/"ordered"), so the type check matters.
        style = node.get("style")
        return isinstance(style, dict) and bool(style.get("code"))

    def _walk(node, in_verbatim: bool) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item, in_verbatim)
            return
        if not isinstance(node, dict):
            return
        node_type = node.get("type")
        verbatim = (in_verbatim or node_type in _SLACK_VERBATIM_RICH_TEXT_TYPES
                    or _is_code_styled(node))
        if node_type == "user" and not verbatim and node.get("user_id", ""):
            mentions.append(f"<@{node['user_id']}>")
        for key in ("elements", "element", "text", "fields"):
            child = node.get(key)
            if child is None:
                continue
            if isinstance(child, str):
                if not verbatim and node_type != "plain_text":
                    mentions.extend(_extract_mention_tokens(child))
                continue
            _walk(child, verbatim)

    # Every step is type-guarded; only pathological nesting can fail. Keep what was
    # collected before it rather than breaking the gate.
    try:
        _walk(blocks, False)
    except RecursionError:
        logger.debug("[Slack] Block Kit tree too deep for mention detection; partial result kept")
    return mentions


def _collect_slack_attachment_mentions(
    attachments: list, authored_urls: Optional[set[str]] = None
) -> list:
    """Return ``<@UID>`` mention tokens authored in legacy ``attachments``.

    Alertmanager, Grafana, PagerDuty and CI bots post with an empty top-level
    ``text`` and the real content — including any ``<@UID>`` addressed at the
    bot — inside attachment fields or attachment-nested ``blocks``. Without this
    such a message carries no detectable mention at all.

    Message unfurls and forwarded shares are skipped: they carry someone else's
    words, so a pasted permalink must not summon the bot. This mirrors the
    ``is_msg_unfurl`` skip on the agent-text path and the ``rich_text_quote``
    carve-out in :func:`_collect_slack_block_mentions`.

    ``fallback`` is deliberately not scanned — Slack never renders it, so a
    mention living only there is invisible in the channel and notifies nobody.
    """
    mentions: list = []
    if not isinstance(attachments, (list, tuple)):
        return mentions
    for att in attachments:
        if not isinstance(att, dict):
            continue
        if _classify_slack_attachment(att, authored_urls or set()) != "content":
            continue
        strings = [
            att[key]
            for key in ("pretext", "title", "text")
            if isinstance(att.get(key), str)
        ]
        fields = att.get("fields")
        if isinstance(fields, (list, tuple)):
            for field in fields:
                if not isinstance(field, dict):
                    continue
                strings += [
                    field[key]
                    for key in ("title", "value")
                    if isinstance(field.get(key), str)
                ]
        for value in strings:
            mentions.extend(_extract_mention_tokens(value))
        nested = att.get("blocks")
        if nested:
            mentions += _collect_slack_block_mentions(nested)
    return mentions


def _slack_recovered_mentions(event: dict) -> list:
    """Return ``<@UID>`` tokens addressed outside a message's flat ``text``.

    Slack's flat ``text`` carries neither Block-Kit-only mentions nor mentions
    authored in legacy ``attachments`` (#52387), so the gates would never see
    them. They are returned as a separate list rather than spliced into the
    routing text on purpose: that text is also matched against user-configured
    wake-word regexes and inspected for a *leading* mention, and appending
    tokens to it would break anchored patterns and make an attachment-only
    mention masquerade as the message's opening token.
    """
    mentions: list = []
    blocks = event.get("blocks")
    if blocks:
        mentions += _collect_slack_block_mentions(blocks)
    attachments = event.get("attachments")
    if attachments:
        mentions += _collect_slack_attachment_mentions(
            attachments, _collect_slack_authored_urls(event)
        )
    # The same user is often addressed in more than one carrier (a section
    # block and an attachment mirroring it), so dedupe.
    return list(dict.fromkeys(mentions))


def _normalize_slack_http_url(value: Any) -> Optional[str]:
    """Return a minimally normalized HTTP(S) URL, or ``None`` if invalid."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            return None
        hostname = parsed.hostname.lower()
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        userinfo = ""
        if "@" in parsed.netloc:
            userinfo = parsed.netloc.rsplit("@", 1)[0] + "@"
        port = parsed.port
        netloc = f"{userinfo}{hostname}{f':{port}' if port is not None else ''}"
        path = parsed.path.rstrip("/")
        return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, parsed.fragment))
    except (TypeError, ValueError):
        return None


def _extract_authored_urls_from_slack_blocks(blocks: Any) -> list[str]:
    """Return URLs from valid, non-verbatim authored Block Kit content.

    Quoted, forwarded, preformatted, and code-styled subtrees describe content
    authored elsewhere and cannot establish provenance for sibling attachments.
    A malformed top-level block list, or one nested past the recursion limit, fails
    closed for provenance so arbitrary nested ``url`` keys cannot hide a genuine
    legacy attachment as an unfurl.
    """
    if not isinstance(blocks, list) or any(
        not isinstance(block, dict) or not isinstance(block.get("type"), str)
        for block in blocks
    ):
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _walk(node: Any, verbatim: bool = False) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item, verbatim)
            return
        if not isinstance(node, dict):
            return

        node_type = node.get("type")
        style = node.get("style")
        nested_verbatim = (
            verbatim
            or node_type in _SLACK_VERBATIM_RICH_TEXT_TYPES
            or (isinstance(style, dict) and style.get("code") is True)
        )
        if nested_verbatim:
            return

        for key in ("url", "image_url", "external_url"):
            value = node.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                if value not in seen:
                    seen.add(value)
                    found.append(value)
        for value in node.values():
            if isinstance(value, (dict, list)):
                _walk(value, nested_verbatim)

    try:
        _walk(blocks)
    except RecursionError:
        logger.debug("[Slack] Block Kit tree too deep for URL provenance; treating as none")
        return []
    return found


def _collect_slack_authored_urls(event: dict) -> set[str]:
    """Collect HTTP(S) URLs authored in top-level text and Block Kit only."""
    urls: set[str] = set()
    if not isinstance(event, dict):
        return urls

    text = event.get("text")
    if isinstance(text, str):
        for match in _SLACK_AUTHORED_URL_RE.finditer(text):
            raw = match.group("mrkdwn") or match.group(0)
            # Trim common prose punctuation from bare URLs. Slack mrkdwn URLs
            # are captured without their closing ``>`` and need no trimming.
            if not match.group("mrkdwn"):
                raw = raw.rstrip(".,;:!?")
                for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
                    while raw.endswith(closer) and raw.count(closer) > raw.count(opener):
                        raw = raw[:-1]
            normalized = _normalize_slack_http_url(raw)
            if normalized:
                urls.add(normalized)

    for raw in _extract_authored_urls_from_slack_blocks(event.get("blocks")):
        normalized = _normalize_slack_http_url(raw)
        if normalized:
            urls.add(normalized)
    return urls


def _classify_slack_attachment(attachment: Any, authored_urls: set[str]) -> str:
    """Classify a Slack attachment as content, share, or automatic unfurl.

    Explicit Slack flags are authoritative. Otherwise URL provenance must tie
    the attachment back to a URL authored outside the attachment; uncertain
    legacy attachments fail open as content. Shares cannot contribute control
    mentions, but remain visible to the agent as user-provided context.
    """
    if not isinstance(attachment, dict):
        return "content"
    if attachment.get("is_share") is True:
        return "share"
    if (
        attachment.get("is_app_unfurl") is True
        or attachment.get("is_msg_unfurl") is True
    ):
        return "unfurl"
    if not authored_urls:
        return "content"
    # Legacy alert/CI attachments commonly link their subject to the same URL
    # authored in top-level blocks. Structured alert content is positive
    # evidence that this is the message body, not Slack-generated preview data.
    if any(key in attachment for key in ("pretext", "fields", "blocks")):
        return "content"
    for key in ("original_url", "from_url", "title_link"):
        normalized = _normalize_slack_http_url(attachment.get(key))
        if normalized and normalized in authored_urls:
            return "unfurl"
    return "content"


class SlackMentionGateMixin:
    """Decide whether an inbound Slack message addresses this bot (mixed into ``SlackAdapter``)."""

    config: Any

    def _slack_event_mentions_bot(self, event: dict, bot_uid: str) -> bool:
        """Return True when ``event`` @-mentions ``bot_uid`` in any carrier.

        Checks the flat ``text`` plus the mentions recovered from Block Kit
        blocks and legacy ``attachments`` (#52387) — where alert/CI apps put
        them when the top-level text is empty.
        """
        if not bot_uid:
            return False
        token = f"<@{bot_uid}>"
        top_level_text = event.get("text")
        if isinstance(top_level_text, str) and token in _extract_mention_tokens(
            top_level_text
        ):
            return True
        return token in _slack_recovered_mentions(event)

    def _slack_mention_gate_inputs(
        self, event: dict, bot_uid: str, flat_text: str = ""
    ) -> tuple[str, bool]:
        """Return ``(routing_text, is_mentioned)`` for the channel routing gates.

        ``routing_text`` is the flat message text and nothing else — the
        wake-word patterns and the leading-mention check both read it, so
        recovered mentions are reported through ``is_mentioned`` instead of
        being spliced in (see :func:`_slack_recovered_mentions`).
        """
        routing_text = flat_text or event.get("text") or ""
        is_mentioned = bool(
            self._slack_event_mentions_bot(event, bot_uid)
            or self._slack_message_matches_mention_patterns(routing_text)
        )
        return routing_text, is_mentioned

    def _slack_message_addressed_to_other_user(self, text: str, self_uids: set) -> bool:
        """True when the first token is a user mention (``<@U123>``/``<@U123|name>``)
        of someone other than the bot; ``<!here>``/``<#C…>`` address the room, not a person."""
        match = text and re.match(r"\s*<@([^>|\s]+)(?:\|[^>]*)?>", text)
        return bool(match) and match.group(1) not in self_uids

    def _slack_message_mentions_self(self, text: str, self_uids: set) -> bool:
        """True when ``text`` @-mentions this bot anywhere, in either ``<@U123>`` or
        ``<@U123|name>`` form (``is_mentioned`` only recognises the former)."""
        return bool(text) and any(
            re.search(rf"<@{re.escape(uid)}(?:\|[^>]*)?>", text) for uid in self_uids)

    def _slack_mention_patterns(self) -> list[re.Pattern]:
        """Compile (cached) wake-word regexes from ``slack.mention_patterns`` (list/str) or
        ``SLACK_MENTION_PATTERNS`` (JSON list or newline/comma-separated)."""
        cached = getattr(self, "_compiled_mention_patterns", None)
        if cached is not None:
            return cached
        patterns = self.config.extra.get("mention_patterns") if self.config.extra else None
        if patterns is None:
            raw = (_get_scoped_secret("SLACK_MENTION_PATTERNS", "") or "").strip()
            if raw:
                try:
                    import json as _json
                    patterns = _json.loads(raw)
                except Exception:
                    patterns = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        if isinstance(patterns, str):
            patterns = [patterns]
        compiled: list[re.Pattern] = []
        if isinstance(patterns, list):
            for pat in patterns:
                if not isinstance(pat, str) or not pat.strip():
                    continue
                try:
                    compiled.append(re.compile(pat, re.IGNORECASE))
                except re.error as exc:
                    logger.warning("[Slack] Invalid mention pattern %r: %s", pat, exc)
        elif patterns is not None:
            logger.warning(
                "[Slack] mention_patterns must be a list or string; got %s", type(patterns).__name__
            )
        if compiled:
            logger.info("[Slack] Loaded %d mention pattern(s)", len(compiled))
        self._compiled_mention_patterns = compiled
        return compiled

    def _slack_message_matches_mention_patterns(self, text: str) -> bool:
        """Return True when ``text`` matches a configured wake-word pattern."""
        return bool(text) and any(p.search(text) for p in self._slack_mention_patterns())
