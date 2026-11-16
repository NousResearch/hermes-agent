"""Slack forwarded ("shared") messages: what the agent reads for each one.

A forward reaches the bot as an ``attachments[]`` entry flagged ``is_share``. Every real payload
checked also carries ``is_msg_unfurl`` (the flag the adapter skips so that a pasted permalink's
unfurl never echoes a message back), plus ``message_blocks`` (the original's Block Kit),
``text``/``fallback``, ``author_name``/``author_id``, ``channel_id``, ``from_url`` (the
permalink), ``ts``, ``channel_name`` when the bot can see the source and ``files`` when the
original had files. A pasted permalink unfurl never sets ``is_share``, so it is the positive
discriminator here and the unfurl skip stays as it was.
"""

import re
from typing import Any, Callable, Collection

from agent.compression_marker import ELISION_MARKER_MAX_LEN, elide

#: One ceiling for every forward in a message: a long ("Show more") original arrives whole, and
#: several forwards share the budget like the unfurl path's nested blocks do.
SLACK_FORWARDS_MAX_CHARS = 8000
#: Shorter bodies are always rendered: a substring match on a few words is not proof the message
#: already carries the forward, and dropping it silently is the bug this module fixes.
_DEDUPE_MIN_CHARS = 20
_FILE_URL_KEYS = ("url_private_download", "url_private")
_QUOTED_NOTE = "(content quoted in the message above)"
_ENTITIES = {"&lt;": "<", "&gt;": ">", "&amp;": "&"}

RenderBlocks = Callable[[list], str]


def slack_file_marker(file_obj: dict[str, Any]) -> str:
    """Render a compact text marker for a Slack file so text-only context shows attachments. Name is
    sanitized (newlines/brackets stripped) so a hostile filename can't fake context structure."""
    name = str(file_obj.get("name") or file_obj.get("title") or file_obj.get("id") or "file")
    name = re.sub(r"[\r\n\[\]]+", " ", name).strip() or "file"
    mimetype = str(file_obj.get("mimetype") or "")
    for kind in ("image", "video", "audio"):
        if mimetype.startswith(kind + "/"):
            return f"[{kind}: {name}]"
    return f"[file: {name} ({mimetype})]" if mimetype else f"[file: {name}]"


def is_forward(att: Any) -> bool:
    return isinstance(att, dict) and bool(att.get("is_share"))


def own_bot_ids(primary: str | None, per_team: dict[str, str]) -> frozenset[str]:
    """Every bot user id this adapter posts as: a multi-workspace install (and an Enterprise Grid
    workspace installed separately) has a different bot user per team, and a permalink to the
    bot's message in one workspace can be pasted into another."""
    return frozenset(uid for uid in (primary, *per_team.values()) if uid)


def is_linked_message(att: Any, own_ids: Collection[str] = ()) -> bool:
    """A pasted permalink's unfurl of someone else's message. Slack ships the linked message's
    content even when the bot cannot read the source conversation, so it is the only copy the
    agent gets. An unfurl of this bot's own message, or one Slack does not attribute, is treated
    as an echo and stays skipped."""
    if not isinstance(att, dict) or att.get("is_share") or not att.get("is_msg_unfurl"):
        return False
    author = str(att.get("author_id") or "")
    return bool(author) and author not in own_ids


def is_shared_message(att: Any, own_ids: Collection[str] = ()) -> bool:
    """A forward or a linked message: content the user is pointing the agent at."""
    return is_forward(att) or is_linked_message(att, own_ids)


def files_with_forwarded(files: Any, attachments: Any) -> list[dict]:
    """``files`` plus the files of forwarded messages, deduped by file id (first copy wins)."""
    merged = [f for f in (files or []) if isinstance(f, dict)]
    seen = {f.get("id") for f in merged if f.get("id")}
    for att in attachments or []:
        for f in (att.get("files") or []) if is_forward(att) else []:
            if isinstance(f, dict) and not (f.get("id") and f.get("id") in seen):
                seen.add(f.get("id"))
                merged.append(f)
    return merged


def _inline(value: Any) -> str:
    """Author and channel names are user-controlled: one line, no brackets."""
    return re.sub(r"[\r\n\[\]]+", " ", str(value or "")).strip()


def _forward_blocks(att: dict) -> list[dict]:
    blocks: list[dict] = []
    for entry in att.get("message_blocks") or []:
        message = entry.get("message") if isinstance(entry, dict) else None
        if isinstance(message, dict) and isinstance(message.get("blocks"), list):
            blocks += [b for b in message["blocks"] if isinstance(b, dict)]
    if not blocks and isinstance(att.get("blocks"), list):
        blocks = [b for b in att["blocks"] if isinstance(b, dict)]
    return blocks


def _block_text(block: dict, render_blocks: RenderBlocks) -> str:
    """One block's text. ``section``/``header``/``context`` carry an app post's body; rich_text
    and tables go through the adapter's renderer."""
    if block.get("type") not in ("section", "header", "context"):
        return render_blocks([block]).strip()
    items = [block.get("text")] + list(block.get("fields") or []) + list(block.get("elements") or [])
    return "\n".join(i["text"].strip() for i in items
                     if isinstance(i, dict) and isinstance(i.get("text"), str) and i["text"].strip())


def forward_body(att: dict, render_blocks: RenderBlocks) -> str:
    """The original's content: its blocks in their original order, else ``text``, else Slack's
    own ``fallback`` rendering."""
    body = "\n".join(t for t in (_block_text(b, render_blocks) for b in _forward_blocks(att)) if t)
    if not body:
        body = re.sub(r"&(lt|gt|amp);", lambda m: _ENTITIES[m.group(0)], str(att.get("text") or "")).strip()
    return body or str(att.get("fallback") or "").strip()


def _file_line(att: dict) -> str:
    markers = [
        slack_file_marker(f) + ("" if any(f.get(k) for k in _FILE_URL_KEYS)
                                or f.get("file_access") == "check_file_info" else " (not downloadable)")
        for f in att.get("files") or [] if isinstance(f, dict)]
    return ("Files: " + " ".join(markers)) if markers else ""


def forward_section(att: dict, render_blocks: RenderBlocks, limit: int = SLACK_FORWARDS_MAX_CHARS,
                    body: str | None = None) -> str:
    """One forward as the agent reads it: who and where from, the permalink, the content (capped
    with the shared elision marker) and its files; or an explicit marker when Slack sent no
    readable content, so a forward never disappears without a trace."""
    who = _inline(att.get("author_name") or att.get("author_subname") or att.get("author_id"))
    where = f" in #{_inline(att['channel_name'])}" if att.get("channel_name") else ""
    kind = "Forwarded" if is_forward(att) else "Linked"
    header = f"{kind} Slack message{' from ' + who if who else ''}{where}"
    link = _inline(att.get("from_url"))
    body = forward_body(att, render_blocks) if body is None else body
    files = _file_line(att)
    if not body and not files:
        return f"[{header} is present but its content is not available{': ' + link if link else ''}]"
    lines = [f"[{header}]", f"Link: {link}" if link else "", elide(body, max(limit, ELISION_MARKER_MAX_LEN)),
             files, "[End of forwarded message]"]
    return "\n".join(line for line in lines if line)


def _message_refs(att: dict) -> set[str]:
    """Ways to name the shared message: its permalink (query dropped) and channel + ts."""
    refs = {str(att.get("from_url") or "").split("?", 1)[0]}
    if att.get("channel_id") and att.get("ts"):
        refs.add(f"{att['channel_id']}/{att['ts']}")
    return refs - {""}


def shared_message_sections(attachments: Any, render_blocks: RenderBlocks, own_ids: Collection[str] = (),
                             seen_text: str = "") -> list[str]:
    """Every forward and linked message in ``attachments``, rendered under ONE body budget. A
    linked message that is also forwarded in the same post renders once, as the forward (which
    also carries its files). A body already quoted in ``seen_text`` keeps its source line
    and files but is not repeated."""
    atts = [a for a in attachments or [] if is_forward(a)]
    atts += [a for a in attachments or [] if is_linked_message(a, own_ids)]
    budget, flat_seen = SLACK_FORWARDS_MAX_CHARS, " ".join((seen_text or "").split())
    seen_refs: set[str] = set()
    sections: list[str] = []
    for att in atts:
        if is_forward(att):
            seen_refs |= _message_refs(att)
        elif _message_refs(att) & seen_refs:
            continue  # the same message is already here as a forward
        body = forward_body(att, render_blocks)
        if len(body) >= _DEDUPE_MIN_CHARS and " ".join(body.split()) in flat_seen:
            body = _QUOTED_NOTE
        sections.append(forward_section(att, render_blocks, limit=budget, body=body))
        budget = max(budget - len(body), 0)
    return sections


def append_forwards(text: str, attachments: Any, render_blocks: RenderBlocks, own_ids: Collection[str] = ()) -> str:
    """Append every forward and linked message in ``attachments`` to ``text`` (the user's own
    words stay first)."""
    sections = shared_message_sections(attachments, render_blocks, own_ids, text)
    return (text.strip() + "\n\n" + "\n\n".join(sections)).strip() if sections else text
