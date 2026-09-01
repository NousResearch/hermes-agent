"""Idea Box flow — capture → dedup → card → confirm/reject.

The flow is the orchestrator for the Idea Box bounded flow.  It ties together
the data models and the dedup checker, and provides the interface the agent
uses to process ``/idea <text>`` messages.

Key design decisions (from PRD):
- Sahil confirmation is a hard gate — no auto-promotion
- Confirmed ideas create a Kanban triage task via kanban_create with --triage
- Rejected ideas are logged to LLM Wiki as rejected-idea provenance
- No idea auto-creates a backlog task, dispatches a worker, or triggers runtime
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from idea_box.models import DedupResult, IdeaCard, IdeaStatus, SourceRef
from idea_box.dedup import DedupChecker

logger = logging.getLogger("idea_box.flow")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Match "/idea <text>" or "/idea" with no args (error)
IDEA_COMMAND_RE = re.compile(r"^/idea\s+(.+)$", re.DOTALL)

# Expected channel name for idea capture
IDEA_CHANNEL_NAME = "idea-box"


def parse_idea_command(text: str) -> Optional[str]:
    """Extract the idea text from a ``/idea <text>`` message.

    Returns the idea text, or None if the message is not an /idea command.
    """
    if not text:
        return None
    match = IDEA_COMMAND_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Flow result
# ---------------------------------------------------------------------------

@dataclass
class FlowResult:
    """Outcome of an idea box flow run.

    Encapsulates everything the agent needs to present to Sahil or act on.
    """

    card: IdeaCard
    action: str               # "present" | "duplicate" | "confirmed" | "rejected" | "wiki"
    message: str              # Human-readable summary for Discord
    kanban_task_id: Optional[str] = None  # Set on confirmation
    wiki_path: Optional[str] = None        # Set on wiki logging


# ---------------------------------------------------------------------------
# IdeaBoxFlow
# ---------------------------------------------------------------------------

class IdeaBoxFlow:
    """Orchestrates the full idea box flow.

    Usage:
        flow = IdeaBoxFlow()
        result = flow.capture(text, source_ref)
        # Present result.card to Sahil
        # On confirm: flow.confirm(result.card) → creates Kanban triage task
        # On reject: flow.reject(result.card, reason) → logs to LLM Wiki
    """

    def __init__(
        self,
        *,
        dedup_checker: Optional[DedupChecker] = None,
        kanban_create_fn: Any = None,
        wiki_log_fn: Any = None,
    ):
        self._dedup = dedup_checker or DedupChecker()
        # Injection points for testing; fall back to real backends at runtime
        self._kanban_create_fn = kanban_create_fn
        self._wiki_log_fn = wiki_log_fn

    # -- Step 1: Capture + Dedup ------------------------------------------

    def capture(self, idea_text: str, source: SourceRef) -> FlowResult:
        """Capture an idea, run dedup, and produce a card.

        This is the entry point when ``/idea <text>`` is received.
        The card is returned with dedup_status set to either
        ``DUPLICATE`` or ``NOVEL``.
        """
        if not idea_text or not idea_text.strip():
            return FlowResult(
                card=IdeaCard(summary="", source=source),
                action="error",
                message="No idea text provided. Usage: `/idea <your idea>`",
            )

        # Run dedup
        dedup_result: DedupResult = self._dedup.check(idea_text)

        # Build tags from simple keyword extraction
        tags = self._extract_tags(idea_text)

        if dedup_result.is_duplicate:
            card = IdeaCard(
                summary=idea_text,
                source=source,
                tags=tags,
                dedup_status=IdeaStatus.DUPLICATE,
                dedup_matches=dedup_result.matches,
            )
            match_lines = []
            for m in dedup_result.matches:
                source_label = m.get("source", "?")
                title = m.get("title", m.get("summary", ""))
                ref = m.get("ref_id", "")
                score = m.get("score", 0)
                match_lines.append(
                    f"  - **{source_label}**: {title} (ref: {ref}, score: {score})"
                )
            message = (
                f"**Potential duplicate idea detected.**\n\n"
                f"Your idea: {idea_text}\n\n"
                f"**Matches found:**\n" + "\n".join(match_lines) + "\n\n"
                f"Reply `confirm` to create a triage task anyway, or `reject` to discard."
            )
            return FlowResult(card=card, action="duplicate", message=message)

        # Novel idea
        card = IdeaCard(
            summary=idea_text,
            source=source,
            tags=tags,
            dedup_status=IdeaStatus.NOVEL,
            dedup_matches=[],
        )
        message = card.render_card() + "\n\nReply `confirm` to create a triage task, `reject` to discard, or `wiki` to also create an LLM Wiki entry."
        return FlowResult(card=card, action="present", message=message)

    # -- Step 2a: Confirm → Kanban triage --------------------------------

    def confirm(
        self,
        card: IdeaCard,
        *,
        assignee: str = "triage",
        create_wiki: bool = False,
    ) -> FlowResult:
        """Confirm an idea and create a Kanban triage task.

        Uses the kanban_create tool with triage=True.  The task body
        includes the idea card data and source reference.
        """
        if card.dedup_status not in (IdeaStatus.NOVEL, IdeaStatus.DUPLICATE):
            return FlowResult(
                card=card,
                action="error",
                message=f"Idea already {card.dedup_status.value} — cannot confirm.",
            )

        body = self._build_task_body(card)

        task_id = self._create_kanban_task(
            title=card.summary[:120],
            body=body,
            assignee=assignee,
            triage=True,
        )

        card.idea_id = task_id
        card.dedup_status = IdeaStatus.CONFIRMED

        message = f"Idea confirmed. Kanban triage task created: **{task_id}**"

        # Optional wiki entry
        wiki_path = None
        if create_wiki:
            wiki_path = self._log_to_wiki(card, action="confirmed")
            if wiki_path:
                message += f"\nLLM Wiki entry: {wiki_path}"

        return FlowResult(
            card=card,
            action="confirmed",
            message=message,
            kanban_task_id=task_id,
            wiki_path=wiki_path,
        )

    # -- Step 2b: Reject → LLM Wiki provenance ----------------------------

    def reject(self, card: IdeaCard, reason: str = "") -> FlowResult:
        """Reject an idea and log it to the LLM Wiki for provenance.

        The rejection is logged so rejected ideas are searchable later,
        preventing repeated capture of the same rejected concept.
        """
        card.rejection_reason = reason or "No reason given"
        card.dedup_status = IdeaStatus.REJECTED

        wiki_path = self._log_to_wiki(card, action="rejected")

        message = f"Idea rejected. Logged to LLM Wiki: {wiki_path or '(wiki unavailable)'}"

        return FlowResult(
            card=card,
            action="rejected",
            message=message,
            wiki_path=wiki_path,
        )

    # -- Helpers ----------------------------------------------------------

    def _extract_tags(self, text: str) -> list[str]:
        """Simple keyword extraction for tags.

        Extracts words > 4 chars that aren't common stop words.
        Keeps top 5 by frequency.  This is intentionally simple —
        the agent can refine tags during presentation.
        """
        stop = {"the", "this", "that", "with", "from", "have", "would",
                "could", "should", "there", "their", "about", "which",
                "when", "what", "where", "while", "after", "before"}
        words = [w.lower() for w in text.split() if len(w) > 4 and w.lower() not in stop]
        # Deduplicate preserving order, top 5
        seen: set[str] = set()
        tags: list[str] = []
        for w in words:
            if w not in seen:
                seen.add(w)
                tags.append(w)
            if len(tags) >= 5:
                break
        return tags

    def _build_task_body(self, card: IdeaCard) -> str:
        """Build the Kanban task body from the idea card."""
        lines = [
            "## Idea Box Capture",
            "",
            f"**Summary:** {card.summary}",
            "",
            f"**Tags:** {', '.join(card.tags) if card.tags else '(none)'}",
            "",
            f"**Source:** {card.source.platform}/{card.source.channel_name}",
            f"**Message ID:** {card.source.message_id}",
            f"**User:** {card.source.user_name} ({card.source.user_id})",
            f"**Captured at:** {card.created_at}",
        ]
        if card.dedup_matches:
            lines.append("")
            lines.append("**Dedup matches (noted but not blocking):**")
            for m in card.dedup_matches:
                lines.append(f"  - {m.get('source', '?')}: {m.get('title', '')}")
        lines.append("")
        lines.append("Created via `/idea` command in #idea-box.")
        return "\n".join(lines)

    def _create_kanban_task(
        self,
        *,
        title: str,
        body: str,
        assignee: str,
        triage: bool,
    ) -> str:
        """Create a Kanban triage task.

        Uses the kanban_create tool handler when available, or the
        kanban_db API directly as a fallback.
        """
        if self._kanban_create_fn is not None:
            result = self._kanban_create_fn(
                title=title,
                body=body,
                assignee=assignee,
                triage=triage,
            )
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                return result.get("task_id", "")
            return str(result)

        # Fallback: use kanban_db directly
        try:
            from hermes_cli import kanban_db as kb
            conn = kb.connect()
            try:
                new_id = kb.create_task(
                    conn,
                    title=title,
                    body=body,
                    assignee=assignee,
                    triage=triage,
                    created_by="idea-box",
                )
            finally:
                conn.close()
            return new_id
        except Exception as e:
            logger.error("Kanban task creation failed: %s", e)
            raise

    def _log_to_wiki(self, card: IdeaCard, *, action: str) -> Optional[str]:
        """Log an idea (confirmed or rejected) to the LLM Wiki.

        Returns the wiki file path, or None if the wiki is unavailable.
        """
        if self._wiki_log_fn is not None:
            return self._wiki_log_fn(card=card, action=action)

        # Fallback: write to the wiki directory directly
        try:
            wiki_path = Path(os.environ.get("WIKI_PATH", str(Path.home() / "docs" / "wiki")))
            wiki_path.mkdir(parents=True, exist_ok=True)

            # Write to raw/ideas/ as provenance
            ideas_dir = wiki_path / "raw" / "ideas"
            ideas_dir.mkdir(parents=True, exist_ok=True)

            safe_id = (card.idea_id or card.source.message_id or "rejected").replace("/", "_")
            filename = f"{action}-{safe_id}.md"
            filepath = ideas_dir / filename

            lines = [
                f"# Idea: {card.summary[:80]}",
                "",
                f"**Status:** {action}",
                f"**Captured:** {card.created_at}",
                f"**Source:** {card.source.platform}/{card.source.channel_name} (msg {card.source.message_id})",
                f"**User:** {card.source.user_name}",
                f"**Tags:** {', '.join(card.tags) if card.tags else '(none)'}",
            ]
            if action == "rejected":
                lines.append(f"**Rejection reason:** {card.rejection_reason}")
            if card.idea_id:
                lines.append(f"**Kanban task:** {card.idea_id}")
            lines.append("")
            lines.append(f"## Full text")
            lines.append("")
            lines.append(card.summary)
            lines.append("")
            if card.dedup_matches:
                lines.append("## Dedup matches")
                for m in card.dedup_matches:
                    lines.append(f"- {m.get('source', '?')}: {m.get('title', '')}")
                lines.append("")

            filepath.write_text("\n".join(lines), encoding="utf-8")
            return str(filepath)
        except Exception as e:
            logger.warning("Wiki logging failed: %s", e)
            return None