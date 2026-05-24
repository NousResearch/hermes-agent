"""Review Inbox — manages the knowledge review queue using Card Store.

Replaces the old JSON-based review queue with Card Store backed knowledge
cards in ``review_status='pending_review'`` state.

Features:
- Priority scoring: relevance × urgency × (1 - duplicate_penalty)
- Auto-duplicate detection on queue entry
- Reject → archive to review-queue/rejected/ in the restored Obsidian graph
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.card_store import CardStore, KNOWLEDGE_REVIEW_STATUSES
from agent.knowledge_relevance import KnowledgeRelevanceEngine

logger = logging.getLogger(__name__)


class ReviewInbox:
    """Manages the knowledge review queue via Card Store."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        if vault_path is None:
            vault_path = Path(os.environ.get("OBSIDIAN_VAULT_PATH", ""))
            if not vault_path.exists():
                vault_path = Path.home() / "ObsidianVault" / "HermesAgent"
        self.vault_path = Path(vault_path)
        self.store = CardStore()
        self.relevance_engine = KnowledgeRelevanceEngine(vault_path=self.vault_path)

    def queue_knowledge(
        self,
        title: str,
        body: str,
        source: str = "",
        domains: Optional[List[str]] = None,
        origin_project: str = "",
        summary: str = "",
    ) -> Dict[str, Any]:
        """Add knowledge to the review queue.

        Auto-detects duplicates against pending entries.

        Returns:
            Dict with card_id, status, and optional duplicate_candidates.
        """
        # Check for existing duplicates in the pending queue
        pending = self.store.list_knowledge_cards(
            review_status="pending_review", limit=200
        )[0]
        duplicate_candidates = []
        for card in pending:
            existing_words = set(_tokenize(card.get("body", "")))
            existing_title = set(_tokenize(card.get("title", "")))
            new_words = set(_tokenize(body))
            new_title_words = set(_tokenize(title))
            all_words = new_words | new_title_words | existing_words | existing_title
            if not all_words:
                continue
            overlap = (new_words & existing_words) | (new_title_words & existing_title)
            similarity = len(overlap) / len(all_words) if all_words else 0
            if similarity >= 0.8:
                duplicate_candidates.append({
                    "card_id": card["id"],
                    "title": card["title"],
                    "similarity": round(similarity, 3),
                })

        # Create as pending_review card
        card_id = self.store.create_knowledge_card(
            title=title,
            body=body,
            source=source,
            domains=domains,
            origin_project=origin_project,
            review_status="pending_review",
            project_fit=self._estimate_project_fit(domains),
        )

        # If duplicates found, mark this card
        if duplicate_candidates:
            self.store.update_knowledge_card(
                card_id,
                review_status="duplicate",
                duplicate_of=duplicate_candidates[0]["card_id"],
            )
            logger.info(
                "ReviewInbox: queued '%s' as duplicate of %s (similarity=%.2f)",
                title, duplicate_candidates[0]["card_id"],
                duplicate_candidates[0]["similarity"],
            )
        else:
            logger.info("ReviewInbox: queued '%s' (%s)", title, card_id)

        result = {
            "success": True,
            "card_id": card_id,
            "status": "pending_review",
        }
        if duplicate_candidates:
            result["duplicate_candidates"] = duplicate_candidates
        return result

    def list_pending(self) -> List[Dict[str, Any]]:
        """List all pending review items with priority scores."""
        cards, _ = self.store.list_knowledge_cards(
            review_status="pending_review", limit=500
        )
        for card in cards:
            card["priority_score"] = self.get_priority_score(card)
        # Sort by priority descending
        cards.sort(key=lambda c: c.get("priority_score", 0), reverse=True)
        return cards

    def approve(self, card_id: str) -> Dict[str, Any]:
        """Approve a knowledge card → status=approved."""
        ok = self.store.update_knowledge_card(card_id, review_status="approved")
        if ok:
            logger.info("ReviewInbox: approved %s", card_id)
        return {"success": ok, "action": "approve", "card_id": card_id}

    def reject(self, card_id: str, reason: str = "") -> Dict[str, Any]:
        """Reject a knowledge card → status=rejected, archive to .rejected/."""
        card = self.store.get_knowledge_card(card_id)
        if not card:
            return {"success": False, "error": f"Card {card_id} not found"}

        ok = self.store.update_knowledge_card(
            card_id, review_status="rejected"
        )

        # Archive to vault
        archived_path = self._archive_rejected(card, reason)

        logger.info("ReviewInbox: rejected %s → %s", card_id, archived_path or "no archive")
        return {
            "success": ok,
            "action": "reject",
            "card_id": card_id,
            "archived_path": archived_path,
        }

    def defer(self, card_id: str, reason: str = "") -> Dict[str, Any]:
        """Defer a knowledge card → status=deferred."""
        ok = self.store.update_knowledge_card(card_id, review_status="deferred")
        logger.info("ReviewInbox: deferred %s", card_id)
        return {"success": ok, "action": "defer", "card_id": card_id}

    def mark_duplicate(self, card_id: str, duplicate_of: str) -> Dict[str, Any]:
        """Mark a card as duplicate of another."""
        ok = self.store.update_knowledge_card(
            card_id, review_status="duplicate", duplicate_of=duplicate_of
        )
        logger.info("ReviewInbox: marked %s as duplicate of %s", card_id, duplicate_of)
        return {"success": ok, "action": "mark_duplicate", "card_id": card_id}

    def request_revision(self, card_id: str, feedback: str = "") -> Dict[str, Any]:
        """Request revision on a card → status=revision_requested."""
        ok = self.store.update_knowledge_card(card_id, review_status="revision_requested")
        logger.info("ReviewInbox: revision requested for %s", card_id)
        return {"success": ok, "action": "request_revision", "card_id": card_id}

    def get_priority_score(self, card: Dict[str, Any]) -> float:
        """Calculate priority score for a review card.

        Formula: relevance × urgency × (1 - duplicate_penalty)

        - relevance: cross-project relevance score (0-1)
        - urgency: days in queue factor (capped at 1.0)
        - duplicate_penalty: similarity to existing cards (0-1)
        """
        # Relevance score
        relevance = self._relevance_score(card)

        # Urgency: older = more urgent (capped at 30 days)
        try:
            created = datetime.strptime(card.get("created_at", ""), "%Y-%m-%d %H:%M:%S")
            age_days = (datetime.now() - created).days
            urgency = min(age_days / 30, 1.0)
        except (ValueError, TypeError):
            urgency = 0.5

        # Duplicate penalty
        duplicate_penalty = 0.0
        if card.get("duplicate_of"):
            duplicate_penalty = 0.5  # Already flagged as duplicate

        return round(relevance * urgency * (1 - duplicate_penalty), 3)

    # ── Internal helpers ─────────────────────────────────────────────

    def _relevance_score(self, card: Dict[str, Any]) -> float:
        """Estimate cross-project relevance for a card."""
        body = card.get("body", "")
        origin = card.get("origin_project", "")
        if not body or not origin:
            return 0.3  # Default moderate relevance
        try:
            is_relevant = self.relevance_engine.is_cross_project_relevant(body, origin)
            return 0.8 if is_relevant else 0.3
        except Exception:
            return 0.5

    def _estimate_project_fit(self, domains: Optional[List[str]]) -> float:
        """Estimate project fit based on domains."""
        if not domains:
            return 0.5
        # More domains = broader applicability = higher fit
        return min(0.5 + len(domains) * 0.1, 1.0)

    def _archive_rejected(self, card: Dict[str, Any], reason: str) -> Optional[str]:
        """Archive a rejected card to the Obsidian review queue."""
        try:
            rejected_dir = self.vault_path / "review-queue" / "rejected"
            rejected_dir.mkdir(parents=True, exist_ok=True)

            slug = _slugify(card.get("title", "unknown"))
            timestamp = datetime.now().strftime("%Y-%m-%d")
            filename = f"{slug}_{timestamp}.md"
            filepath = rejected_dir / filename

            # Handle duplicates
            counter = 1
            while filepath.exists():
                filename = f"{slug}_{timestamp}_{counter}.md"
                filepath = rejected_dir / filename
                counter += 1

            frontmatter = (
                f"---\n"
                f"title: {card.get('title', '')}\n"
                f"status: rejected\n"
                f"rejected_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"origin_project: {card.get('origin_project', '')}\n"
                f"domains: {json.dumps(card.get('domains', []))}\n"
                f"---\n\n"
            )
            if reason:
                frontmatter += f"> **Rejection reason:** {reason}\n\n"

            content = frontmatter + card.get("body", "")
            filepath.write_text(content, encoding="utf-8")
            return str(filepath)
        except Exception as e:
            logger.error("Failed to archive rejected card %s: %s", card.get("id"), e)
            return None


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for keyword overlap."""
    import re
    return re.findall(r'\b\w+\b', text.lower())


def _slugify(text: str) -> str:
    """Convert title to filename-safe slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-') or "untitled"
