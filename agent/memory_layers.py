"""Layered memory routing and lightweight retrieval helpers.

This module implements Hermes' memory taxonomy without turning memory into a
flat vector junk drawer.  It deliberately separates:

- curated always-on memory (tiny MEMORY.md / USER.md entries)
- procedural memory (skills)
- canonical domain stores (Obsidian, Drive, repos)
- episodic recall (session_search / transcripts)
- optional semantic recall (MemPalace CLI, local-first when installed)
- compression hygiene (Caveman-style removal of padding, not fact rewriting)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)


class MemoryDestination(str, Enum):
    """Canonical shelf for a piece of information."""

    CURATED_MEMORY = "curated_memory"
    CURATED_USER = "curated_user"
    SKILL = "skill"
    DOMAIN_STORE = "domain_store"
    SESSION_SEARCH = "session_search"
    SEMANTIC_RECALL = "semantic_recall"
    NONE = "none"


class MemoryKind(str, Enum):
    """High-level type of memory candidate."""

    USER_PREFERENCE = "user_preference"
    ENVIRONMENT_FACT = "environment_fact"
    PROCEDURE = "procedure"
    ARTIFACT = "artifact"
    EPISODIC = "episodic"
    EPHEMERAL_PROGRESS = "ephemeral_progress"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MemoryRouteDecision:
    kind: MemoryKind
    primary: MemoryDestination
    action: str
    reason: str
    domain: str = ""
    secondary: tuple[MemoryDestination, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["primary"] = self.primary.value
        data["secondary"] = [d.value for d in self.secondary]
        return data


class LayeredMemoryRouter:
    """Route memory candidates to the right durable shelf.

    The router is intentionally heuristic and conservative.  It does not write
    anything by itself; it returns a decision so the agent/tool can choose the
    correct Hermes mechanism (memory tool, skill_manage, Obsidian/Drive file,
    session_search, or skip).
    """

    _PREFERENCE_RE = re.compile(
        r"\b(user|nate|princebark)\b.*\b(prefers?|likes?|dislikes?|expects?|wants?|prioriti[sz]es?)\b|"
        r"\b(prefers?|likes?|dislikes?|expects?|wants?|prioriti[sz]es?)\b",
        re.IGNORECASE,
    )
    _PROCEDURE_RE = re.compile(
        r"\b(workflow|procedure|process|steps?|when .* then|use when|run .* before|"
        r"how we processed|filing workflow|classif(?:y|ied)|recipe filing|deploy|smoke[- ]test)\b",
        re.IGNORECASE,
    )
    _ARTIFACT_RE = re.compile(
        r"https?://|\b(file this|save this|store this|receipt|recipe|document|pdf|obsidian|google drive|drive|vault|repo)\b",
        re.IGNORECASE,
    )
    _EPHEMERAL_RE = re.compile(
        r"\b(PR\s*#?\d+|issue\s*#?\d+|commit\s+[0-9a-f]{7,40}|phase\s+\d+\s+(done|complete)|merged|submitted|fixed bug|temporary todo)\b",
        re.IGNORECASE,
    )
    _ENV_RE = re.compile(
        r"\b(path is|repo is|workspace is|project uses|tool quirk|installed|configured|host is|port \d+)\b",
        re.IGNORECASE,
    )

    def route(self, content: str, *, metadata: Optional[Mapping[str, Any]] = None) -> MemoryRouteDecision:
        text = (content or "").strip()
        meta = dict(metadata or {})
        lower = text.lower()

        if not text:
            return MemoryRouteDecision(
                kind=MemoryKind.UNKNOWN,
                primary=MemoryDestination.NONE,
                action="skip",
                reason="Empty content has no durable memory value.",
            )

        if self._EPHEMERAL_RE.search(text):
            return MemoryRouteDecision(
                kind=MemoryKind.EPHEMERAL_PROGRESS,
                primary=MemoryDestination.NONE,
                action="skip",
                reason="Task progress, PRs, issues, commits, and phase status become stale; rely on session_search/git instead.",
            )

        if self._PROCEDURE_RE.search(text):
            domain = self._domain_for(text, meta)
            return MemoryRouteDecision(
                kind=MemoryKind.PROCEDURE,
                primary=MemoryDestination.SKILL,
                action="create_or_update_skill",
                reason="Reusable workflow/procedure belongs in procedural memory, not always-on curated memory.",
                domain=domain,
                secondary=(MemoryDestination.SESSION_SEARCH,),
            )

        # Explicit user preferences about where/how artifacts should be kept are
        # preferences, not the artifact itself.  A URL or "file this" request is
        # still routed below to the domain store.
        if self._PREFERENCE_RE.search(text) and not re.search(r"https?://|\bfile this\b|\bsave this\b|\bstore this\b", text, re.IGNORECASE):
            return MemoryRouteDecision(
                kind=MemoryKind.USER_PREFERENCE,
                primary=MemoryDestination.CURATED_USER,
                action="remember",
                reason="Stable user preference/expectation that should influence future sessions.",
            )

        if self._ARTIFACT_RE.search(text):
            domain = self._domain_for(text, meta)
            return MemoryRouteDecision(
                kind=MemoryKind.ARTIFACT,
                primary=MemoryDestination.DOMAIN_STORE,
                action="store_canonical_artifact",
                reason="Artifacts should live in their canonical domain store with memory containing only stable pointers if needed.",
                domain=domain,
                secondary=(MemoryDestination.SESSION_SEARCH, MemoryDestination.SEMANTIC_RECALL),
            )

        if self._ENV_RE.search(text):
            return MemoryRouteDecision(
                kind=MemoryKind.ENVIRONMENT_FACT,
                primary=MemoryDestination.CURATED_MEMORY,
                action="remember",
                reason="Stable environment or project convention suitable for compact curated memory.",
            )

        if meta.get("source") == "conversation" or "we discussed" in lower:
            return MemoryRouteDecision(
                kind=MemoryKind.EPISODIC,
                primary=MemoryDestination.SESSION_SEARCH,
                action="recall_on_demand",
                reason="Episodic conversation facts are already in transcripts; use session_search unless promoted.",
                secondary=(MemoryDestination.SEMANTIC_RECALL,),
            )

        return MemoryRouteDecision(
            kind=MemoryKind.UNKNOWN,
            primary=MemoryDestination.SESSION_SEARCH,
            action="recall_on_demand",
            reason="No clear durable routing signal; avoid bloating curated memory.",
        )

    @staticmethod
    def _domain_for(text: str, metadata: Mapping[str, Any]) -> str:
        if metadata.get("domain"):
            return str(metadata["domain"])
        lower = text.lower()
        if "recipe" in lower or "obsidian" in lower or "vault" in lower:
            return "obsidian"
        if "receipt" in lower or "google drive" in lower or "drive" in lower:
            return "google_drive"
        if "repo" in lower or "commit" in lower or "pr" in lower:
            return "repository"
        return "general"


class CavemanCompressor:
    """Tiny deterministic compression pass for memory/skill hygiene.

    This is not a summarizer and does not rewrite facts.  It removes common
    assistant padding, collapses whitespace, and optionally enforces a char cap
    by cutting at a sentence boundary.  Use for memory entries, skill snippets,
    and context summaries — not for changing the assistant's main voice.
    """

    DEFAULT_PADDING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bGreat question!?\s*",
            r"\bI'd be happy to help\.?\s*",
            r"\bI(?:'|’)ll (?:go ahead and )?\b",
            r"\bLet me (?:go ahead and )?\b",
            r"\bIt is important to note that\s*",
            r"\bIn conclusion,\s*",
            r"\bTo summarize,\s*",
            r"\bAs an AI(?: language model)?,?\s*",
        )
    )

    def __init__(self, max_chars: int | None = None) -> None:
        self.max_chars = max_chars

    def compress(self, text: str, *, max_chars: int | None = None) -> str:
        out = text or ""
        for pattern in self.DEFAULT_PADDING_PATTERNS:
            out = pattern.sub("", out)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = "\n".join(line.strip() for line in out.strip().splitlines())
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)

        limit = max_chars if max_chars is not None else self.max_chars
        if limit and len(out) > limit:
            out = self._truncate_sentence(out, limit)
        return out

    @staticmethod
    def _truncate_sentence(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        cut = text[: max(0, limit - 1)].rstrip()
        sentence_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("\n"))
        if sentence_end >= max(40, int(limit * 0.45)):
            cut = cut[: sentence_end + 1].rstrip()
        return cut.rstrip(" ,;:") + "…"


class MemPalaceAdapter:
    """Optional CLI adapter for local-first MemPalace semantic recall.

    Hermes does not import MemPalace as a hard dependency.  If users install it
    (for example with ``uv tool install mempalace``), this adapter can query it
    via CLI.  When disabled or unavailable it silently returns no context.
    """

    def __init__(
        self,
        *,
        binary: str = "mempalace",
        scope: str = "",
        enabled: bool = False,
        timeout: float = 8.0,
        cwd: str | Path | None = None,
    ) -> None:
        self.binary = binary or "mempalace"
        self.scope = scope or ""
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self.cwd = Path(cwd) if cwd else None

    def build_search_command(self, query: str, *, limit: int = 5) -> list[str]:
        cmd = [self.binary, "search", query, "--limit", str(int(limit))]
        if self.scope:
            cmd.extend(["--scope", self.scope])
        return cmd

    def prefetch(self, query: str, *, limit: int = 5) -> str:
        if not self.enabled or not query or not query.strip():
            return ""
        cmd = self.build_search_command(query, limit=limit)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.cwd) if self.cwd else None,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("MemPalace prefetch unavailable: %s", exc)
            return ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("MemPalace prefetch failed: %s", exc)
            return ""

        if proc.returncode != 0:
            logger.debug("MemPalace search exited %s: %s", proc.returncode, proc.stderr[:300])
            return ""
        output = (proc.stdout or "").strip()
        if not output:
            return ""
        return "## MemPalace Semantic Recall\n" + output


def decision_to_json(decision: MemoryRouteDecision, *, compressed_content: str = "") -> str:
    payload: dict[str, Any] = {"success": True, "decision": decision.to_dict()}
    if compressed_content:
        payload["compressed_content"] = compressed_content
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "CavemanCompressor",
    "LayeredMemoryRouter",
    "MemPalaceAdapter",
    "MemoryDestination",
    "MemoryKind",
    "MemoryRouteDecision",
    "decision_to_json",
]
