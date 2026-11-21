"""LLM-driven SKILL.md mutation for the AIDE² self-improvement loop.

Phase 4 of the AIDE² self-evaluation plan (see
``docs/aide-squared-roadmap.md``). Phase 1 stubbed
``HermesSquaredEngine._apply_mutation``; Phase 4 wires it to an
LLM-driven mutator that generates the new SKILL.md from the current
content plus the skill's evaluation summary.

Design
------

Two protocols decouple mutation generation from mutation application:

- :class:`SkillMuter` — given a ``MutationContext``, returns a
  :class:`MutationProposal` containing the new content. The mutator
  is the LLM-bound part: ``DefaultSkillMuter`` calls
  :mod:`agent.auxiliary_client.call_llm` with a structured prompt
  and parses the response.
- :class:`SkillMuterApplier` — takes a ``MutationProposal`` and
  performs the file I/O: backup current SKILL.md to
  ``SKILL.md.bak``, write the new content, and roll back on
  failure. The applier is the side-effect-bound part: it owns the
  backup/restore dance and never invokes any LLM.

This split lets tests inject either side independently. The mutator
can be a fake that returns canned proposals; the applier can be
exercised against a real ``tmp_path`` without LLM calls.

The mutator emits a full replacement SKILL.md rather than a diff
because:

1. LLM-generated diffs are notoriously unreliable (off-by-one,
   context mismatches). A full file is the most verifiable form.
2. The ``file_ops`` tool in Hermes does not currently expose a
   unified-diff entry point — promoting diff support is out of
   scope for this PR.
3. Atomicity is trivial with a full file: backup + write.

The applier therefore does the simplest possible thing: backup,
write, optional rollback. No diff application.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationContext:
    """All inputs the mutator needs to generate a new SKILL.md.

    Attributes:
        skill_id: Identifier of the skill being mutated. Used in
            the prompt for the LLM.
        current_content: Full text of the current SKILL.md.
        strategy: One of the strategy strings emitted by
            ``HermesSquaredEngine._generate_proposal``:
            ``add_validation``, ``rewrite_skill``,
            ``fundamental_rewrite``, ``optimize``.
        private_score: Average private score from the ledger.
        public_score: Average public score from the ledger.
        correction_rate: User-correction rate (0-1).
        success_rate: Skill success rate (0-1).
        notes: Free-form context (e.g. "reward hacking suspected",
            "low success rate").
        model_kwargs: Forwarded to ``call_llm`` — same shape as
            ``EvalInvocation.model_kwargs``. Use to pin a model.
    """

    skill_id: str
    current_content: str
    strategy: str
    private_score: float = 0.0
    public_score: float = 0.0
    correction_rate: float = 0.0
    success_rate: float = 0.0
    notes: str = ""
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MutationProposal:
    """The mutator's output: a new SKILL.md plus diagnostic info.

    Attributes:
        new_content: The full new SKILL.md content. Empty string on
            failure (caller should treat as no-mutation).
        reasoning: One-paragraph explanation of what changed. May be
            empty.
        success: True iff a usable ``new_content`` was produced.
        error: Human-readable error description on failure.
        model: The model identifier that produced the proposal.
            None on failure.
    """

    new_content: str
    reasoning: str = ""
    success: bool = False
    error: Optional[str] = None
    model: Optional[str] = None


@dataclass(frozen=True)
class ApplyResult:
    """The applier's output: did the file write succeed?

    Attributes:
        success: True iff the new content was written and the
            backup was created.
        backup_path: Path to the backup file (``SKILL.md.bak``), or
            None on failure / when the applier chose not to back up.
        error: Human-readable error description on failure. None
            on success.
    """

    success: bool
    backup_path: Optional[Path] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class SkillMuter(Protocol):
    """Generates a new SKILL.md from a MutationContext."""

    def mutate(self, context: MutationContext) -> MutationProposal: ...


class SkillMuterApplier(Protocol):
    """Persists a MutationProposal to disk with backup / rollback."""

    def apply(
        self,
        skill_id: str,
        proposal: MutationProposal,
        *,
        hermes_home: Path,
    ) -> ApplyResult: ...

    def rollback(self, skill_id: str, *, hermes_home: Path) -> bool: ...


# ---------------------------------------------------------------------------
# Default mutator
# ---------------------------------------------------------------------------


DEFAULT_MUTATION_PROMPT_TEMPLATE = """\
You are rewriting a Hermes SKILL.md file based on its evaluation history.

Skill ID: {skill_id}

Evaluation summary:
- Private score (objective signal, 0-1): {private_score}
- Public score (agent self-rating, 0-1): {public_score}
- User-correction rate (0-1): {correction_rate}
- Success rate (0-1): {success_rate}
- Diagnosis: {strategy} ({notes})

Current SKILL.md:
<current_skill_md>
{current_content}
</current_skill_md>

Produce a NEW version of the SKILL.md that addresses the diagnosis.

Constraints:
1. Output ONLY the new file content. No commentary, no code fences.
2. Preserve the skill's purpose and core workflow.
3. Address the diagnosis explicitly (add validation steps if
   ``add_validation``; improve clarity if ``rewrite_skill``;
   restructure for reliability if ``fundamental_rewrite``;
   optimize for speed/cost if ``optimize``).
4. Keep the file concise. If you must add content, add it as a
   new section at the end.
"""


class DefaultSkillMuter:
    """LLM-driven mutator that uses auxiliary_client.call_llm."""

    def __init__(
        self,
        *,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        timeout_sec: float = 120.0,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.model_kwargs = dict(model_kwargs or {})
        self.timeout_sec = timeout_sec
        self.prompt_template = prompt_template or DEFAULT_MUTATION_PROMPT_TEMPLATE

    def mutate(self, context: MutationContext) -> MutationProposal:
        """Call the LLM and parse the response into a MutationProposal."""
        try:
            from agent.auxiliary_client import call_llm
        except ImportError as e:
            return MutationProposal(
                new_content="",
                success=False,
                error=f"auxiliary_client unavailable: {e}",
            )

        prompt = self.prompt_template.format(
            skill_id=context.skill_id,
            private_score=f"{context.private_score:.3f}",
            public_score=f"{context.public_score:.3f}",
            correction_rate=f"{context.correction_rate:.3f}",
            success_rate=f"{context.success_rate:.3f}",
            strategy=context.strategy,
            notes=context.notes or "(none)",
            current_content=context.current_content,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = call_llm(
                messages=messages,
                timeout=self.timeout_sec,
                **self.model_kwargs,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "DefaultSkillMuter.mutate: call_llm failed for %s: %s",
                context.skill_id,
                e,
            )
            return MutationProposal(new_content="", success=False, error=str(e))

        text = _extract_text(response)
        model_id = getattr(response, "model", None)
        new_content, reasoning = parse_mutation_response(text)
        if new_content is None or not new_content.strip():
            return MutationProposal(
                new_content="",
                reasoning=reasoning[:500],
                success=False,
                error="LLM response did not contain a parseable SKILL.md",
                model=model_id,
            )
        return MutationProposal(
            new_content=new_content,
            reasoning=reasoning,
            success=True,
            model=model_id,
        )


# ---------------------------------------------------------------------------
# Default applier
# ---------------------------------------------------------------------------


class DefaultSkillMuterApplier:
    """Backup + atomic write + rollback.

    Filesystem layout::

        {hermes_home}/skills/<skill_id>/SKILL.md        # current
        {hermes_home}/skills/<skill_id>/SKILL.md.bak    # rollback only (deleted on success)

    On apply:
      1. If SKILL.md exists, copy it to SKILL.md.bak.
      2. Write the new content to a temp file, then os.replace() it
         atomically over SKILL.md.
      3. On success: delete SKILL.md.bak (prevents stale-rollback Bug 3).
         On failure: leave bak in place so rollback can restore.

    On rollback:
      - If SKILL.md.bak exists, overwrite SKILL.md with its contents.
        Returns True on success, False if no backup exists.
      - Rollback only restores from the LAST failed apply's backup; a
        successful apply always deletes the backup so no stale content
        can be resurrected.

    The applier never throws on failure — it surfaces errors via
    ``ApplyResult.error``. Callers can choose to log + ignore.
    """

    BACKUP_FILENAME = "SKILL.md.bak"

    def apply(
        self,
        skill_id: str,
        proposal: MutationProposal,
        *,
        hermes_home: Path,
    ) -> ApplyResult:
        if not proposal.success or not proposal.new_content.strip():
            return ApplyResult(
                success=False,
                error=(
                    proposal.error
                    or "cannot apply: proposal is unsuccessful or has empty content"
                ),
            )

        skill_file = self._skill_file(skill_id, hermes_home)
        backup_path = self._backup_path(skill_id, hermes_home)
        try:
            skill_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ApplyResult(success=False, error=f"mkdir failed: {e}")

        # Step 1: backup (if SKILL.md exists).
        backup_created = False
        if skill_file.exists():
            try:
                shutil.copy2(skill_file, backup_path)
                backup_created = True
            except OSError as e:
                return ApplyResult(success=False, error=f"backup failed: {e}")

        # Step 2: atomic write via temp file + os.replace().
        # This prevents concurrent readers from seeing partial content
        # and prevents truncate-on-write data loss (fixes Bug 2 partial-write).
        tmp_path = skill_file.with_suffix(".tmp")
        try:
            tmp_path.write_text(proposal.new_content, encoding="utf-8")
            os.replace(tmp_path, skill_file)  # atomic on POSIX; fallback on Windows
        except OSError as e:
            # Clean up the temp file if it exists.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Roll back if we created a backup.
            if backup_created:
                try:
                    shutil.copy2(backup_path, skill_file)
                except OSError:
                    pass  # Best-effort rollback.
            return ApplyResult(
                success=False,
                backup_path=backup_path if backup_created else None,
                error=f"write failed: {e}",
            )

        # Step 3: success — delete the backup so it cannot be used to
        # resurrect stale content later (fixes Bug 3 stale-rollback).
        if backup_created:
            try:
                backup_path.unlink()
            except OSError:
                pass  # Best-effort; backup is now stale anyway.

        return ApplyResult(success=True, backup_path=None)

    def rollback(self, skill_id: str, *, hermes_home: Path) -> bool:
        skill_file = self._skill_file(skill_id, hermes_home)
        backup_path = self._backup_path(skill_id, hermes_home)
        if not backup_path.exists():
            return False
        try:
            shutil.copy2(backup_path, skill_file)
            return True
        except OSError:
            return False

    @staticmethod
    def _skill_file(skill_id: str, hermes_home: Path) -> Path:
        return hermes_home / "skills" / skill_id / "SKILL.md"

    @staticmethod
    def _backup_path(skill_id: str, hermes_home: Path) -> Path:
        return (
            hermes_home / "skills" / skill_id / DefaultSkillMuterApplier.BACKUP_FILENAME
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_mutation_prompt(context: MutationContext) -> str:
    """Build the default mutation prompt. Exposed for tests +
    downstream customizations.
    """
    return DEFAULT_MUTATION_PROMPT_TEMPLATE.format(
        skill_id=context.skill_id,
        private_score=f"{context.private_score:.3f}",
        public_score=f"{context.public_score:.3f}",
        correction_rate=f"{context.correction_rate:.3f}",
        success_rate=f"{context.success_rate:.3f}",
        strategy=context.strategy,
        notes=context.notes or "(none)",
        current_content=context.current_content,
    )


def parse_mutation_response(text: str) -> tuple[Optional[str], str]:
    """Extract ``(new_content, reasoning)`` from the LLM response.

    The default prompt instructs the model to emit raw content with
    no commentary. Real models occasionally violate this, so we
    handle three common patterns:

    1. Raw content (the ideal case).
    2. ``` fenced block with optional language hint.
    3. A trailing ``<reasoning>...</reasoning>`` section that we
       strip out and return separately.

    Returns ``(None, "")`` if the response is empty.
    """
    if not text or not text.strip():
        return (None, "")

    # Strip ``` fenced blocks first.
    cleaned = re.sub(r"```(?:[a-zA-Z0-9_+-]*)\s*\n?", "", text)
    cleaned = cleaned.replace("```", "")

    # If the model added a trailing reasoning section, split.
    # Use DOTALL so . matches newline; require the </reasoning> tag
    # to be at the VERY end of the string (possibly with trailing
    # whitespace) so that any content BEFORE the tag is preserved.
    # Bug 5 fix: multiline $ would match before a trailing newline,
    # causing valid content to be silently discarded.
    reasoning = ""
    reason_match = re.search(
        r"(?ims)^\s*<reasoning>\s*(.*?)\s*</reasoning>\s*$", cleaned, flags=re.DOTALL
    )
    if reason_match:
        reasoning = reason_match.group(1).strip()
        before = cleaned[: reason_match.start()]
        # Require that content exists BEFORE the reasoning tag and
        # is not just whitespace. If there is real content, keep it.
        if before.strip():
            cleaned = before.rstrip()
        else:
            cleaned = ""

    # Strip a leading preamble if the model added one.
    # Bug 6 fix: a valid skill file must contain a heading (#) or code fence (```).
    # Single-line/no-newline preambles like "Here is the new SKILL.md:"
    # are rejected entirely. Multi-line preambles are stripped only when
    # a real heading/fence marker appears AFTER the preamble.
    first_newline = cleaned.find("\n")
    if first_newline == -1:
        # No newline at all — plain text, no heading/fence marker.
        # This is a single-line preamble like "Here is the new SKILL.md:".
        first_line = cleaned.strip()
        has_marker = "#" in cleaned or "```" in cleaned
        if not has_marker and first_line and len(first_line) < 200:
            # No heading, no fence, single line — not a valid skill.
            return (None, reasoning)
        # Otherwise keep as-is (might be plain-text skill content).
    else:
        # Has newlines — check if a heading or fence appears after line 1.
        after_first = cleaned[first_newline:]
        marker_match = re.search(r"\n(#|\```)", after_first)
        if marker_match:
            # A heading/fence appears after the preamble — strip preamble.
            # Only strip short preambles (genuine prose, not real content).
            preamble = cleaned[: first_newline + marker_match.start()]
            if len(preamble.strip()) < 200:
                cleaned = after_first[marker_match.start() :]


    new_content = cleaned.strip()
    if not new_content:
        return (None, reasoning)
    return (new_content, reasoning)


def _extract_text(response: Any) -> str:
    """Pull the textual content from a call_llm response, tolerating
    both object-style and dict-style responses.
    """
    try:
        choices = getattr(response, "choices", None)
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
    except (AttributeError, IndexError):
        pass
    if isinstance(response, Mapping):
        choices = response.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


__all__ = [
    "MutationContext",
    "MutationProposal",
    "ApplyResult",
    "SkillMuter",
    "SkillMuterApplier",
    "DefaultSkillMuter",
    "DefaultSkillMuterApplier",
    "DEFAULT_MUTATION_PROMPT_TEMPLATE",
    "build_mutation_prompt",
    "parse_mutation_response",
]
