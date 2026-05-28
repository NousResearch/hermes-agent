#!/usr/bin/env python3
"""
CONTEXT ORCHESTRATOR — Active context window management for Hermes Agent.

Handles three lifecycle phases:
  1. SESSION PREP  — Load identity + memory state into context window
  2. MID-SESSION   — Monitor usage, trim low-priority blocks, offload to memory
  3. SESSION END   — Persist important context, prune, generate summary

Priority tiers (highest to lowest, last trimmed first):
  T0 — Identity (context-architect.md, SOUL.md)
  T1 — Active task state (working memory, current goal)
  T2 — Recent high-importance episodes (last 24h, importance >= 5)
  T3 — Semantic facts (relevant to current task)
  T4 — Recent low-importance episodes (older, importance < 5)
  T5 — Tool output history (trimmed aggressively)
  T6 — Conversation history (oldest first to trim)

Refactored: module-level globals replaced with ContextOrchestrator class
to support concurrent multi-session operation in the gateway.
"""

import os, sys, json, time, hashlib, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Resolve the module path to find memory_palace and other co-located modules.
# Works whether context_orchestrator.py lives in scripts/ or at the pipeline root.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    _this_dir,                          # same directory (pipeline root on Linux)
    os.path.join(_this_dir, 'scripts'),  # scripts/ subdir (Mac staging)
]
for _mp in _candidates:
    if os.path.isdir(_mp):
        if _mp not in sys.path:
            sys.path.insert(0, _mp)
            break

from memory_palace import (
    store_episode, recall_episodes,
    store_fact, recall_facts,
    set_working, get_working, clear_working,
    get_stats, prune_expired, auto_prune,
)

# ── Paths ──────────────────────────────────────────────────────
HERMES_HOME = Path(os.path.expanduser("~/.hermes"))
ARCHITECT_FILE = HERMES_HOME / "context-architect.md"
SOUL_FILE = HERMES_HOME / "SOUL.md"
STATE_OF_AFFAIRS = HERMES_HOME / "state_of_affairs.md"
MEMORY_DB = HERMES_HOME / "memory-palace" / "palace.db"
LOG_DIR = HERMES_HOME / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Context budget (fits within typical 8K–16K windows)
BUDGET_TOKENS = 14000         # hard ceiling before aggressive trim (raised from 12K)
WARNING_TOKENS = 10000        # soft warning, start light trim
TARGET_POST_TRIM = 5000       # target after heavy trim (lowered from 6K)
EST_TOKENS_PER_CHAR = 0.25    # rough: 1 char ≈ 0.25 tokens

logger = logging.getLogger("context_orchestrator")


class ContextOrchestrator:
    """
    Per-session context lifecycle manager.

    Each gateway session gets its own instance, keyed by session_key.
    This eliminates the module-level globals that previously broke
    concurrent multi-session operation.

    Trim-suppression mechanisms:
      - pause_trimming(reason) / resume_trimming(): session-wide toggle.
        Auto-resumes after MAX_PAUSE_DURATION (default 3600s) to prevent
        unbounded memory growth.
      - set_block_protection(block_id, protected): per-block flag that
        makes a specific block survive any trim regardless of tier.
    """

    # Safety ceiling — trim suppression cannot last longer than this.
    MAX_PAUSE_DURATION = 3600  # seconds (1 hour)

    # Model classification for trim strategy (class-level, shared across instances)
    _PREMIUM_INDICATORS = [
        "deepseek-v4-pro", "deepseek-v4-flash",
        "ring-2.6-1t", "ring-2",
        "claude-sonnet", "claude-4",
        "gpt-4", "gpt-oss-120b", "gpt-oss-20b",
        "glm-4.7",
        "qwen3-coder-30b", "qwen3-235b", "qwen3-480b",
        "kimi-k2.5", "minimax-m2.1",
    ]

    def __init__(self, session_key: str):
        self.session_key = session_key
        self._active_blocks: list[dict] = []
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # ── Trim-suppression state ──────────────────────────────────
        self._trimming_paused: bool = False
        self._pause_reason: str | None = None
        self._pause_start_time: float | None = None
        self._protected_blocks: set[str] = set()

        # ── Auto-sculpt state ────────────────────────────────────
        self._turn_count: int = 0
        self.MAX_TURNS_BEFORE_SCULPT: int = int(
            os.environ.get("MAX_TURNS_BEFORE_SCULPT", "20")
        )

        # ── Tuning observation log ───────────────────────────────
        self._tuning_log: list[dict] = []

    # ── Pause / Resume / Protect API ─────────────────────────────

    def pause_trimming(self, reason: str = "") -> dict:
        """Pause all context trimming for this session."""
        self._trimming_paused = True
        self._pause_reason = reason or "manual"
        self._pause_start_time = time.time()
        logger.info("[%s] Trimming paused (reason: %s)", self.session_key, reason)
        return {"status": "paused", "reason": reason}

    def resume_trimming(self, reason: str = "") -> dict:
        """Resume context trimming for this session."""
        was_paused = self._trimming_paused
        self._trimming_paused = False
        self._pause_reason = None
        self._pause_start_time = None
        if was_paused:
            logger.info("[%s] Trimming resumed (reason: %s)", self.session_key, reason)
        return {"status": "resumed", "reason": reason}

    def set_block_protected(self, block_id: str, protected: bool = True) -> dict:
        """Mark a block as protected (skip trimming) or unprotect it."""
        if protected:
            self._protected_blocks.add(block_id)
        else:
            self._protected_blocks.discard(block_id)
        return {"block_id": block_id, "protected": protected}

    def is_paused(self) -> bool:
        return self._trimming_paused

    def get_pause_info(self) -> dict | None:
        if not self._trimming_paused:
            return None
        elapsed = 0.0
        if self._pause_start_time is not None:
            elapsed = time.time() - self._pause_start_time
        return {
            "reason": self._pause_reason,
            "elapsed_seconds": round(elapsed, 1),
            "auto_resume_in_seconds": max(0, self.MAX_PAUSE_DURATION - elapsed),
        }

    def log_tuning_observation(self, category: str, detail: str, **kwargs):
        """Record a tuning observation for later review and parameter adjustment.

        Stores structured observations during the session so we can track:
        - How often trimming fires and at what token counts
        - Compression ratios achieved
        - Turn counts between sculpts
        - Any anomalies (failed compression, memory pressure, etc.)

        Observations are persisted to the memory palace at session end.
        """
        entry = {
            "timestamp": time.time(),
            "turn": self._turn_count,
            "category": category,
            "detail": detail,
            "extra": kwargs,
        }
        self._tuning_log.append(entry)
        logger.info("[%s] Tuning [%s] turn=%d: %s %s",
                    self.session_key, category, self._turn_count, detail,
                    json.dumps(kwargs) if kwargs else "")

    def _check_sculpt_trigger(self, current_tokens: int) -> dict | None:
        """Check if we should auto-sculpt based on turn count or token pressure.

        Returns a sculpt result dict if triggered, None otherwise.

        Trigger conditions (any one is enough):
          1. Turn count >= MAX_TURNS_BEFORE_SCULPT
          2. Token usage is above 90% of budget and we've had 5+ turns
        """
        turned = self._turn_count >= self.MAX_TURNS_BEFORE_SCULPT
        pressured = (current_tokens > BUDGET_TOKENS * 0.9
                     and self._turn_count >= 5)

        if not turned and not pressured:
            return None

        reason = "turn_limit" if turned else "token_pressure"
        self.log_tuning_observation("sculpt_trigger", reason,
                                     turns=self._turn_count,
                                     tokens=current_tokens,
                                     budget=BUDGET_TOKENS)
        return self.sculpt_session(reason)

    def sculpt_session(self, reason: str = "manual") -> dict:
        """Sculpt the session: persist all context to memory palace and reset.

        This is the 'manual reset' — equivalent to ending and restarting
        but without clearing the turn counter. Keeps the user's conversation
        continuity while giving the model a clean context slate.
        """
        # Persist remaining blocks to memory palace before reset
        for block in self._active_blocks:
            if block.get("content") and block["tier"] <= 3:
                store_episode(
                    self._session_id, "sculpt_checkpoint",
                    block["content"][:500],
                    importance=4 if block["tier"] <= 1 else 2,
                    tags=[f"tier{block['tier']}", "sculpt", reason]
                )

        # Log sculpt event
        store_episode(
            self._session_id, "sculpt_event",
            f"Auto-sculpt triggered: {reason} | turns={self._turn_count} | "
            f"active_blocks={len(self._active_blocks)}",
            importance=6, tags=["sculpt", "checkpoint", reason]
        )

        # Clear active blocks and rebuild from fresh start
        self._active_blocks = []
        result = self.start_session(
            task="sculpt_reset",
            phase="post_sculpt"
        )

        logger.info("[%s] Session sculpted (reason=%s, saved %d blocks)",
                    self.session_key, reason, len(self._active_blocks))
        return {
            "sculpted": True,
            "reason": reason,
            "saved_blocks": len(self._active_blocks),
            "new_context_tokens": result.get("total_est_tokens", 0),
        }

    # ── Token estimation ────────────────────────────────────────

    @staticmethod
    def _est_tokens(text: str) -> int:
        """Rough token estimate."""
        return int(len(text) * EST_TOKENS_PER_CHAR)

    @staticmethod
    def _load_file(path: Path, max_chars: int = 8000) -> str:
        """Load file, capped to max_chars."""
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[:max_chars]

    # ── PHASE 1: SESSION PREP ──────────────────────────────────

    def start_session(self, task: str = "idle", phase: str = "startup") -> dict:
        """
        Called at session start. Loads prioritized context blocks.
        Returns the composed context string and metadata.
        """
        self._active_blocks = []

        # Run auto-prune on session start to keep DB bounded
        auto_prune()

        # Set working memory
        set_working("active_task", {"task": task, "phase": phase, "since": self._session_id})

        # Tier 0: Identity (always present, never trimmed)
        identity = self._load_file(ARCHITECT_FILE, 4000)
        soul = self._load_file(SOUL_FILE, 2000)
        state = self._load_file(STATE_OF_AFFAIRS, 2000)
        self._active_blocks.append({
            "id": "t0_identity", "tier": 0, "persist": True,
            "content": f"## IDENTITY & PURPOSE\n{identity}",
            "tokens": self._est_tokens(identity)
        })
        if soul.strip():
            self._active_blocks.append({
                "id": "t0_soul", "tier": 0, "persist": True,
                "content": f"## SOUL & OPERATING PRINCIPLES\n{soul}",
                "tokens": self._est_tokens(soul)
            })
        if state.strip():
            self._active_blocks.append({
                "id": "t0_state_of_affairs", "tier": 0, "persist": True,
                "content": f"## LIVE STATE OF AFFAIRS\n{state}",
                "tokens": self._est_tokens(state)
            })

        # Tier 1: Active task state
        working = get_working("active_task") or {}
        self._active_blocks.append({
            "id": "t1_task", "tier": 1, "persist": False,
            "content": (
                f"## CURRENT TASK\n"
                f"- Task: {working.get('task', task)}\n"
                f"- Phase: {working.get('phase', phase)}\n"
                f"- Since: {working.get('since', self._session_id)}"
            ),
            "tokens": 50  # small
        })

        # Tier 2: Recent high-importance episodes (last 24h, importance >= 5)
        recent = recall_episodes(hours=24, min_importance=5, limit=8)
        if recent:
            ep_text = "\n".join(
                f"- [{e.get('category','')}] {e.get('content','')[:120]}"
                for e in recent[:6]
            )
            self._active_blocks.append({
                "id": "t2_recent_high", "tier": 2, "persist": False,
                "content": f"## RECENT HIGHLIGHTS\n{ep_text}",
                "tokens": self._est_tokens(ep_text)
            })

        # Tier 3: Semantic facts relevant to task
        facts = recall_facts(task, limit=8)
        if facts:
            fact_text = "\n".join(
                f"- **{f.get('concept','')}**: {f.get('description','')[:150]}"
                for f in facts[:5]
            )
            self._active_blocks.append({
                "id": "t3_semantic", "tier": 3, "persist": False,
                "content": f"## RELEVANT FACTS\n{fact_text}",
                "tokens": self._est_tokens(fact_text)
            })

        # Tier 4: Recent lower-importance episodes
        older = recall_episodes(hours=72, min_importance=1, limit=5)
        older = [e for e in older if not any(
            r.get('content','') == e.get('content','') for r in recent
        )]  # dedup with tier 2
        if older:
            old_text = "\n".join(
                f"- [{e.get('category','')}] {e.get('content','')[:100]}"
                for e in older[:4]
            )
            self._active_blocks.append({
                "id": "t4_recent_low", "tier": 4, "persist": False,
                "content": f"## BACKGROUND CONTEXT\n{old_text}",
                "tokens": self._est_tokens(old_text)
            })

        # Compose and report
        total_tokens = sum(b["tokens"] for b in self._active_blocks)
        context_str = "\n\n".join(b["content"] for b in self._active_blocks)

        return {
            "session_id": self._session_id,
            "total_blocks": len(self._active_blocks),
            "total_est_tokens": total_tokens,
            "budget": BUDGET_TOKENS,
            "headroom": BUDGET_TOKENS - total_tokens,
            "context": context_str,
        }

    def get_context(self) -> str:
        """Get the current composed context string."""
        result = self.start_session(task="__internal__", phase="__none__")
        return result["context"]

    # ── PHASE 2: MID-SESSION TRIM ──────────────────────────────

    def trim_context(self, current_usage_tokens: int, force: bool = False,
                     target_model: str | None = None) -> dict:
        """
        Called when context window is getting full.
        Drops lowest-tier blocks first until under budget.

        The *target_model* parameter controls trim strategy:
          - "small" / "local" models (e.g. qwen3:8b) -> deletion-only.
            These have tight context budgets, so we drop entire blocks
            rather than spend tokens rephrasing them.
          - "large" / "premium" models (e.g. DeepSeek v4-pro, Ring-2.6-1t,
            claude-sonnet) -> compression-eligible.  Mid-tier blocks (T3-T4)
            are rephrased/summarised before eviction so the agent retains
            semantic content at lower token cost.
          - None (default) -> conservative deletion-only (same as "small").

        Trim-suppression:
          - If trimming is paused (via pause_trimming()), this method
            returns immediately with status="paused" and no blocks changed.
          - Blocks marked as protected (via set_block_protected()) are
            skipped during eviction/compression regardless of tier.

        Returns what was trimmed (or a paused status).
        """
        if not self._active_blocks:
            return {"trimmed": 0, "message": "No blocks to trim"}

        # ── Auto-resume check: lift pause if duration exceeded ──────
        if self._trimming_paused:
            if self._pause_start_time is not None and (
                time.time() - self._pause_start_time > self.MAX_PAUSE_DURATION
            ):
                logger.warning(
                    "[%s] Trim pause auto-resumed after %ds (reason: %s)",
                    self.session_key, self.MAX_PAUSE_DURATION, self._pause_reason,
                )
                self.resume_trimming(reason="auto-resume: MAX_PAUSE_DURATION exceeded")

        # ── If still paused, skip trimming entirely ─────────────────
        if self._trimming_paused:
            return {
                "trimmed": 0,
                "status": "paused",
                "message": (
                    f"Trimming paused (reason: {self._pause_reason or 'unknown'}). "
                    f"Call resume_trimming() to resume."
                ),
            }

        # Strategy selection
        _compressible = False
        if target_model:
            _model_lower = target_model.lower()
            _compressible = any(ind in _model_lower for ind in self._PREMIUM_INDICATORS)

        threshold = WARNING_TOKENS if not force else TARGET_POST_TRIM
        if current_usage_tokens < threshold and not force:
            return {"trimmed": 0, "message": f"Within budget ({current_usage_tokens} < {threshold})"}

        # Auto-sculpt check: trigger before heavy trim
        sculpt_result = self._check_sculpt_trigger(current_usage_tokens)
        if sculpt_result:
            return sculpt_result

        # Sort: highest tier number = lowest priority = trim first
        self._active_blocks.sort(key=lambda b: b["tier"], reverse=True)

        trimmed = []
        new_blocks = []
        for block in self._active_blocks:
            block_id = block.get("id", "")

            # Skip protected blocks regardless of tier
            if block_id in self._protected_blocks:
                new_blocks.append(block)
                continue

            if block["tier"] >= 5 and not block.get("persist", False):
                # Tiers 5-6: always trim first (tool output, raw conversation)
                trimmed.append({
                    "id": block_id, "tier": block["tier"],
                    "tokens": block["tokens"],
                    "strategy": "delete"
                })
            elif block["tier"] >= 3 and not block.get("persist", False) and current_usage_tokens > TARGET_POST_TRIM:
                # Tiers 3-4: trim if still over target
                if _compressible:
                    _compressed = self._compress_block(block)
                    if _compressed and len(_compressed) < len(block["content"]):
                        # Replace with compressed version instead of dropping
                        block["content"] = _compressed
                        block["tokens"] = self._est_tokens(_compressed)
                        block["_was_compressed"] = True
                        new_blocks.append(block)
                        trimmed.append({
                            "id": block_id, "tier": block["tier"],
                            "tokens": block["tokens"],
                            "strategy": "compress"
                        })
                    else:
                        trimmed.append({
                            "id": block_id, "tier": block["tier"],
                            "tokens": block["tokens"],
                            "strategy": "delete"
                        })
                else:
                    trimmed.append({
                        "id": block_id, "tier": block["tier"],
                        "tokens": block["tokens"],
                        "strategy": "delete"
                    })
                current_usage_tokens -= block["tokens"]
            else:
                new_blocks.append(block)

        self._active_blocks = new_blocks
        saved = sum(t["tokens"] for t in trimmed)

        # Log trim event for tuning analysis
        self.log_tuning_observation("trim", f"Trimmed {len(trimmed)} blocks",
                                     saved_tokens=saved,
                                     blocks_deleted=len(trimmed),
                                     current_usage=current_usage_tokens,
                                     budget=BUDGET_TOKENS,
                                     target=TARGET_POST_TRIM)

        # Auto-review at 20-turn mark
        if self._turn_count >= self.MAX_TURNS_BEFORE_SCULPT:
            review = self.review_performance()
            logger.info("[%s] === PERFORMANCE REVIEW at turn %d === %s",
                        self.session_key, self._turn_count,
                        json.dumps(review.get("recommendations", []), default=str))

        # Persist trimmed high-value content to memory palace
        for t in trimmed:
            if t["tier"] <= 3:  # worth saving
                store_episode(
                    self._session_id, "context_evict",
                    f"[{t['id']}] Trimmed block (tier {t['tier']}, {t['tokens']} tokens, strategy={t.get('strategy','delete')})",
                    importance=3, tags=["trim", "context", f"tier{t['tier']}"]
                )

        return {
            "trimmed_blocks": len(trimmed),
            "tokens_recovered": saved,
            "remaining_blocks": len(self._active_blocks),
            "current_usage_tokens": current_usage_tokens,
            "message": f"Trimmed {len(trimmed)} blocks, recovered ~{saved} tokens",
        }

    def _compress_block(self, block: dict) -> str | None:
        """
        Attempt to compress a context block by rephrasing it into a concise
        summary.  Returns the compressed text or None if compression would
        not reduce size.

        This runs locally -- no external API call.  The compression is a simple
        extractive + abstractive heuristic: keep the first sentence verbatim
        (which usually contains the key fact) and strip elaboration.
        """
        content = block.get("content", "")
        if not content or len(content) < 200:
            return None  # Too short to compress meaningfully

        lines = content.strip().split("\n")
        if len(lines) <= 2:
            return None  # Already compact

        # Strategy: keep the first line (usually the topic/header) and
        # the first sentence of body content.  This preserves the semantic
        # anchor at minimal token cost.
        header = lines[0].strip()
        # Find first substantive sentence in remaining lines
        body_summary = ""
        for line in lines[1:]:
            stripped = line.strip().lstrip("-* ")
            if stripped and len(stripped) > 15:
                body_summary = stripped
                break

        if not body_summary:
            return None

        compressed = f"{header}\n> {body_summary} [...]"
        # Only return if we actually saved tokens (target >40% reduction)
        if self._est_tokens(compressed) < self._est_tokens(content) * 0.6:
            return compressed
        return None

    # ── PHASE 3: SESSION END ───────────────────────────────────

    def register_tool_output(self, tool_name: str, output: str):
        """Register a tool output as a trimable block (tier 5)."""
        tok = self._est_tokens(output)
        self._active_blocks.append({
            "id": f"tool_{tool_name}_{len(self._active_blocks)}",
            "tier": 5, "persist": False,
            "content": f"[TOOL: {tool_name}]\n{output[:2000]}",
            "tokens": tok
        })

    def register_conversation_turn(self, role: str, content: str):
        """Register a conversation turn (trimmed last). Increments turn counter."""
        self._active_blocks.append({
            "id": f"msg_{role}_{len(self._active_blocks)}",
            "tier": 6, "persist": False,
            "content": f"[{role}]: {content[-1500:]}",
            "tokens": self._est_tokens(content[-1500:])
        })
        self._turn_count += 1
        self.log_tuning_observation("turn", f"Turn {self._turn_count} registered",
                                     role=role, tokens=self._est_tokens(content[-1500:]))

    def review_performance(self) -> dict:
        """Review session performance metrics. Called at the 20-turn mark or end.

        Returns a dict with metrics and recommendations for tuning parameters.
        """
        total_turns = self._turn_count
        total_tuning_entries = len(self._tuning_log)
        trim_events = [e for e in self._tuning_log if e["category"] == "trim"]
        sculpt_events = [e for e in self._tuning_log if e["category"] == "sculpt_trigger"]

        trim_count = len(trim_events)
        sculpt_count = len(sculpt_events)

        # Calculate average tokens at trim time
        trim_tokens = [e["extra"].get("tokens", 0) for e in trim_events if "extra" in e]
        avg_tokens_at_trim = sum(trim_tokens) / max(len(trim_tokens), 1)

        # Check if we're trimming too aggressively, too early, or too late
        recommendations = []
        if trim_count > total_turns * 0.6 and total_turns >= 5:
            recommendations.append({
                "issue": "trim_firing_too_often",
                "detail": f"Trim fired {trim_count}/{total_turns} turns (>{total_turns * 0.6 * 100:.0f}%)",
                "suggestion": "TRIM_THRESHOLD may be too low or TARGET_POST_TRIM too aggressive"
            })
        elif trim_count == 0 and total_turns >= 5:
            recommendations.append({
                "issue": "trim_never_fired",
                "detail": f"No trim events in {total_turns} turns",
                "suggestion": "Budget may be too generous or context blocks are too small"
            })

        if sculpt_count > 0:
            avg_turns_to_sculpt = total_turns / sculpt_count
            if avg_turns_to_sculpt < 10:
                recommendations.append({
                    "issue": "sculpt_too_early",
                    "detail": f"Auto-sculpt every ~{avg_turns_to_sculpt:.1f} turns",
                    "suggestion": "Raise MAX_TURNS_BEFORE_SCULPT or increase BUDGET_TOKENS"
                })
        elif total_turns >= 20:
            recommendations.append({
                "issue": "sculpt_never_triggered",
                "detail": f"No sculpt in {total_turns} turns despite 20-turn limit",
                "suggestion": "Check if token_pressure trigger is too conservative (90% threshold)"
            })

        # Memory palace stats
        stats = get_stats()

        report = {
            "session_id": self._session_id,
            "total_turns": total_turns,
            "trim_events": trim_count,
            "sculpt_events": sculpt_count,
            "avg_tokens_at_trim": round(avg_tokens_at_trim),
            "memory_stats": stats,
            "current_budget": BUDGET_TOKENS,
            "current_target_post_trim": TARGET_POST_TRIM,
            "current_max_turns_sculpt": self.MAX_TURNS_BEFORE_SCULPT,
            "recommendations": recommendations,
        }

        # Persist the review
        store_episode(
            self._session_id, "performance_review",
            json.dumps(report, default=str),
            importance=8, tags=["review", "performance", "tuning"]
        )

        # Log key findings
        for rec in recommendations:
            logger.warning("[%s] TUNING RECOMMENDATION: %s — %s (suggestion: %s)",
                          self.session_key, rec["issue"], rec["detail"], rec["suggestion"])

        return report

    def end_session(self, summary: str | None = None):
        """
        Called at session end. Persists important state, runs maintenance.
        """
        # Run auto-prune at session end and get DB stats
        prune_result = auto_prune()

        # Save remaining blocks to memory palace
        for block in self._active_blocks:
            if block["tier"] <= 2 and block.get("content"):
                store_episode(
                    self._session_id, "context_snapshot",
                    block["content"][:500],
                    importance=5 if block["tier"] <= 1 else 3,
                    tags=[f"tier{block['tier']}", "context_end"]
                )

        if summary:
            store_episode(self._session_id, "session_summary", summary,
                          importance=7, tags=["summary", "session_end"])

        # Always run performance review at session end
        review = None
        if self._turn_count > 0:
            review = self.review_performance()

        # Clear working memory
        clear_working()

        # Run maintenance
        maint = prune_expired()
        stats = get_stats()

        result = {
            "session_id": self._session_id,
            "blocks_saved": len([b for b in self._active_blocks if b["tier"] <= 2]),
            "maintenance": maint,
            "prune_result": prune_result,
            "final_stats": stats,
        }
        if review:
            result["performance_review"] = review

        self._active_blocks = []
        return result


# ═══════════════════════════════════════════════════════════════
# GLOBAL REGISTRY — maps session_key → ContextOrchestrator instance
# The gateway uses this to retrieve or create per-session orchestrators.
# ═══════════════════════════════════════════════════════════════

_orchestrators: dict[str, ContextOrchestrator] = {}


def get_orchestrator(session_key: str) -> ContextOrchestrator:
    """Get or create a ContextOrchestrator for the given session key."""
    if session_key not in _orchestrators:
        _orchestrators[session_key] = ContextOrchestrator(session_key)
    return _orchestrators[session_key]


def drop_orchestrator(session_key: str | None, summary: str | None = None):
    """Remove a session's orchestrator from the registry (cleanup).

    Calls end_session() before discarding so that working memory is
    persisted to the palace and maintenance runs.  The summary string
    is stored as a session-level episode for later retrieval.
    """
    if session_key and session_key in _orchestrators:
        orch = _orchestrators.pop(session_key)
        try:
            orch.end_session(summary=summary)
        except Exception:
            # Never let cleanup crash the gateway; log and continue
            import traceback
            logger.exception("end_session failed for %s", session_key)


def _get_all_sessions() -> dict[str, ContextOrchestrator]:
    """Return all active orchestrators (for maintenance/cleanup)."""
    return dict(_orchestrators)


# ═══════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBILITY WRAPPERS — gateway expects module-level funcs
# ═══════════════════════════════════════════════════════════════

def gateway_message_start(user_input: str = "", gateway_session_id: str = "default") -> dict:
    """Start a context-orchestrator session and return initial context info."""
    orch = get_orchestrator(gateway_session_id)
    result = orch.start_session(task=user_input or "chat", phase="gateway")
    return result


def gateway_trim_check(current_tokens: int = 0, gateway_session_id: str = "default") -> dict:
    """Check if context trimming is needed and perform it."""
    orch = get_orchestrator(gateway_session_id)
    result = orch.trim_context(current_usage_tokens=current_tokens)
    return result


def gateway_register_turn(role: str, content: str, gateway_session_id: str = "default") -> dict:
    """Register a conversation turn with the context orchestrator."""
    orch = get_orchestrator(gateway_session_id)
    orch.register_conversation_turn(role, content)
    return {"status": "registered", "role": role}


def gateway_message_end(summary: str | None = None, gateway_session_id: str = "default") -> dict:
    """End a session and persist state via the context orchestrator."""
    orch = get_orchestrator(gateway_session_id)
    result = orch.end_session(summary=summary)
    return result


# ═══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — standalone self-test still works
# ═══════════════════════════════════════════════════════════════

def _legacy_self_test():
    """
    Run the original self-test using the new class-based API.
    Called only when __name__ == '__main__'.
    """
    print("=" * 50)
    print("  CONTEXT ORCHESTRATOR — SELF-TEST")
    print("=" * 50 + "\n")

    orch = get_orchestrator("test_session_001")

    # Phase 1: Start session
    print("Phase 1: start_session()")
    result = orch.start_session(task="Build context trimming pipeline", phase="infrastructure")
    print(f"  Blocks loaded: {result['total_blocks']}")
    print(f"  Estimated tokens: {result['total_est_tokens']}")
    print(f"  Headroom: {result['headroom']} tokens")
    print(f"  Context preview ({len(result['context'])} chars):")
    print("  ---")
    for line in result['context'].split('\n')[:12]:
        print(f"  | {line}")
    print("  ---\n")

    # Simulate tool output
    print("Registering tool output...")
    orch.register_tool_output("terminal", "output: 42 files processed, 3 errors found...")
    orch.register_tool_output("search_files", "Found 15 matches in /codebase")
    print(f"  Blocks now: {len(orch._active_blocks)}")

    # Simulate conversation
    print("Registering conversation turns...")
    orch.register_conversation_turn("user", "Can you help me fix this? Here's the error...")
    orch.register_conversation_turn("assistant", "Sure, the issue is in your config...")
    print(f"  Blocks now: {len(orch._active_blocks)}")

    # Phase 2: Trim
    print("\nPhase 2: trim_context(simulated 11000 tokens)")
    trim_result = orch.trim_context(current_usage_tokens=11000)
    print(f"  Trimmed: {trim_result['trimmed_blocks']} blocks")
    print(f"  Recovered: ~{trim_result['tokens_recovered']} tokens")
    print(f"  Remaining blocks: {trim_result['remaining_blocks']}")

    # Phase 3: End session
    print("\nPhase 3: end_session()")
    end = orch.end_session(summary="Built context orchestrator with 3-phase lifecycle")
    print(f"  Blocks saved to memory: {end['blocks_saved']}")
    print(f"  Maintenance: {end['maintenance']}")
    print(f"  Final memory stats: {end['final_stats']}")

    # Test registry
    print("\nTesting global registry...")
    orch2 = get_orchestrator("test_session_001")
    assert orch2 is orch, "Registry should return same instance for same key"
    drop_orchestrator("test_session_001")
    orch3 = get_orchestrator("test_session_001")
    assert orch3 is not orch, "Registry should create new instance after drop"
    print("  Registry: instance isolation OK")
    drop_orchestrator("test_session_001")

    # Test pause/resume
    print("\nTesting pause/resume...")
    orch4 = get_orchestrator("test_pause_001")
    orch4.start_session(task="test", phase="test")
    result = orch4.pause_trimming(reason="test pause")
    assert result["status"] == "paused"
    assert orch4.is_paused() is True
    info = orch4.get_pause_info()
    assert info is not None
    assert info["reason"] == "test pause"
    result = orch4.resume_trimming(reason="test resume")
    assert result["status"] == "resumed"
    assert orch4.is_paused() is False
    assert orch4.get_pause_info() is None
    print("  Pause/resume: OK")
    drop_orchestrator("test_pause_001")

    # Test block protection
    print("Testing block protection...")
    orch5 = get_orchestrator("test_protect_001")
    orch5.start_session(task="test", phase="test")
    # Protect a block
    result = orch5.set_block_protected("t0_identity", protected=True)
    assert result["protected"] is True
    assert "t0_identity" in orch5._protected_blocks
    # Unprotect
    result = orch5.set_block_protected("t0_identity", protected=False)
    assert result["protected"] is False
    assert "t0_identity" not in orch5._protected_blocks
    print("  Block protection: OK")
    drop_orchestrator("test_protect_001")

    print("\nAll tests passed. Context orchestrator working. Ready for integration.")
    print(f"   DB size: {MEMORY_DB.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    _legacy_self_test()