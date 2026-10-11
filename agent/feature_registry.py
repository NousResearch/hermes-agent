"""Declarative registry of Hermes capabilities for proactive suggestion.

PR-B of the "Feature Onboarding" initiative: the agent learns which Hermes
capability fits the user's current need and *suggests* it (advisory only —
never auto-executes by default).

Design invariants (see SECURITY-BASELINE.md in the PR):

* **No new core tools.** Every registry entry references an existing,
  already-audited tool / skill / capability. The narrow-waist rule from
  AGENTS.md holds: nothing here ships on the API call schema.
* **Advisory by default.** The router emits *text suggestions* appended to
  the current turn's API copy (the ``ext_prefetch_cache`` sidecar). It never
  invokes a tool by itself and never weakens approval/redaction/egress gates.
* **Explainable.** Each suggestion carries a ``why`` string so users (and
  reviewers) know exactly why it fired.
* **Local, offline, cache-safe.** Pure Python keyword/signal matching over
  the current user message; no network; no system-prompt mutation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Feature:
    """A Hermes capability that the router may suggest.

    ``keywords`` and ``signals`` are matched against the current user message
    (case-insensitive substring/regex). ``min_confidence`` gates firing.
    ``suggested_capability`` names an existing tool / slash command / skill
    the agent should consider — resolved against a whitelist at router init.
    """

    id: str
    name: str
    keywords: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()            # regex patterns (compiled at init)
    suggested_capability: str = ""            # e.g. "delegate_task", "/whats-new"
    benefit: str = ""                          # what the user gains
    min_confidence: float = 0.7
    auto_apply_safe: bool = False              # True only when invoking the
                                               # capability has zero
                                               # side effects (e.g. /help)
    # Compiled regexes (populated by compile_features).
    _compiled: tuple = field(default=(), repr=False, compare=False)

    def match(self, text: str) -> float:
        """Return a confidence score for ``text`` — the raw signal hit count.

        Keywords/patterns are OR semantics: hitting ANY signal means the
        feature is relevant. The score is the raw number of signals that
        fired (not capped), so ``min_confidence`` acts as a genuine
        "how many signals must fire" gate:

          * ``0.7``  → at least 1 hit fires
          * ``1.5``  → at least 2 hits required
          * ``2.0``  → at least 2 hits (integer boundaries inclusive)

        Capping at 1.0 would make multi-signal thresholds unreachable
        (review #81582 issue 1) — do not cap.
        """
        hits = 0
        for kw in self.keywords:
            if kw.lower() in text.lower():
                hits += 1
        for pat in self._compiled:
            if pat.search(text):
                hits += 1
        return float(hits)


def compile_features(features: List[Feature]) -> List[Feature]:
    """Compile regex patterns into a list of frozen Feature instances."""
    out: List[Feature] = []
    for f in features:
        compiled = tuple(
            re.compile(p, re.IGNORECASE) for p in f.patterns
        )
        out.append(
            Feature(
                id=f.id,
                name=f.name,
                keywords=f.keywords,
                patterns=f.patterns,
                suggested_capability=f.suggested_capability,
                benefit=f.benefit,
                min_confidence=f.min_confidence,
                auto_apply_safe=f.auto_apply_safe,
                _compiled=compiled,
            )
        )
    return out


# ---------------------------------------------------------------------------
# The registry.  Every entry references EXISTING capabilities.
# ---------------------------------------------------------------------------

# Core model-tool names the router may suggest.  These mirror the narrow
# waist of the agent toolset and are static: every one is a first-party
# model tool shipped by the agent core.
TOOL_CAPABILITIES = frozenset({
    "delegate_task",        # parallel subagents
    "cronjob",              # scheduled jobs
    "web_search",           # web research
    "browser_navigate",     # browser automation
    "terminal",             # shell
    "memory",               # save durable facts
    "skill_manage",         # create/update skills
    "kanban",               # multi-agent work queue
    "moa",                  # mixture of agents
    "computer_use",         # desktop control
})


def resolve_known_capabilities() -> frozenset[str]:
    """Return the capability names the router may suggest.

    Tools come from the static ``TOOL_CAPABILITIES`` set; slash commands are
    resolved **at runtime** from the live command registry so a registry
    entry can only suggest a command the installed product actually ships.
    A command that exists only in an unmerged companion PR is automatically
    dropped until it lands (review #81582: ``/whats-new`` gated on #81580),
    and a command removed from the product stops being suggested without a
    manual whitelist edit.  The resolution is defensive: if the command
    module cannot be imported, we fall back to tools-only rather than
    raising (a broken registry must never break a turn).
    """
    names = set(TOOL_CAPABILITIES)
    try:
        from hermes_cli import commands as _commands_module

        # Read the registry attribute at call time (not import time) so
        # monkeypatched registries in tests and future dynamic registries
        # take effect without a reload.
        for cmd in getattr(_commands_module, "COMMAND_REGISTRY", ()):
            names.add(f"/{cmd.name}")
            for alias in cmd.aliases:
                names.add(f"/{alias}")
    except Exception as e:  # noqa: BLE001 — defensive fallback
        logger.debug(
            "feature registry: could not resolve slash commands "
            "(falling back to tools only): %s", e,
        )
    return frozenset(names)


def _seed_features() -> List[Feature]:
    return [
        Feature(
            id="parallel_subtasks",
            name="Parallelize independent subtasks",
            keywords=(
                "parallel", "batch", "同时", "并行", "批量",
                "multi-task", "multitask", "in parallel",
            ),
            # Positive signals only; "each of" / "all of them" are dropped
            # because they match sequential phrasing ("do each of these
            # sequentially").  "several files"/"multiple repos" alone are
            # also ambiguous (a single serial task can touch several files).
            patterns=(
                r"\b(?:run|execute|do|handle)\s+(?:these|those|all|both)\s+\d+\s+(?:files|tasks|jobs|repos)\b",
                r"\b(?:in parallel|concurrently|simultaneously)\b",
                r"(?:这|那|这些|那些|全部)\s*\d+\s*(?:个|份|台|条)?\s*(?:文件|任务|项目|仓库|脚本|目录)",
            ),
            suggested_capability="delegate_task",
            benefit="Run independent subtasks in parallel (~3x wall-clock reduction on typical batches).",
            min_confidence=0.7,
        ),
        Feature(
            id="scheduled_recurring",
            name="Schedule recurring work",
            keywords=(
                "every day", "daily", "weekly", "every morning", "每天",
                "每周", "recurring", "scheduled", "remind me", "每天早上",
            ),
            # Bare 每 matches 每次/每个/每秒 (unrelated); use concrete
            # time-anchored phrases only.
            patterns=(
                r"(?:每天|每日|每周|每个月|every\s+(?:day|morning|week|month|hour)\b)",
                r"at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)",
            ),
            suggested_capability="cronjob",
            benefit="Automate recurring checks/pushes without manual re-runs.",
            min_confidence=0.7,
        ),
        Feature(
            id="web_research",
            name="Research the web",
            keywords=(
                "research", "web search", "find out", "look up", "搜索",
                "investigate", "sources", "citations", "上网查", "查一下",
            ),
            # "search" alone is too broad (search the codebase, search the
            # directory) — require web-ish phrasing.  latest/current/today
            # also match local tasks ("current directory") so they are only
            # combined with the web-ish keywords via the pattern below.
            patterns=(
                r"\b(?:search|look up|find)\s+(?:the\s+)?(?:web|internet|online|latest|current news|today'?s)\b",
                r"\b(?:what|who|how|is|are)\s+.*\b(?:latest|recent|today|current)\b",
            ),
            suggested_capability="web_search",
            benefit="Ground answers in live web sources with citations.",
            min_confidence=0.7,
        ),
        Feature(
            id="remember_fact",
            name="Persist a durable fact",
            keywords=(
                "remember", "don't forget", "记住", "我的偏好", "以后",
                "from now on", "always remember",
            ),
            # bare "always"/"prefer" match "the function always returns None"
            # or "I'd prefer you didn't" — require explicit memory intent.
            patterns=(
                r"\b(?:remember that|remember to|remember:)\b",
                r"\b(?:i|我)\s+(?:prefer|like|喜欢|偏好)\s+.*\b(?:always|以后|from now on)\b",
                r"\b(?:please|帮我)\s+(?:remember|记住)\b",
            ),
            suggested_capability="memory",
            benefit="Make the preference persist across sessions.",
            min_confidence=0.7,
        ),
        Feature(
            id="release_brief",
            name="Check what's new in this release",
            keywords=(
                "what's new", "whats new", "new features", "changelog",
                "更新了什么", "新功能", "release notes",
            ),
            patterns=(r"\b(?:what's\s+new|new\s+features|changelog|release\s+notes)\b",),
            suggested_capability="/whats-new",
            benefit="See the current release's new features and how to use them.",
            min_confidence=0.7,
            auto_apply_safe=True,  # read-only informational command
        ),
    ]


class FeatureRegistry:
    """Loads, compiles, and serves the feature registry."""

    def __init__(self, features: Optional[List[Feature]] = None):
        raw = features if features is not None else _seed_features()
        # Capability guard: drop any entry naming a capability that does not
        # exist in the INSTALLED product.  Tools are a static set; slash
        # commands are resolved at runtime from the live command registry,
        # so a seed naming a command from an unmerged companion PR is
        # dropped until that PR lands (and re-enabled automatically after).
        known = resolve_known_capabilities()
        kept: List[Feature] = []
        for f in raw:
            if f.suggested_capability and f.suggested_capability not in known:
                logger.warning(
                    "feature %s references unknown capability %r — dropped",
                    f.id, f.suggested_capability,
                )
                continue
            kept.append(f)
        self.features = compile_features(kept)
        self._capability_map = {f.id: f for f in self.features}

    def suggest(self, text: str, *, min_confidence: float | None = None) -> Optional[Feature]:
        """Return the best-matching feature for ``text``, or None.

        ``min_confidence`` overrides the per-feature threshold (used by the
        router's global conservative default).
        """
        best: Optional[Feature] = None
        best_score = 0.0
        for f in self.features:
            threshold = min_confidence if min_confidence is not None else f.min_confidence
            score = f.match(text)
            if score >= threshold and score > best_score:
                best = f
                best_score = score
        return best

    def suggest_text(self, text: str, *, min_confidence: float | None = None) -> str:
        """Return a formatted suggestion string, or '' when nothing matches."""
        f = self.suggest(text, min_confidence=min_confidence)
        if f is None:
            return ""
        lines = [
            f"[feature-suggestion] Consider using **{f.name}** "
            f"(capability: `{f.suggested_capability}`).",
            f"  Why: {f.benefit}",
            "  (Advisory — nothing runs without your approval. Dismiss with "
            "`proactive_features.enabled: false`.)",
        ]
        return "\n".join(lines)
