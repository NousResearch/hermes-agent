"""Fork-invariant registry: fork-required features an upstream sync must NOT drop.

WHY THIS FILE EXISTS
--------------------
This fork tracks NousResearch/hermes-agent. A sync (`scripts/sync-upstream.sh`
+ nightly `hermes update`) merges upstream/main. A merge can *silently* remove
or replace a fork feature with **no git conflict** — e.g. upstream refactors a
module away, or replaces the fork's variant of a tool with its own. `git` only
flags overlapping-line conflicts; a clean deletion or adjacent-line swap sails
through. The motivating incident: a sync swapped the fork's `web_extract`
pipeline for upstream's with no conflict and no guard firing.

`scripts/sync-upstream.sh::check_fork_invariants` is a grep-based *absence* net
(a reverted upstream symbol must NOT reappear). This file is the complementary
*presence* net: each fork feature below MUST still exist after a merge. Because
`testpaths = ["tests"]` and this runs in the CI aggregate that gates auto-merge
(`.github/workflows/ci.yml::all-checks-pass`, wired in the 2026-06-30 CI-gating
work), a sync PR that deletes a fork feature turns this suite RED and cannot
auto-merge — it hands off for manual review instead of regressing the fleet.

HOW TO MAINTAIN
---------------
- Adding a fork feature that a future sync could silently undo? Add an entry.
- Retiring a fork delta on purpose (e.g. upstream now covers it)? Delete its
  entry in the SAME change that removes the delta — don't let this go stale.
- Each entry records a `drop_when` so a future syncer knows when the guard is
  obsolete rather than load-bearing.

A failure here does not mean "the fork is broken" — it means "an upstream merge
removed something the fleet depends on; re-apply it before merging."
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect

import pytest

# --- Module-presence checks (locate-only; no import side effects) -----------
# find_spec confirms the module *file* is still present on the fork without
# executing it, so an optional runtime dep (mattermost driver, etc.) can't turn
# a genuine "feature deleted" into a confusing import error, and vice-versa.
FORK_MODULES = [
    pytest.param(
        "gateway.platforms.config_api",
        "Master Console config API (read/write agent config.yaml over HTTP)",
        "upstream ships an equivalent console config API",
        id="console-config-api",
    ),
    pytest.param(
        "gateway.platforms.kanban_api",
        "Master Console Kanban dispatch-state API",
        "upstream ships an equivalent Kanban/dispatch API",
        id="console-kanban-api",
    ),
    pytest.param(
        "gateway.platforms.soul_api",
        "Master Console SOUL.md editing API",
        "upstream ships an equivalent SOUL/persona API",
        id="console-soul-api",
    ),
    pytest.param(
        "gateway.platforms.actions_api",
        "Master Console pending-actions (tool-gate approval) API",
        "upstream ships an equivalent pending-actions API",
        id="console-actions-api",
    ),
    pytest.param(
        "tools.tool_gate",
        "tool-gate approval system (inline/deferred approval + Kanban staging)",
        "upstream lands a first-class tool-approval gate (candidate to upstream)",
        id="tool-gate-module",
    ),
    pytest.param(
        "tools.skill_discover_tool",
        "skill_discover / skill_acquire trust-gated hub-install tools",
        "upstream ships trust-gated skill discovery + install",
        id="skill-discover-module",
    ),
    pytest.param(
        "plugins.platforms.mattermost.adapter",
        "Mattermost platform adapter (Master Console chat surface)",
        "the fleet no longer uses Mattermost",
        id="mattermost-adapter",
    ),
    pytest.param(
        "plugins.platforms.mattermost.approval",
        "Mattermost interactive tool-gate approval cards",
        "the fleet no longer uses Mattermost approvals",
        id="mattermost-approval",
    ),
]


@pytest.mark.parametrize("module, what, drop_when", FORK_MODULES)
def test_fork_module_present(module: str, what: str, drop_when: str) -> None:
    assert importlib.util.find_spec(module) is not None, (
        f"FORK FEATURE REGRESSED BY SYNC: module '{module}' is gone — {what}. "
        f"An upstream merge removed it. Re-apply before merging. "
        f"Drop this guard only when: {drop_when}. See tests/fork/README.md."
    )


# --- Structural / behavioral checks on core modules (safe to import) --------

def test_skill_discovery_tools_in_core_and_toolset() -> None:
    """skill_discover/skill_acquire must stay in core AND the `skills` toolset.

    They belong to the `skills` toolset; if they drop out of `_HERMES_CORE_TOOLS`
    the `skills` toolset becomes a superset of the hermes-cli composite and never
    reverse-maps as enabled — the two tools silently go dark in a default CLI
    session (the latent bug fixed in the 2026-06-30 green-main work). Drop this
    guard only if skill_discover/skill_acquire are removed on purpose.
    """
    toolsets = importlib.import_module("toolsets")
    # getattr keeps these as Any: TOOLSETS is a heterogeneous dict literal, and a
    # structural typecheck otherwise unions unrelated value types into the `in`.
    core_tools = getattr(toolsets, "_HERMES_CORE_TOOLS")
    all_toolsets = getattr(toolsets, "TOOLSETS")
    for tool in ("skill_discover", "skill_acquire"):
        assert tool in core_tools, (
            f"FORK FEATURE REGRESSED: '{tool}' fell out of _HERMES_CORE_TOOLS — "
            "an upstream merge reverted the core-tools list. See tests/fork/README.md."
        )
        assert tool in all_toolsets["skills"]["tools"], (
            f"FORK FEATURE REGRESSED: '{tool}' fell out of the 'skills' toolset. "
            "See tests/fork/README.md."
        )


def test_web_extract_is_deterministic_no_llm_summarization() -> None:
    """web_extract must return clean page text deterministically — NO LLM step.

    This is the exact feature a past sync silently swapped. The fork's contract:
    clean markdown/text with a deterministic head+tail+footer window, never an
    LLM summary (which costs tokens and hides content from the agent). We assert
    the tool schema still advertises 'no LLM summarization' AND the handler
    exists. Drop this guard only if the fork deliberately reintroduces an LLM
    distillation pass for web_extract.
    """
    web_tools = importlib.import_module("tools.web_tools")
    assert hasattr(web_tools, "web_extract_tool"), (
        "FORK FEATURE REGRESSED: web_extract_tool handler is gone. "
        "See tests/fork/README.md."
    )
    desc = str(web_tools.WEB_EXTRACT_SCHEMA.get("description", "")).lower()
    assert "no llm summarization" in desc, (
        "FORK FEATURE REGRESSED BY SYNC: web_extract no longer advertises "
        "'no LLM summarization' — an upstream merge may have swapped the fork's "
        "deterministic pipeline for an LLM-distillation one (the original "
        "motivating regression). See tests/fork/README.md."
    )


def test_tool_gate_public_surface_present() -> None:
    """The tool-gate approval system's public entrypoints must survive a sync.

    These back the Master Console's inline/deferred approval + Kanban staging.
    Drop this guard only if the gate is removed or upstreamed (the roadmap's
    prime upstream-contribution candidate).
    """
    tool_gate = importlib.import_module("tools.tool_gate")
    for fn in ("gate_enabled", "requires_approval", "stage_deferred", "approve_action"):
        assert callable(getattr(tool_gate, fn, None)), (
            f"FORK FEATURE REGRESSED: tool_gate.{fn} is missing or not callable — "
            "an upstream merge altered the approval-gate surface. "
            "See tests/fork/README.md."
        )


def test_cron_storage_stays_per_profile() -> None:
    """cron/jobs.py must resolve storage from the active profile home, not root.

    The fleet runs one `--profile <name>` gateway per agent, each ticking its
    own ~/.hermes/profiles/<name>/cron. Anchoring at get_default_hermes_root()
    collapses every agent onto one shared store and orphans profile-local jobs
    (issue #4707 security boundary). This mirrors the grep guard in
    scripts/sync-upstream.sh; here as a positive assertion too. Drop only once
    upstream lands per-job profile-execution scoping (#48649).
    """
    jobs = importlib.import_module("cron.jobs")
    # Parse the AST and inspect real name references, not the raw text — the
    # module's own comments explain to NOT use get_default_hermes_root(), and a
    # substring scan would trip on that prose (as the sync-script grep guard once
    # did). Only actual code references count as a regression.
    tree = ast.parse(inspect.getsource(jobs))
    referenced = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "get_hermes_home" in referenced, (
        "FORK FEATURE REGRESSED: cron/jobs.py no longer references "
        "get_hermes_home — per-profile cron isolation may be broken. "
        "See tests/fork/README.md."
    )
    assert "get_default_hermes_root" not in referenced, (
        "FORK INVARIANT VIOLATED: cron/jobs.py references get_default_hermes_root "
        "— an upstream merge re-landed root-anchored cron storage (revert of "
        "a5c09fd17/#32091), collapsing per-profile isolation. See tests/fork/README.md."
    )
