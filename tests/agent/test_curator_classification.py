"""Tests for the curator consolidated-vs-pruned classifier.

The classifier splits skills that disappeared between the before/after
snapshots into two buckets:

- "consolidated" — absorbed into an umbrella; content still lives
  under another skill's files
- "pruned" — archived for staleness; content not preserved elsewhere

Without the split the report lumped everything under "Skills archived",
which misled users into thinking consolidated skills had been pruned.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    yield curator


def test_classify_consolidated_via_write_file_evidence(curator_env):
    """skill_manage write_file on umbrella references/<removed>.md = consolidated."""
    result = curator_env._classify_removed_skills(
        removed=["axolotl-training"],
        added=[],
        after_names={"training-platforms", "keeper"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "write_file",
                    "name": "training-platforms",
                    "file_path": "references/axolotl-training.md",
                    "file_content": "# Axolotl\n...",
                }),
            },
        ],
    )
    assert len(result["consolidated"]) == 1
    assert result["consolidated"][0]["name"] == "axolotl-training"
    assert result["consolidated"][0]["into"] == "training-platforms"
    assert result["pruned"] == []






















def test_report_md_splits_consolidated_and_pruned_sections(curator_env):
    """End-to-end: REPORT.md shows both sections distinctly."""
    curator = curator_env
    start = datetime.now(timezone.utc)

    before = [
        {"name": "absorbed-skill", "state": "active", "pinned": False},
        {"name": "dead-skill", "state": "stale", "pinned": False},
        {"name": "keeper", "state": "active", "pinned": False},
    ]
    after = [
        {"name": "keeper", "state": "active", "pinned": False},
        {"name": "umbrella", "state": "active", "pinned": False},
    ]

    run_dir = curator._write_run_report(
        started_at=start,
        elapsed_seconds=60.0,
        auto_counts={"checked": 3, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no auto changes",
        before_report=before,
        before_names={r["name"] for r in before},
        after_report=after,
        llm_meta={
            "final": "Consolidated absorbed-skill into umbrella. Pruned dead-skill.",
            "summary": "1 consolidated, 1 pruned",
            "model": "m",
            "provider": "p",
            "error": None,
            "tool_calls": [
                {
                    "name": "skill_manage",
                    "arguments": json.dumps({
                        "action": "create",
                        "name": "umbrella",
                        "content": "# umbrella\n\nAbsorbed absorbed-skill.",
                    }),
                },
            ],
        },
    )

    payload = json.loads((run_dir / "run.json").read_text())
    # Both lists exist and are disjoint
    consolidated_names = {e["name"] for e in payload["consolidated"]}
    assert consolidated_names == {"absorbed-skill"}
    # `pruned` holds full dicts {name, source, reason}; `pruned_names` is the
    # flat list for quick scans / legacy compat.
    pruned_names = payload["pruned_names"]
    assert pruned_names == ["dead-skill"]
    assert all(isinstance(e, dict) and "name" in e for e in payload["pruned"])
    # The union still matches the legacy "archived" field for backward compat
    assert set(payload["archived"]) == consolidated_names | set(pruned_names)
    # counts exposed
    assert payload["counts"]["consolidated_this_run"] == 1
    assert payload["counts"]["pruned_this_run"] == 1

    md = (run_dir / "REPORT.md").read_text()
    # Two separate sections, not a single "Skills archived" lump
    assert "Consolidated into umbrella skills" in md
    assert "Pruned — archived for staleness" in md
    assert "`absorbed-skill` → merged into `umbrella`" in md
    assert "`dead-skill`" in md
    # The old single-lump section should not appear
    assert "### Skills archived" not in md


# ---------------------------------------------------------------------------
# _parse_structured_summary — extracting the model's required YAML block
# ---------------------------------------------------------------------------


def test_parse_structured_summary_happy_path(curator_env):
    text = (
        "Long human summary here. I processed clusters X, Y, Z.\n\n"
        "## Structured summary (required)\n"
        "```yaml\n"
        "consolidations:\n"
        "  - from: anthropic-api\n"
        "    into: llm-providers\n"
        "    reason: duplicate of the generic llm-providers skill\n"
        "  - from: openai-api\n"
        "    into: llm-providers\n"
        "    reason: same — merged with sibling\n"
        "prunings:\n"
        "  - name: random-old-notes\n"
        "    reason: pre-curator garbage, no overlap\n"
        "```\n"
    )
    out = curator_env._parse_structured_summary(text)
    assert len(out["consolidations"]) == 2
    assert out["consolidations"][0] == {
        "from": "anthropic-api",
        "into": "llm-providers",
        "reason": "duplicate of the generic llm-providers skill",
    }
    assert len(out["prunings"]) == 1
    assert out["prunings"][0]["reason"] == "pre-curator garbage, no overlap"


def test_parse_structured_summary_missing_block(curator_env):
    out = curator_env._parse_structured_summary("No block in this text.")
    assert out == {"consolidations": [], "prunings": []}










# ---------------------------------------------------------------------------
# _reconcile_classification — merging model block with heuristic
# ---------------------------------------------------------------------------












def test_reconcile_model_block_visible_in_full_report(curator_env):
    """End-to-end: LLM final response with the YAML block → reasons in REPORT.md."""
    import json as _json
    from datetime import datetime as _dt, timezone as _tz

    start = _dt.now(_tz.utc)
    before = [
        {"name": "anthropic-api", "state": "active", "pinned": False},
        {"name": "stale-thing", "state": "stale", "pinned": False},
    ]
    after = [{"name": "llm-providers", "state": "active", "pinned": False}]

    llm_final_text = (
        "Processed 3 clusters. Absorbed anthropic-api into llm-providers.\n\n"
        "## Structured summary (required)\n"
        "```yaml\n"
        "consolidations:\n"
        "  - from: anthropic-api\n"
        "    into: llm-providers\n"
        "    reason: duplicate content, now a subsection\n"
        "prunings:\n"
        "  - name: stale-thing\n"
        "    reason: pre-curator junk, no overlap with anything\n"
        "```\n"
    )

    run_dir = curator_env._write_run_report(
        started_at=start,
        elapsed_seconds=30.0,
        auto_counts={"checked": 2, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="none",
        before_report=before,
        before_names={r["name"] for r in before},
        after_report=after,
        llm_meta={
            "final": llm_final_text,
            "summary": "1 consolidated, 1 pruned",
            "model": "m",
            "provider": "p",
            "error": None,
            "tool_calls": [
                {"name": "skill_manage", "arguments": _json.dumps({
                    "action": "create",
                    "name": "llm-providers",
                    "content": "# llm-providers\nIncludes anthropic-api",
                })},
            ],
        },
    )

    payload = _json.loads((run_dir / "run.json").read_text())
    cons = payload["consolidated"][0]
    assert cons["name"] == "anthropic-api"
    assert cons["into"] == "llm-providers"
    assert cons["reason"] == "duplicate content, now a subsection"
    assert cons["source"] == "model+audit"  # model AND heuristic both had it

    pruned = payload["pruned"][0]
    assert pruned["name"] == "stale-thing"
    assert pruned["reason"] == "pre-curator junk, no overlap with anything"

    md = (run_dir / "REPORT.md").read_text()
    assert "duplicate content, now a subsection" in md
    assert "pre-curator junk" in md


# ---------------------------------------------------------------------------
# _extract_absorbed_into_declarations — authoritative signal from delete calls
# ---------------------------------------------------------------------------


def test_extract_absorbed_into_picks_up_consolidation(curator_env):
    """Delete call with absorbed_into=<umbrella> yields a declaration."""
    declarations = curator_env._extract_absorbed_into_declarations([
        {
            "name": "skill_manage",
            "arguments": json.dumps({
                "action": "delete",
                "name": "narrow-skill",
                "absorbed_into": "umbrella",
            }),
        },
    ])
    assert declarations == {
        "narrow-skill": {"into": "umbrella", "declared": True},
    }






def test_extract_absorbed_into_ignores_non_delete_actions(curator_env):
    """Patch, create, write_file etc. must not leak into declarations."""
    declarations = curator_env._extract_absorbed_into_declarations([
        {
            "name": "skill_manage",
            "arguments": json.dumps({
                "action": "patch",
                "name": "umbrella",
                "old_string": "...",
                "new_string": "...",
                "absorbed_into": "something",  # bogus on non-delete, must be ignored
            }),
        },
    ])
    assert declarations == {}










# ---------------------------------------------------------------------------
# _reconcile_classification with absorbed_into declarations (authoritative)
# ---------------------------------------------------------------------------


def test_reconcile_absorbed_into_beats_everything_else(curator_env):
    """Model declared absorbed_into at delete; YAML/heuristic disagree — declaration wins.

    This is the exact #18671 regression: the model forgets to emit the YAML
    summary block, the heuristic's substring match misses because the
    umbrella's patch content doesn't literally contain the old skill's
    slug. Previously this fell through to 'no-evidence fallback' prune,
    which dropped the cron ref instead of rewriting. With absorbed_into
    declared, the model tells us directly.
    """
    out = curator_env._reconcile_classification(
        removed=["pr-review-format"],
        heuristic={"consolidated": [], "pruned": [{"name": "pr-review-format"}]},
        model_block={"consolidations": [], "prunings": []},  # model forgot YAML block
        destinations={"hermes-agent-dev"},
        absorbed_declarations={
            "pr-review-format": {"into": "hermes-agent-dev", "declared": True},
        },
    )
    assert len(out["consolidated"]) == 1
    assert out["pruned"] == []
    e = out["consolidated"][0]
    assert e["name"] == "pr-review-format"
    assert e["into"] == "hermes-agent-dev"
    assert "absorbed_into" in e["source"]










def test_reconcile_mixed_declarations_and_legacy_calls(curator_env):
    """Real-world run: some deletes declared absorbed_into, some didn't.
    Declared ones use the authoritative path; others fall through to YAML/heuristic.
    """
    out = curator_env._reconcile_classification(
        removed=["declared-cons", "declared-prune", "legacy-cons", "legacy-prune"],
        heuristic={
            "consolidated": [
                {"name": "legacy-cons", "into": "umbrella-a", "evidence": "..."},
            ],
            "pruned": [{"name": "legacy-prune"}],
        },
        model_block={"consolidations": [], "prunings": []},
        destinations={"umbrella-a", "umbrella-b"},
        absorbed_declarations={
            "declared-cons": {"into": "umbrella-b", "declared": True},
            "declared-prune": {"into": "", "declared": True},
        },
    )
    cons_by_name = {e["name"]: e for e in out["consolidated"]}
    pruned_by_name = {e["name"]: e for e in out["pruned"]}

    assert "declared-cons" in cons_by_name
    assert cons_by_name["declared-cons"]["into"] == "umbrella-b"
    assert "absorbed_into" in cons_by_name["declared-cons"]["source"]

    assert "legacy-cons" in cons_by_name
    assert cons_by_name["legacy-cons"]["into"] == "umbrella-a"
    assert "tool-call audit" in cons_by_name["legacy-cons"]["source"]

    assert "declared-prune" in pruned_by_name
    assert "model-declared prune" in pruned_by_name["declared-prune"]["source"]

    assert "legacy-prune" in pruned_by_name
    assert "no-evidence fallback" in pruned_by_name["legacy-prune"]["source"]


# ---------------------------------------------------------------------------
# _build_rename_summary — surfaces the "where did my skills go?" map to the
# user-visible curator summary (gateway 💾 line, CLI Rich panel,
# `hermes curator status`). The full data has always been in REPORT.md on
# disk; this helper makes it visible without digging.
# ---------------------------------------------------------------------------


def test_rename_summary_empty_when_nothing_archived(curator_env):
    """No removals = empty string (no log noise on no-op ticks)."""
    result = curator_env._build_rename_summary(
        before_names={"alpha", "beta"},
        after_report=[
            {"name": "alpha", "state": "active"},
            {"name": "beta", "state": "active"},
        ],
        tool_calls=[],
        model_final="",
    )
    assert result == ""




def test_rename_summary_pruned_marked_explicitly(curator_env):
    """Pruned skills (no umbrella) say `pruned (stale)` so users don't think they were merged."""
    result = curator_env._build_rename_summary(
        before_names={"old-flaky-thing", "keeper"},
        after_report=[{"name": "keeper", "state": "active"}],
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "old-flaky-thing",
                    "absorbed_into": "",
                }),
            },
        ],
        model_final="",
    )
    assert "old-flaky-thing — pruned (stale)" in result
    assert "→" not in result.split("old-flaky-thing")[1].splitlines()[0]


def test_rename_summary_caps_at_ten_with_more_indicator(curator_env):
    """Large consolidations don't blow up the log line — cap + `… and N more`."""
    removed = [f"skill-{i}" for i in range(15)]
    tool_calls = [
        {
            "name": "skill_manage",
            "arguments": json.dumps({
                "action": "delete",
                "name": name,
                "absorbed_into": "umbrella",
            }),
        }
        for name in removed
    ]
    result = curator_env._build_rename_summary(
        before_names=set(removed) | {"umbrella"},
        after_report=[{"name": "umbrella", "state": "active"}],
        tool_calls=tool_calls,
        model_final="",
    )
    assert "archived 15 skill(s):" in result
    assert "… and 5 more" in result
    # Exactly 10 bullets shown
    bullet_count = sum(1 for ln in result.splitlines() if ln.startswith("  • "))
    assert bullet_count == 10


def test_rename_summary_mixed_consolidation_and_pruning(curator_env):
    """Consolidated entries come first, pruned entries follow — matches REPORT.md ordering."""
    result = curator_env._build_rename_summary(
        before_names={"merge-me", "drop-me", "umbrella"},
        after_report=[{"name": "umbrella", "state": "active"}],
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "merge-me",
                    "absorbed_into": "umbrella",
                }),
            },
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "drop-me",
                    "absorbed_into": "",
                }),
            },
        ],
        model_final="",
    )
    lines = result.splitlines()
    merge_idx = next(i for i, ln in enumerate(lines) if "merge-me" in ln)
    drop_idx = next(i for i, ln in enumerate(lines) if "drop-me" in ln)
    assert merge_idx < drop_idx, "consolidated should render before pruned"
    assert "merge-me → umbrella" in lines[merge_idx]
    assert "drop-me — pruned (stale)" in lines[drop_idx]


# ---------------------------------------------------------------------------
# Pin hint — surfaces `hermes curator pin <umbrella>` in the rename block so
# users learn the command exists at the moment they care (a consolidation
# just landed against their library). The hint is gated on having at least
# one umbrella destination — pruned-only runs skip it.
# ---------------------------------------------------------------------------






