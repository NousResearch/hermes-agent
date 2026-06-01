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


def test_classify_pruned_when_no_destination_reference(curator_env):
    """Removed skill with no referencing tool call = pruned."""
    result = curator_env._classify_removed_skills(
        removed=["old-stale-thing"],
        added=[],
        after_names={"keeper"},
        tool_calls=[
            {"name": "skills_list", "arguments": "{}"},
            {"name": "skill_manage", "arguments": json.dumps({
                "action": "patch", "name": "keeper",
                "old_string": "foo", "new_string": "bar",
            })},
        ],
    )
    assert result["consolidated"] == []
    assert len(result["pruned"]) == 1
    assert result["pruned"][0]["name"] == "old-stale-thing"


def test_classify_consolidated_into_newly_created_umbrella(curator_env):
    """Removed skill absorbed into a skill that was created THIS run."""
    result = curator_env._classify_removed_skills(
        removed=["anthropic-api"],
        added=["llm-providers"],  # new umbrella
        after_names={"llm-providers"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "create",
                    "name": "llm-providers",
                    "content": "# LLM Providers\n\n## anthropic-api\nMerged from the old anthropic-api skill.\n",
                }),
            },
        ],
    )
    assert len(result["consolidated"]) == 1
    assert result["consolidated"][0]["name"] == "anthropic-api"
    assert result["consolidated"][0]["into"] == "llm-providers"


def test_classify_handles_underscore_hyphen_variants(curator_env):
    """Names with hyphens match underscore forms in paths/content and vice versa."""
    result = curator_env._classify_removed_skills(
        removed=["open-webui-setup"],
        added=[],
        after_names={"webui"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "write_file",
                    "name": "webui",
                    "file_path": "references/open_webui_setup.md",
                    "file_content": "...",
                }),
            },
        ],
    )
    assert len(result["consolidated"]) == 1
    assert result["consolidated"][0]["into"] == "webui"


def test_classify_self_reference_does_not_count(curator_env):
    """A tool call that targets the removed skill itself is NOT consolidation."""
    # e.g. the curator patched the skill once and later archived it
    result = curator_env._classify_removed_skills(
        removed=["doomed"],
        added=[],
        after_names={"keeper"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "patch",
                    "name": "doomed",  # same as removed
                    "old_string": "x",
                    "new_string": "y",
                }),
            },
        ],
    )
    assert result["consolidated"] == []
    assert result["pruned"][0]["name"] == "doomed"


def test_classify_destination_must_exist_after_run(curator_env):
    """A reference to a skill that doesn't exist after the run can't be the umbrella."""
    result = curator_env._classify_removed_skills(
        removed=["thing"],
        added=[],
        after_names={"keeper"},  # "ghost" not in here
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "write_file",
                    "name": "ghost",  # not in after_names
                    "file_path": "references/thing.md",
                    "file_content": "...",
                }),
            },
        ],
    )
    assert result["consolidated"] == []
    assert result["pruned"][0]["name"] == "thing"


def test_classify_mixed_run_produces_both_buckets(curator_env):
    """A realistic run: one skill consolidated, one skill pruned."""
    result = curator_env._classify_removed_skills(
        removed=["absorbed-skill", "dead-skill"],
        added=["umbrella"],
        after_names={"umbrella", "keeper"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "write_file",
                    "name": "umbrella",
                    "file_path": "references/absorbed-skill.md",
                    "file_content": "...",
                }),
            },
        ],
    )
    assert len(result["consolidated"]) == 1
    assert result["consolidated"][0]["name"] == "absorbed-skill"
    assert result["consolidated"][0]["into"] == "umbrella"
    assert len(result["pruned"]) == 1
    assert result["pruned"][0]["name"] == "dead-skill"


def test_classify_handles_malformed_arguments_string(curator_env):
    """Truncated/malformed JSON in arguments falls back to substring match."""
    # Arguments truncated to 400 chars may not parse as JSON.
    truncated_raw = (
        '{"action":"write_file","name":"umbrella","file_path":"references/'
        'absorbed-skill.md","file_content":"long content that was cut off mid'
    )
    result = curator_env._classify_removed_skills(
        removed=["absorbed-skill"],
        added=[],
        after_names={"umbrella"},
        tool_calls=[
            {"name": "skill_manage", "arguments": truncated_raw},
        ],
    )
    # Fallback substring match finds "absorbed-skill" in the raw truncated string
    # even though json.loads fails — but it can't identify target="umbrella"
    # because _raw is the only haystack and there's no dict access. The
    # classifier only promotes to "consolidated" if it can identify a target
    # skill from args.get("name"). Ensure we fail safe: no false positive.
    # (This is a correctness floor — better to prune-label than hallucinate
    # an umbrella that wasn't really used.)
    assert result["consolidated"] == []
    assert len(result["pruned"]) == 1


def test_classify_no_false_positive_short_name_in_file_path(curator_env):
    """Short skill name that is a substring of another filename = pruned, not consolidated."""
    # e.g. "api" should NOT match "references/api-design.md"
    result = curator_env._classify_removed_skills(
        removed=["api"],
        added=[],
        after_names={"conventions"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "write_file",
                    "name": "conventions",
                    "file_path": "references/api-design.md",
                    "file_content": "# API Design\n...",
                }),
            },
        ],
    )
    assert result["consolidated"] == [], (
        f"Short name 'api' should NOT match file_path 'references/api-design.md'"
    )
    assert len(result["pruned"]) == 1
    assert result["pruned"][0]["name"] == "api"


def test_classify_no_false_positive_short_name_in_content(curator_env):
    """Short skill name embedded in longer word in content = pruned, not consolidated."""
    # e.g. "test" should NOT match content "running latest tests"
    result = curator_env._classify_removed_skills(
        removed=["test"],
        added=[],
        after_names={"umbrella"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "patch",
                    "name": "umbrella",
                    "old_string": "old",
                    "new_string": "running latest tests with pytest",
                }),
            },
        ],
    )
    assert result["consolidated"] == [], (
        f"Short name 'test' should NOT match 'latest' via word boundary"
    )
    assert len(result["pruned"]) == 1


def test_classify_still_matches_exact_word_in_content(curator_env):
    """Word-boundary match still works for exact word occurrences."""
    # "api" SHOULD match content "use the api gateway"
    result = curator_env._classify_removed_skills(
        removed=["api"],
        added=[],
        after_names={"gateway"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "edit",
                    "name": "gateway",
                    "content": "# Gateway\n\nUse the api gateway for all requests.\n",
                }),
            },
        ],
    )
    assert len(result["consolidated"]) == 1, (
        f"'api' should match as a standalone word in content"
    )
    assert result["consolidated"][0]["into"] == "gateway"


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


def test_parse_structured_summary_malformed_yaml(curator_env):
    text = "```yaml\nthis: is\n  not: [valid yaml\n```"
    out = curator_env._parse_structured_summary(text)
    assert out == {"consolidations": [], "prunings": []}


def test_parse_structured_summary_empty_lists(curator_env):
    text = "```yaml\nconsolidations: []\nprunings: []\n```"
    out = curator_env._parse_structured_summary(text)
    assert out == {"consolidations": [], "prunings": []}


def test_parse_structured_summary_ignores_bare_strings(curator_env):
    """Entries that aren't dicts (e.g. a model wrote bare names) are skipped."""
    text = (
        "```yaml\n"
        "consolidations:\n"
        "  - just-a-bare-string\n"
        "  - from: real-entry\n"
        "    into: umbrella\n"
        "    reason: valid\n"
        "prunings: []\n"
        "```"
    )
    out = curator_env._parse_structured_summary(text)
    assert len(out["consolidations"]) == 1
    assert out["consolidations"][0]["from"] == "real-entry"


def test_parse_structured_summary_missing_required_fields(curator_env):
    """Consolidation entries without from+into are skipped."""
    text = (
        "```yaml\n"
        "consolidations:\n"
        "  - from: only-from\n"
        "    reason: no into\n"
        "  - into: only-into\n"
        "  - from: good\n"
        "    into: umbrella\n"
        "prunings: []\n"
        "```"
    )
    out = curator_env._parse_structured_summary(text)
    assert len(out["consolidations"]) == 1
    assert out["consolidations"][0]["from"] == "good"


# ---------------------------------------------------------------------------
# _reconcile_classification — merging model block with heuristic
# ---------------------------------------------------------------------------


def test_reconcile_model_wins_when_umbrella_exists(curator_env):
    """Model claim + umbrella in destinations → model authority (with reason)."""
    out = curator_env._reconcile_classification(
        removed=["anthropic-api"],
        heuristic={"consolidated": [], "pruned": [{"name": "anthropic-api"}]},
        model_block={
            "consolidations": [{
                "from": "anthropic-api",
                "into": "llm-providers",
                "reason": "duplicate",
            }],
            "prunings": [],
        },
        destinations={"llm-providers"},
    )
    assert len(out["consolidated"]) == 1
    e = out["consolidated"][0]
    assert e["name"] == "anthropic-api"
    assert e["into"] == "llm-providers"
    assert e["reason"] == "duplicate"
    assert e["source"] == "model"
    assert out["pruned"] == []


def test_reconcile_model_hallucinates_umbrella(curator_env):
    """Model names a non-existent umbrella — downgrade, prefer heuristic if any."""
    out = curator_env._reconcile_classification(
        removed=["thing"],
        heuristic={
            "consolidated": [{"name": "thing", "into": "real-umbrella", "evidence": "..."}],
            "pruned": [],
        },
        model_block={
            "consolidations": [{
                "from": "thing",
                "into": "nonexistent-umbrella",
                "reason": "confused",
            }],
            "prunings": [],
        },
        destinations={"real-umbrella"},
    )
    assert len(out["consolidated"]) == 1
    e = out["consolidated"][0]
    assert e["into"] == "real-umbrella"
    assert "tool-call audit" in e["source"]
    assert e["model_claimed_into"] == "nonexistent-umbrella"


def test_reconcile_model_hallucinates_with_no_heuristic_evidence(curator_env):
    """Model names a non-existent umbrella AND no tool-call evidence → prune."""
    out = curator_env._reconcile_classification(
        removed=["ghost"],
        heuristic={"consolidated": [], "pruned": [{"name": "ghost"}]},
        model_block={
            "consolidations": [{
                "from": "ghost",
                "into": "nonexistent",
                "reason": "wrong",
            }],
            "prunings": [],
        },
        destinations={"real-umbrella"},
    )
    assert out["consolidated"] == []
    assert len(out["pruned"]) == 1
    assert "fallback" in out["pruned"][0]["source"]


def test_reconcile_heuristic_catches_model_omission(curator_env):
    """Model forgot to list a consolidation, heuristic found it."""
    out = curator_env._reconcile_classification(
        removed=["forgotten"],
        heuristic={
            "consolidated": [{
                "name": "forgotten",
                "into": "umbrella",
                "evidence": "write_file on umbrella referenced forgotten.md",
            }],
            "pruned": [],
        },
        model_block={"consolidations": [], "prunings": []},
        destinations={"umbrella"},
    )
    assert len(out["consolidated"]) == 1
    e = out["consolidated"][0]
    assert e["into"] == "umbrella"
    assert "model omitted" in e["source"]


def test_reconcile_model_prunes_with_reason(curator_env):
    """Model says pruned, heuristic agrees, we surface the reason."""
    out = curator_env._reconcile_classification(
        removed=["stale-skill"],
        heuristic={"consolidated": [], "pruned": [{"name": "stale-skill"}]},
        model_block={
            "consolidations": [],
            "prunings": [{"name": "stale-skill", "reason": "superseded by bundled skill"}],
        },
        destinations=set(),
    )
    assert len(out["pruned"]) == 1
    e = out["pruned"][0]
    assert e["reason"] == "superseded by bundled skill"
    assert e["source"] == "model"


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


def test_extract_absorbed_into_empty_string_is_explicit_prune(curator_env):
    """absorbed_into='' is recorded as an explicit prune declaration."""
    declarations = curator_env._extract_absorbed_into_declarations([
        {
            "name": "skill_manage",
            "arguments": json.dumps({
                "action": "delete",
                "name": "stale",
                "absorbed_into": "",
            }),
        },
    ])
    assert declarations == {"stale": {"into": "", "declared": True}}


def test_extract_absorbed_into_missing_arg_ignored(curator_env):
    """Delete call without absorbed_into is skipped — fallback to heuristic."""
    declarations = curator_env._extract_absorbed_into_declarations([
        {
            "name": "skill_manage",
            "arguments": json.dumps({
                "action": "delete",
                "name": "legacy-skill",
            }),
        },
    ])
    assert declarations == {}


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


def test_extract_absorbed_into_accepts_dict_arguments(curator_env):
    """arguments can arrive as a dict (defensive path) — still works."""
    declarations = curator_env._extract_absorbed_into_declarations([
        {
            "name": "skill_manage",
            "arguments": {
                "action": "delete",
                "name": "narrow",
                "absorbed_into": "umbrella",
            },
        },
    ])
    assert declarations == {"narrow": {"into": "umbrella", "declared": True}}


def test_extract_absorbed_into_strips_whitespace(curator_env):
    declarations = curator_env._extract_absorbed_into_declarations([
        {
            "name": "skill_manage",
            "arguments": json.dumps({
                "action": "delete",
                "name": "  narrow  ",
                "absorbed_into": "  umbrella  ",
            }),
        },
    ])
    assert declarations == {"narrow": {"into": "umbrella", "declared": True}}


def test_extract_absorbed_into_ignores_non_skill_manage_calls(curator_env):
    declarations = curator_env._extract_absorbed_into_declarations([
        {"name": "terminal", "arguments": json.dumps({"command": "ls"})},
        {"name": "read_file", "arguments": json.dumps({"path": "/tmp/x"})},
    ])
    assert declarations == {}


def test_extract_absorbed_into_handles_malformed_arguments(curator_env):
    """Garbage JSON in arguments must not crash the extractor."""
    declarations = curator_env._extract_absorbed_into_declarations([
        {"name": "skill_manage", "arguments": "{not json"},
        {"name": "skill_manage", "arguments": None},
        {"name": "skill_manage"},  # no arguments key at all
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


def test_reconcile_absorbed_into_empty_is_explicit_prune(curator_env):
    """absorbed_into='' takes precedence and routes to pruned, not fallback."""
    out = curator_env._reconcile_classification(
        removed=["stale"],
        heuristic={"consolidated": [], "pruned": [{"name": "stale"}]},
        model_block={"consolidations": [], "prunings": []},
        destinations=set(),
        absorbed_declarations={
            "stale": {"into": "", "declared": True},
        },
    )
    assert out["consolidated"] == []
    assert len(out["pruned"]) == 1
    assert "model-declared prune" in out["pruned"][0]["source"]


def test_reconcile_absorbed_into_nonexistent_target_falls_through(curator_env):
    """If the declared umbrella doesn't exist in destinations, fall through to
    heuristic/YAML logic. Shouldn't happen in practice (the tool validates at
    delete time) but the reconciler is defensive."""
    out = curator_env._reconcile_classification(
        removed=["thing"],
        heuristic={
            "consolidated": [{"name": "thing", "into": "real-umbrella", "evidence": "..."}],
            "pruned": [],
        },
        model_block={"consolidations": [], "prunings": []},
        destinations={"real-umbrella"},
        absorbed_declarations={
            "thing": {"into": "ghost-umbrella", "declared": True},
        },
    )
    assert len(out["consolidated"]) == 1
    assert out["consolidated"][0]["into"] == "real-umbrella"
    assert "tool-call audit" in out["consolidated"][0]["source"]


def test_reconcile_declaration_preserves_yaml_reason(curator_env):
    """When the model both declared absorbed_into AND emitted YAML with reason,
    the reason carries through so REPORT.md still has it."""
    out = curator_env._reconcile_classification(
        removed=["narrow"],
        heuristic={"consolidated": [], "pruned": []},
        model_block={
            "consolidations": [{
                "from": "narrow",
                "into": "umbrella",
                "reason": "duplicate of umbrella's main content",
            }],
            "prunings": [],
        },
        destinations={"umbrella"},
        absorbed_declarations={
            "narrow": {"into": "umbrella", "declared": True},
        },
    )
    assert len(out["consolidated"]) == 1
    e = out["consolidated"][0]
    assert e["into"] == "umbrella"
    assert "absorbed_into" in e["source"]
    assert e["reason"] == "duplicate of umbrella's main content"


def test_reconcile_without_declarations_preserves_legacy_behavior(curator_env):
    """Backward compat: no absorbed_declarations arg → all existing logic intact."""
    out = curator_env._reconcile_classification(
        removed=["thing"],
        heuristic={
            "consolidated": [{"name": "thing", "into": "umbrella", "evidence": "..."}],
            "pruned": [],
        },
        model_block={"consolidations": [], "prunings": []},
        destinations={"umbrella"},
        # no absorbed_declarations — defaults to None → behaves identically to pre-change
    )
    assert len(out["consolidated"]) == 1
    assert out["consolidated"][0]["into"] == "umbrella"


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


def test_rename_summary_consolidation_shows_target(curator_env):
    """Consolidated skills render as `name → umbrella` with the actual target."""
    result = curator_env._build_rename_summary(
        before_names={"pdf-extraction", "docx-extraction", "document-tools"},
        after_report=[{"name": "document-tools", "state": "active"}],
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "pdf-extraction",
                    "absorbed_into": "document-tools",
                }),
            },
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "docx-extraction",
                    "absorbed_into": "document-tools",
                }),
            },
        ],
        model_final="",
    )
    assert "archived 2 skill(s):" in result
    assert "pdf-extraction → document-tools" in result
    assert "docx-extraction → document-tools" in result
    assert "full report: hermes curator status" in result


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


def test_rename_summary_pin_hint_appears_when_consolidation_produced_umbrella(curator_env):
    """When at least one skill was absorbed into an umbrella, hint at pinning it."""
    result = curator_env._build_rename_summary(
        before_names={"pdf-extraction", "docx-extraction", "document-tools"},
        after_report=[{"name": "document-tools", "state": "active"}],
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "pdf-extraction",
                    "absorbed_into": "document-tools",
                }),
            },
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "docx-extraction",
                    "absorbed_into": "document-tools",
                }),
            },
        ],
        model_final="",
    )
    assert "hermes curator pin document-tools" in result
    assert "keep an umbrella stable" in result


def test_rename_summary_pin_hint_skipped_for_pruned_only_runs(curator_env):
    """Pruned-only runs have nothing surviving to pin — hint should not appear."""
    result = curator_env._build_rename_summary(
        before_names={"old-flaky-thing", "another-stale", "keeper"},
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
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "another-stale",
                    "absorbed_into": "",
                }),
            },
        ],
        model_final="",
    )
    # Block still renders (skills were archived) but no pin hint.
    assert "archived 2 skill(s):" in result
    assert "hermes curator pin" not in result
    assert "keep an umbrella stable" not in result


def test_rename_summary_pin_hint_picks_one_umbrella_when_multiple_absorbed(curator_env):
    """Multiple umbrellas → hint shows one example (alphabetically first), not a list."""
    result = curator_env._build_rename_summary(
        before_names={"a-skill", "b-skill", "umbrella-zeta", "umbrella-alpha"},
        after_report=[
            {"name": "umbrella-zeta", "state": "active"},
            {"name": "umbrella-alpha", "state": "active"},
        ],
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "a-skill",
                    "absorbed_into": "umbrella-zeta",
                }),
            },
            {
                "name": "skill_manage",
                "arguments": json.dumps({
                    "action": "delete",
                    "name": "b-skill",
                    "absorbed_into": "umbrella-alpha",
                }),
            },
        ],
        model_final="",
    )
    # Sorted picks alphabetically first.
    assert "hermes curator pin umbrella-alpha" in result
    # Exactly one hint line, not one per umbrella.
    pin_lines = [ln for ln in result.splitlines() if "hermes curator pin" in ln]
    assert len(pin_lines) == 1
