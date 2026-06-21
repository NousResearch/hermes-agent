"""Unit tests for the system-prompt prelude resolver.

Run (from the worktree, with the venv active):
    python -m pytest tests/test_system_prompt_prelude.py -q
or standalone:
    python tests/test_system_prompt_prelude.py
"""

from __future__ import annotations

import os
import textwrap

import pytest

from agent.system_prompt_prelude import resolve_prelude


def _write(d, name, body):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(body)
    return p


def _make_env(tmp_path, monkeypatch, rules, *, enabled=True, first_match=True, base_dir=None, extra=None):
    """Create prelude files + a standalone config YAML, point the resolver at it."""
    base = base_dir or str(tmp_path)
    blk = {
        "enabled": enabled,
        "base_dir": base,
        "first_match": first_match,
        "rules": rules,
    }
    if extra:
        blk.update(extra)
    cfg = {"system_prompt_prelude": blk}
    import yaml

    cfg_path = os.path.join(str(tmp_path), "prelude-map.yaml")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    monkeypatch.setenv("HERMES_PRELUDE_CONFIG", cfg_path)
    return base


def test_basic_single_file_match(tmp_path, monkeypatch):
    _write(str(tmp_path), "fable.md", "FABLE_BODY")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["fable.md"]}])
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text == "FABLE_BODY"
    assert res.matched_rule == "*opus*"
    assert len(res.files) == 1


def test_stacking_order_preserved(tmp_path, monkeypatch):
    _write(str(tmp_path), "a.md", "AAA")
    _write(str(tmp_path), "b.md", "BBB")
    _write(str(tmp_path), "c.md", "CCC")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["a.md", "b.md", "c.md"]}])
    res = resolve_prelude("anthropic/claude-opus-4-6")
    # joined in the configured order, blank-line separated
    assert res.text == "AAA\n\nBBB\n\nCCC"


def test_first_match_wins_most_specific_first(tmp_path, monkeypatch):
    _write(str(tmp_path), "specific.md", "SPECIFIC")
    _write(str(tmp_path), "generic.md", "GENERIC")
    _make_env(
        tmp_path,
        monkeypatch,
        [
            {"match": "*opus-4-6*", "files": ["specific.md"]},
            {"match": "*opus*", "files": ["generic.md"]},
        ],
    )
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text == "SPECIFIC"
    assert res.matched_rule == "*opus-4-6*"


def test_bare_model_tail_matches(tmp_path, monkeypatch):
    _write(str(tmp_path), "g.md", "GPT")
    _make_env(tmp_path, monkeypatch, [{"match": "*gpt*", "files": ["g.md"]}])
    # bare id (no provider prefix) must also match
    assert resolve_prelude("gpt-5.5").text == "GPT"
    # provider/model form must match too
    assert resolve_prelude("openai/gpt-5.5").text == "GPT"


def test_case_insensitive(tmp_path, monkeypatch):
    _write(str(tmp_path), "g.md", "GEM")
    _make_env(tmp_path, monkeypatch, [{"match": "*gemini*", "files": ["g.md"]}])
    assert resolve_prelude("Google/Gemini-2.5-PRO").text == "GEM"


def test_no_match_returns_empty(tmp_path, monkeypatch):
    _write(str(tmp_path), "g.md", "X")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["g.md"]}])
    res = resolve_prelude("mistral/mistral-large")
    assert res.text == ""
    assert not res  # __bool__ is False


def test_disabled_returns_empty(tmp_path, monkeypatch):
    _write(str(tmp_path), "g.md", "X")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["g.md"]}], enabled=False)
    assert resolve_prelude("anthropic/claude-opus-4-6").text == ""


def test_missing_file_skipped_but_others_kept(tmp_path, monkeypatch):
    _write(str(tmp_path), "present.md", "PRESENT")
    _make_env(
        tmp_path,
        monkeypatch,
        [{"match": "*opus*", "files": ["missing.md", "present.md"]}],
    )
    res = resolve_prelude("anthropic/claude-opus-4-6")
    # missing file is skipped, present one still included
    assert res.text == "PRESENT"
    assert len(res.files) == 1


def test_all_missing_returns_empty(tmp_path, monkeypatch):
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["nope1.md", "nope2.md"]}])
    assert resolve_prelude("anthropic/claude-opus-4-6").text == ""


def test_layered_mode_concatenates_all_matching_rules(tmp_path, monkeypatch):
    _write(str(tmp_path), "base.md", "BASE")
    _write(str(tmp_path), "extra.md", "EXTRA")
    _make_env(
        tmp_path,
        monkeypatch,
        [
            {"match": "*opus*", "files": ["base.md"]},
            {"match": "*claude*", "files": ["extra.md"]},
        ],
        first_match=False,
    )
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text == "BASE\n\nEXTRA"


def test_dedupe_same_file_across_rules(tmp_path, monkeypatch):
    _write(str(tmp_path), "shared.md", "SHARED")
    _make_env(
        tmp_path,
        monkeypatch,
        [
            {"match": "*opus*", "files": ["shared.md"]},
            {"match": "*claude*", "files": ["shared.md"]},
        ],
        first_match=False,
    )
    res = resolve_prelude("anthropic/claude-opus-4-6")
    # shared.md included once, not twice
    assert res.text == "SHARED"
    assert len(res.files) == 1


def test_absolute_path_entry(tmp_path, monkeypatch):
    abs_file = _write(str(tmp_path), "abs.md", "ABSBODY")
    # base_dir points elsewhere; entry is an absolute path
    other = str(tmp_path / "other")
    os.makedirs(other, exist_ok=True)
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": [abs_file]}], base_dir=other)
    assert resolve_prelude("anthropic/claude-opus-4-6").text == "ABSBODY"


def test_empty_model_returns_empty(tmp_path, monkeypatch):
    _write(str(tmp_path), "g.md", "X")
    _make_env(tmp_path, monkeypatch, [{"match": "*", "files": ["g.md"]}])
    assert resolve_prelude("").text == ""
    assert resolve_prelude(None).text == ""


def test_no_config_returns_empty(monkeypatch):
    # No HERMES_PRELUDE_CONFIG and config.yaml has no block -> empty, no raise
    monkeypatch.delenv("HERMES_PRELUDE_CONFIG", raising=False)
    # Point at a nonexistent override to force the fail-soft path deterministically
    monkeypatch.setenv("HERMES_PRELUDE_CONFIG", "/nonexistent/path/xyz.yaml")
    assert resolve_prelude("anthropic/claude-opus-4-6").text == ""


def test_operating_mode_marker_prepended_when_mode_set(tmp_path, monkeypatch):
    _write(str(tmp_path), "f.md", "FABLE_BODY")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "operating_mode": "Fable 5", "files": ["f.md"]}])
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode == "Fable 5"
    assert res.text.lstrip().startswith("<policy_spec>")
    assert "</policy_spec>" in res.text and "<system-reminder>" in res.text  # hybrid: both tags
    assert "MANDATORY" in res.text                                          # the hard mandate
    assert "operating as Fable 5" in res.text  # the transparent self-description hook
    assert "FABLE_BODY" in res.text            # the prelude body still follows
    assert res.text.index('policy_spec') < res.text.index("FABLE_BODY")


def test_profile_alias_still_accepted(tmp_path, monkeypatch):
    """Backward-compat: the deprecated 'profile' key still names the mode."""
    _write(str(tmp_path), "f.md", "BODY")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "profile": "Fable 5", "files": ["f.md"]}])
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode == "Fable 5"
    assert res.text.lstrip().startswith("<policy_spec>")


def test_no_marker_when_mode_absent(tmp_path, monkeypatch):
    _write(str(tmp_path), "f.md", "BODY")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["f.md"]}])
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode is None
    assert "system-reminder" not in res.text and "policy_spec" not in res.text
    assert res.text == "BODY"


def test_custom_operating_mode_marker_template(tmp_path, monkeypatch):
    _write(str(tmp_path), "f.md", "BODY")
    _make_env(
        tmp_path, monkeypatch,
        [{"match": "*opus*", "operating_mode": "Fable 5", "files": ["f.md"]}],
        extra={"operating_mode_marker": "MODE={mode}!"},
    )
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text.startswith("MODE=Fable 5!")


def test_empty_marker_disables_marker_but_keeps_mode(tmp_path, monkeypatch):
    _write(str(tmp_path), "f.md", "BODY")
    _make_env(
        tmp_path, monkeypatch,
        [{"match": "*opus*", "operating_mode": "Fable 5", "files": ["f.md"]}],
        extra={"operating_mode_marker": ""},
    )
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode == "Fable 5"   # mode name still resolved
    assert "system-reminder" not in res.text and "policy_spec" not in res.text
    assert res.text == "BODY"               # but no marker text injected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
