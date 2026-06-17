"""Unit tests for the system-prompt prelude resolver.

Run:
    python -m pytest tests/agent/test_system_prompt_prelude.py -q

Config is read from ``config.yaml`` under a temp ``HERMES_HOME`` (no env
override): the resolver reads the ``system_prompt_prelude`` block via
``hermes_cli.config.load_config`` and derives its default base directory from
the profile-aware ``get_hermes_home()``. Tests set a temp ``HERMES_HOME`` and
write a real ``config.yaml`` there.
"""

from __future__ import annotations

import os

import pytest

from agent.system_prompt_prelude import resolve_prelude


def _write(d, name, body):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(body)
    return p


def _make_env(tmp_path, monkeypatch, rules, *, enabled=True, first_match=True,
              base_dir=None, extra=None, omit_base_dir=False):
    """Point a temp HERMES_HOME at a real config.yaml with the prelude block.

    Returns the effective base directory prelude files should live in. When
    ``omit_base_dir`` is set the block carries no ``base_dir`` so the resolver
    falls back to the profile-aware ``<hermes_home>/system-prompts``.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Drop any context-local override so get_hermes_home() honors HERMES_HOME.
    monkeypatch.delenv("HERMES_PRELUDE_CONFIG", raising=False)

    if omit_base_dir:
        base = str(home / "system-prompts")
    else:
        base = base_dir or str(tmp_path)

    blk = {"enabled": enabled, "first_match": first_match, "rules": rules}
    if not omit_base_dir:
        blk["base_dir"] = base
    if extra:
        blk.update(extra)

    import yaml

    with open(str(home / "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"system_prompt_prelude": blk}, fh)

    # Fresh temp home => unique config path => no load_config cache collision.
    return base


def test_basic_single_file_match(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["house.md"]}])
    _write(base, "house.md", "HOUSE_BODY")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text == "HOUSE_BODY"
    assert res.matched_rule == "*opus*"
    assert len(res.files) == 1


def test_stacking_order_preserved(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["a.md", "b.md", "c.md"]}])
    _write(base, "a.md", "AAA")
    _write(base, "b.md", "BBB")
    _write(base, "c.md", "CCC")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    # joined in the configured order, blank-line separated
    assert res.text == "AAA\n\nBBB\n\nCCC"


def test_first_match_wins_most_specific_first(tmp_path, monkeypatch):
    base = _make_env(
        tmp_path,
        monkeypatch,
        [
            {"match": "*opus-4-6*", "files": ["specific.md"]},
            {"match": "*opus*", "files": ["generic.md"]},
        ],
    )
    _write(base, "specific.md", "SPECIFIC")
    _write(base, "generic.md", "GENERIC")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text == "SPECIFIC"
    assert res.matched_rule == "*opus-4-6*"


def test_bare_model_tail_matches(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*gpt*", "files": ["g.md"]}])
    _write(base, "g.md", "GPT")
    # bare id (no provider prefix) must also match
    assert resolve_prelude("gpt-5.5").text == "GPT"
    # provider/model form must match too
    assert resolve_prelude("openai/gpt-5.5").text == "GPT"


def test_provider_arg_synthesizes_provider_model_candidate(tmp_path, monkeypatch):
    """A provider-qualified glob must match a BARE model when the provider is
    supplied separately (agent.model is bare, agent.provider carries it).

    This is the sweeper's primary concern: the ``provider`` argument at
    ``resolve_prelude(model, provider)`` must build a ``provider/model``
    candidate so ``anthropic/*`` matches ``claude-opus-4-6`` + ``anthropic``.
    """
    base = _make_env(tmp_path, monkeypatch, [{"match": "anthropic/*", "files": ["a.md"]}])
    _write(base, "a.md", "ANTHRO")
    # Bare model with no provider arg: the provider-qualified rule cannot match.
    assert resolve_prelude("claude-opus-4-6").text == ""
    # Bare model WITH provider arg: synthesized 'anthropic/claude-opus-4-6' matches.
    assert resolve_prelude("claude-opus-4-6", "anthropic").text == "ANTHRO"
    # Case-insensitive on the provider too.
    assert resolve_prelude("claude-opus-4-6", "Anthropic").text == "ANTHRO"


def test_provider_arg_not_doubled_when_model_already_qualified(tmp_path, monkeypatch):
    """When the model already carries a provider prefix, the provider arg must
    not produce a doubled ``anthropic/anthropic/...`` candidate."""
    base = _make_env(tmp_path, monkeypatch, [{"match": "anthropic/*", "files": ["a.md"]}])
    _write(base, "a.md", "ANTHRO")
    # model already 'anthropic/claude...' + redundant provider arg still matches once.
    res = resolve_prelude("anthropic/claude-opus-4-6", "anthropic")
    assert res.text == "ANTHRO"


def test_case_insensitive(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*gemini*", "files": ["g.md"]}])
    _write(base, "g.md", "GEM")
    assert resolve_prelude("Google/Gemini-2.5-PRO").text == "GEM"


def test_no_match_returns_empty(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["g.md"]}])
    _write(base, "g.md", "X")
    res = resolve_prelude("mistral/mistral-large")
    assert res.text == ""
    assert not res  # __bool__ is False


def test_disabled_returns_empty(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["g.md"]}], enabled=False)
    _write(base, "g.md", "X")
    assert resolve_prelude("anthropic/claude-opus-4-6").text == ""


def test_missing_file_skipped_but_others_kept(tmp_path, monkeypatch):
    base = _make_env(
        tmp_path,
        monkeypatch,
        [{"match": "*opus*", "files": ["missing.md", "present.md"]}],
    )
    _write(base, "present.md", "PRESENT")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    # missing file is skipped, present one still included
    assert res.text == "PRESENT"
    assert len(res.files) == 1


def test_all_missing_returns_empty(tmp_path, monkeypatch):
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["nope1.md", "nope2.md"]}])
    assert resolve_prelude("anthropic/claude-opus-4-6").text == ""


def test_layered_mode_concatenates_all_matching_rules(tmp_path, monkeypatch):
    base = _make_env(
        tmp_path,
        monkeypatch,
        [
            {"match": "*opus*", "files": ["base.md"]},
            {"match": "*claude*", "files": ["extra.md"]},
        ],
        first_match=False,
    )
    _write(base, "base.md", "BASE")
    _write(base, "extra.md", "EXTRA")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text == "BASE\n\nEXTRA"


def test_dedupe_same_file_across_rules(tmp_path, monkeypatch):
    base = _make_env(
        tmp_path,
        monkeypatch,
        [
            {"match": "*opus*", "files": ["shared.md"]},
            {"match": "*claude*", "files": ["shared.md"]},
        ],
        first_match=False,
    )
    _write(base, "shared.md", "SHARED")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    # shared.md included once, not twice
    assert res.text == "SHARED"
    assert len(res.files) == 1


def test_absolute_path_entry(tmp_path, monkeypatch):
    # base_dir points elsewhere; entry is an absolute path outside it.
    other = str(tmp_path / "other")
    os.makedirs(other, exist_ok=True)
    abs_file = _write(str(tmp_path / "elsewhere"), "abs.md", "ABSBODY")
    _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": [abs_file]}], base_dir=other)
    assert resolve_prelude("anthropic/claude-opus-4-6").text == "ABSBODY"


def test_default_base_dir_is_profile_aware_hermes_home(tmp_path, monkeypatch):
    """With no ``base_dir`` configured, relative files resolve under the
    profile-aware ``<hermes_home>/system-prompts`` (honors HERMES_HOME), not a
    hardcoded ``~/.hermes``."""
    base = _make_env(
        tmp_path, monkeypatch,
        [{"match": "*opus*", "files": ["op.md"]}],
        omit_base_dir=True,
    )
    # base == <hermes_home>/system-prompts
    assert base.endswith(os.path.join("home", "system-prompts"))
    _write(base, "op.md", "FROM_HOME")
    assert resolve_prelude("anthropic/claude-opus-4-6").text == "FROM_HOME"


def test_empty_model_returns_empty(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*", "files": ["g.md"]}])
    _write(base, "g.md", "X")
    assert resolve_prelude("").text == ""
    assert resolve_prelude(None).text == ""


def test_no_config_returns_empty(tmp_path, monkeypatch):
    # A temp HERMES_HOME with no prelude block -> empty, no raise.
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PRELUDE_CONFIG", raising=False)
    with open(str(home / "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write("model: \"\"\n")
    assert resolve_prelude("anthropic/claude-opus-4-6").text == ""


def test_operating_mode_marker_prepended_when_mode_set(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "operating_mode": "House", "files": ["f.md"]}])
    _write(base, "f.md", "PRELUDE_BODY")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode == "House"
    assert res.text.lstrip().startswith("<policy_spec>")
    assert "</policy_spec>" in res.text and "<system-reminder>" in res.text  # hybrid: both tags
    assert "MANDATORY" in res.text                                          # the hard mandate
    assert "operating as House" in res.text   # the transparent self-description hook
    assert "PRELUDE_BODY" in res.text         # the prelude body still follows
    assert res.text.index("policy_spec") < res.text.index("PRELUDE_BODY")


def test_profile_alias_still_accepted(tmp_path, monkeypatch):
    """Backward-compat: the deprecated 'profile' key still names the mode."""
    base = _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "profile": "House", "files": ["f.md"]}])
    _write(base, "f.md", "BODY")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode == "House"
    assert res.text.lstrip().startswith("<policy_spec>")


def test_no_marker_when_mode_absent(tmp_path, monkeypatch):
    base = _make_env(tmp_path, monkeypatch, [{"match": "*opus*", "files": ["f.md"]}])
    _write(base, "f.md", "BODY")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode is None
    assert "system-reminder" not in res.text and "policy_spec" not in res.text
    assert res.text == "BODY"


def test_custom_operating_mode_marker_template(tmp_path, monkeypatch):
    base = _make_env(
        tmp_path, monkeypatch,
        [{"match": "*opus*", "operating_mode": "House", "files": ["f.md"]}],
        extra={"operating_mode_marker": "MODE={mode}!"},
    )
    _write(base, "f.md", "BODY")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.text.startswith("MODE=House!")


def test_empty_marker_disables_marker_but_keeps_mode(tmp_path, monkeypatch):
    base = _make_env(
        tmp_path, monkeypatch,
        [{"match": "*opus*", "operating_mode": "House", "files": ["f.md"]}],
        extra={"operating_mode_marker": ""},
    )
    _write(base, "f.md", "BODY")
    res = resolve_prelude("anthropic/claude-opus-4-6")
    assert res.operating_mode == "House"   # mode name still resolved
    assert "system-reminder" not in res.text and "policy_spec" not in res.text
    assert res.text == "BODY"              # but no marker text injected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
