"""requires_toolsets gate reconciles the singular/plural toolset drift (#99877).

Skill frontmatter is hand-authored, so the natural plural ``files`` diverges
from the canonical toolset id ``file``. The visibility gate used exact string
membership, so a skill declaring ``requires_toolsets: [files]`` failed the gate
in every toolset-filtered session and was silently dropped. These tests pin
that the canonical form is matched, that genuine typos are still gated out, and
that exact-name behaviour (and ``requires_tools``) is unchanged.
"""
from agent.prompt_builder import _skill_should_show


def test_plural_toolset_name_matches_canonical_singular():
    # The regression: 'files' must satisfy a session that offers 'file'.
    assert _skill_should_show({"requires_toolsets": ["files"]}, set(), {"file"}) is True


def test_canonical_singular_still_matches():
    assert _skill_should_show({"requires_toolsets": ["file"]}, set(), {"file"}) is True


def test_required_toolset_absent_still_gates_out():
    # Whichever spelling, if the session genuinely lacks it, hide the skill.
    assert _skill_should_show({"requires_toolsets": ["file"]}, set(), set()) is False
    assert _skill_should_show({"requires_toolsets": ["files"]}, set(), set()) is False


def test_unknown_toolset_name_is_not_force_matched():
    # A real typo (no singular/plural twin in the registry) must NOT be
    # normalized into a match — the skill stays correctly gated.
    assert _skill_should_show({"requires_toolsets": ["filez"]}, set(), {"file"}) is False


def test_terminal_and_files_combo_loads_with_canonical_session():
    # research-paper-writing shape: requires_toolsets: [terminal, files].
    conds = {"requires_toolsets": ["terminal", "files"]}
    assert _skill_should_show(conds, set(), {"terminal", "file"}) is True
    # Missing one of them still gates out.
    assert _skill_should_show(conds, set(), {"terminal"}) is False


def test_fallback_for_toolset_plural_hides_when_canonical_present():
    # fallback_for hides the skill when the primary IS available; the plural
    # drift previously left it wrongly shown.
    conds = {"fallback_for_toolsets": ["files"]}
    assert _skill_should_show(conds, set(), {"file"}) is False   # file present -> hide
    assert _skill_should_show(conds, set(), set()) is True        # absent -> show


def test_non_string_requires_toolsets_element_gates_out():
    # An empty `requires_toolsets:` value parses to None; malformed frontmatter
    # may also carry non-string elements. These must gate the skill out (fail
    # closed), never raise — a crash here aborts the whole index build, which is
    # a worse failure mode than silently hiding one skill.
    assert _skill_should_show({"requires_toolsets": [None]}, set(), {"file"}) is False
    assert _skill_should_show({"requires_toolsets": [123]}, set(), {"file"}) is False
    assert _skill_should_show({"requires_toolsets": ["files", None]}, set(), {"file"}) is False
    # fallback_for with a bad element likewise must not raise.
    assert _skill_should_show({"fallback_for_toolsets": [None]}, set(), {"file"}) is True


def test_requires_tools_exact_match_unchanged():
    # Tool names are exact identifiers (no singular/plural form) — behaviour
    # must be untouched by the toolset canonicalization.
    assert _skill_should_show({"requires_tools": ["read_file"]}, {"read_file"}, set()) is True
    assert _skill_should_show({"requires_tools": ["read_file"]}, set(), set()) is False
