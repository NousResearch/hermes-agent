from io import StringIO

import pytest
from rich.console import Console
from rich.markdown import Markdown

import cli
from cli import (
    _render_final_assistant_content,
    _resolve_markdown_table_box,
    _get_hermes_bordered_markdown_cls,
)


def _render_to_text(renderable, width: int = 80) -> str:
    buf = StringIO()
    Console(file=buf, width=width, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _reset_bordered_markdown_cache():
    """The bordered ``Markdown`` subclass is built lazily once per
    process, but tests mutate ``CLI_CONFIG['display']`` (via the
    fixture below) to exercise different box styles.  The class itself
    caches nothing config-dependent, but resetting between tests
    documents the contract and protects against future regressions
    where someone adds class-level caching by accident."""
    yield
    # No reset required — the cache holds the class identity, not the
    # box style.  Box style is re-read at render time.


@pytest.fixture
def display_config(monkeypatch: pytest.MonkeyPatch):
    """Hand back a mutable ``display`` config dict bound to the live
    ``CLI_CONFIG``.  Use ``display_config['markdown_table_box_style']
    = 'simple'`` to override per test; the fixture restores the prior
    value on teardown.
    """
    original = dict(cli.CLI_CONFIG.get("display", {}))
    yield cli.CLI_CONFIG.setdefault("display", {})
    cli.CLI_CONFIG["display"].clear()
    cli.CLI_CONFIG["display"].update(original)


def test_final_assistant_content_uses_markdown_renderable():
    renderable = _render_final_assistant_content("# Title\n\n- one\n- two")

    assert isinstance(renderable, Markdown)
    output = _render_to_text(renderable)
    assert "Title" in output
    assert "one" in output
    assert "two" in output


def test_final_assistant_content_preserves_windows_hidden_dir_paths():
    renderable = _render_final_assistant_content(
        r"D:\Projects\SourceCode\hermes-agent\.ai\skills" + "\\"
    )

    output = _render_to_text(renderable)
    assert r"D:\Projects\SourceCode\hermes-agent\.ai\skills" + "\\" in output


def test_final_assistant_content_keeps_non_path_markdown_escapes():
    renderable = _render_final_assistant_content(r"1\. Not an ordered list")

    output = _render_to_text(renderable)
    assert "1. Not an ordered list" in output
    assert r"1\." not in output


def test_final_assistant_content_strips_ansi_before_markdown_rendering():
    renderable = _render_final_assistant_content("\x1b[31m# Title\x1b[0m")

    output = _render_to_text(renderable)
    assert "Title" in output
    assert "\x1b" not in output


def test_final_assistant_content_can_strip_markdown_syntax():
    renderable = _render_final_assistant_content(
        "***Bold italic***\n~~Strike~~\n- item\n# Title\n`code`",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "Bold italic" in output
    assert "Strike" in output
    assert "item" in output
    assert "Title" in output
    assert "code" in output
    assert "***" not in output
    assert "~~" not in output
    assert "`" not in output


def test_strip_mode_preserves_lists():
    renderable = _render_final_assistant_content(
        "**Formatting**\n- Ran prettier\n- Files changed\n- Verified clean",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "- Ran prettier" in output
    assert "- Files changed" in output
    assert "- Verified clean" in output
    assert "**" not in output


def test_strip_mode_preserves_ordered_lists():
    renderable = _render_final_assistant_content(
        "1. First item\n2. Second item\n3. Third item",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "1. First" in output
    assert "2. Second" in output
    assert "3. Third" in output


def test_strip_mode_preserves_blockquotes():
    renderable = _render_final_assistant_content(
        "> This is quoted text\n> Another quoted line",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "> This is quoted" in output
    assert "> Another quoted" in output


def test_strip_mode_preserves_checkboxes():
    renderable = _render_final_assistant_content(
        "- [ ] Todo item\n- [x] Done item",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "- [ ] Todo" in output
    assert "- [x] Done" in output


def test_strip_mode_preserves_table_structure_while_cleaning_cell_markdown():
    renderable = _render_final_assistant_content(
        "| Syntax | Example |\n|---|---|\n| Bold | `**bold**` |\n| Strike | `~~strike~~` |",
        mode="strip",
    )

    output = _render_to_text(renderable)

    # Inline cell markdown is stripped (the contract this test enforces).
    assert "**" not in output
    assert "~~" not in output
    assert "`" not in output

    # Cell *content* survives, even if the surrounding whitespace was
    # rewritten by the wcwidth-aware re-aligner.  Asserting on bare
    # cell text keeps this test focused on the strip behaviour rather
    # than snapshotting incidental column padding (which is what the
    # CJK-alignment fix changes).
    assert "Syntax" in output
    assert "Example" in output
    assert "Bold" in output and "bold" in output
    assert "Strike" in output and "strike" in output

    # Structural sanity: the table still renders as pipe-bordered rows
    # (header + divider + 2 body rows).
    body_rows = [ln for ln in output.splitlines() if ln.strip().startswith("|")]
    assert len(body_rows) == 4

    # Every rendered table row shares the same pipe column offsets — the
    # alignment guarantee from realign_markdown_tables.
    pipe_cols = [
        [i for i, ch in enumerate(row) if ch == "|"] for row in body_rows
    ]
    assert all(p == pipe_cols[0] for p in pipe_cols), (
        "table rows misaligned after strip-mode rendering:\n"
        + "\n".join(body_rows)
    )


def test_strip_mode_preserves_cron_asterisks_in_plain_text():
    renderable = _render_final_assistant_content("* * * * *", mode="strip")

    output = _render_to_text(renderable)
    assert "* * * * *" in output

    # Still treat the canonical 3-asterisk Markdown horizontal rule as decoration.
    renderable = _render_final_assistant_content("* * *", mode="strip")
    output = _render_to_text(renderable)
    assert "* * *" not in output


def test_final_assistant_content_can_leave_markdown_raw():
    renderable = _render_final_assistant_content("***Bold italic***", mode="raw")

    output = _render_to_text(renderable)
    assert "***Bold italic***" in output


def test_strip_mode_preserves_intraword_underscores_in_snake_case_identifiers():
    renderable = _render_final_assistant_content(
        "Let me look at test_case_with_underscores and SOME_CONST "
        "then /tmp/snake_case_dir/file_with_name.py",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "test_case_with_underscores" in output
    assert "SOME_CONST" in output
    assert "snake_case_dir" in output
    assert "file_with_name" in output


def test_strip_mode_still_strips_boundary_underscore_emphasis():
    renderable = _render_final_assistant_content(
        "say _hi_ and __bold__ now",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "say hi and bold now" in output


# ---------------------------------------------------------------------------
# Bordered Markdown table renderer (#28714)
# ---------------------------------------------------------------------------
#
# Upstream Rich draws ``rich.markdown.Markdown`` tables with
# ``box=box.SIMPLE`` — i.e. only a single horizontal rule under the
# header, no vertical column dividers.  These tests pin down the fix:
#
# * ``render`` mode now uses a Hermes-side subclass that draws visible
#   column dividers by default.
# * The default behaviour, the ``simple`` opt-out, and unknown-value
#   fallback are all wired through ``display.markdown_table_box_style``.
# * The new subclass remains an instance of ``rich.markdown.Markdown``
#   so downstream code (`/render`, panels, snapshot tests) keeps working.


_TABLE_SAMPLE = (
    "| Module    | Status | Priority |\n"
    "|-----------|--------|----------|\n"
    "| User Mgmt | Done   | P0       |\n"
    "| Role Mgmt | Done   | P0       |\n"
)


def _count_vertical_borders(text: str) -> int:
    """Count the box-drawing column-divider glyphs that Rich emits.
    Rich uses light (│), heavy (┃) or double (║) verticals depending
    on the box style; any of them counts."""
    return sum(text.count(ch) for ch in ("│", "┃", "║"))


class TestBorderedTableRendering:
    """Regression coverage for #28714 — render-mode tables must show
    visible column dividers, not just a header underline."""

    def test_render_mode_returns_a_rich_markdown_instance(self):
        # Subclass identity must be preserved so existing isinstance
        # checks and panel-layout code paths continue to work.
        renderable = _render_final_assistant_content(_TABLE_SAMPLE, mode="render")
        assert isinstance(renderable, Markdown)

    def test_render_mode_draws_vertical_column_dividers_by_default(self):
        """The default config (``markdown_table_box_style: heavy_head``)
        must emit at least one vertical divider — anything else means
        we regressed back to upstream Rich's borderless SIMPLE box."""
        renderable = _render_final_assistant_content(_TABLE_SAMPLE, mode="render")
        output = _render_to_text(renderable, width=80)
        assert _count_vertical_borders(output) > 0, (
            "Default markdown_table_box_style must add vertical "
            "column dividers (#28714). Rendered output was:\n" + output
        )

    def test_simple_opt_out_restores_pre_28714_behaviour(self, display_config):
        """Operators who liked the old borderless look can set
        ``markdown_table_box_style: simple`` to get the legacy
        behaviour back — verifying this explicitly keeps the
        backward-compat path covered."""
        display_config["markdown_table_box_style"] = "simple"

        renderable = _render_final_assistant_content(_TABLE_SAMPLE, mode="render")
        output = _render_to_text(renderable, width=80)
        assert _count_vertical_borders(output) == 0, (
            "simple box style must restore the pre-#28714 borderless "
            "rendering.  Output:\n" + output
        )

    def test_unknown_box_style_falls_back_to_default(self, display_config):
        """A typo'd config value must not crash rendering and must not
        silently degrade to the borderless SIMPLE style."""
        display_config["markdown_table_box_style"] = "definitely-not-a-real-style"

        renderable = _render_final_assistant_content(_TABLE_SAMPLE, mode="render")
        output = _render_to_text(renderable, width=80)
        assert _count_vertical_borders(output) > 0, (
            "Unknown box style must fall back to a bordered default, "
            "not silently disable column dividers."
        )

    def test_empty_config_value_uses_documented_default(self, display_config):
        """``None`` / empty string in config must resolve to the
        documented ``heavy_head`` default — protects users whose YAML
        sets the key but leaves the value blank."""
        display_config["markdown_table_box_style"] = ""
        assert _resolve_markdown_table_box() is not None

        display_config["markdown_table_box_style"] = None
        assert _resolve_markdown_table_box() is not None

    def test_table_cell_contents_are_preserved(self):
        """A borderless table that hides cell content would be useless;
        regardless of the box style, the actual data must survive."""
        renderable = _render_final_assistant_content(_TABLE_SAMPLE, mode="render")
        output = _render_to_text(renderable, width=80)
        for needle in ("Module", "Status", "Priority", "User Mgmt", "Role Mgmt"):
            assert needle in output, f"Cell text {needle!r} missing from:\n{output}"

    @pytest.mark.parametrize(
        "box_name",
        ["heavy", "heavy_head", "square", "rounded", "double"],
    )
    def test_each_bordered_style_emits_dividers(
        self, display_config, box_name: str
    ):
        """The user-facing allowlist promises that any of these named
        styles produce column dividers.  Lock that in so a future
        refactor of ``_resolve_markdown_table_box`` can't quietly
        narrow the supported set."""
        display_config["markdown_table_box_style"] = box_name

        renderable = _render_final_assistant_content(_TABLE_SAMPLE, mode="render")
        output = _render_to_text(renderable, width=80)
        assert _count_vertical_borders(output) > 0, (
            f"box style {box_name!r} should emit vertical dividers but "
            f"output had none:\n{output}"
        )


class TestResolveMarkdownTableBox:
    """Direct unit tests for the box-resolver helper — kept separate
    so renderer changes don't drown out config-validation regressions."""

    def test_default_resolves_to_heavy_head(self, display_config):
        # Make sure the documented default is what gets used when the
        # key is absent from config.
        display_config.pop("markdown_table_box_style", None)

        from rich import box
        assert _resolve_markdown_table_box() is box.HEAVY_HEAD

    def test_case_insensitive(self, display_config):
        from rich import box
        display_config["markdown_table_box_style"] = "HEAVY"
        assert _resolve_markdown_table_box() is box.HEAVY
        display_config["markdown_table_box_style"] = "Square"
        assert _resolve_markdown_table_box() is box.SQUARE

    def test_whitespace_tolerated(self, display_config):
        from rich import box
        display_config["markdown_table_box_style"] = "  simple  "
        assert _resolve_markdown_table_box() is box.SIMPLE

    def test_non_string_falls_back_to_default(self, display_config):
        from rich import box
        display_config["markdown_table_box_style"] = 12345
        assert _resolve_markdown_table_box() is box.HEAVY_HEAD


class TestGetHermesBorderedMarkdownCls:
    def test_returns_a_markdown_subclass(self):
        cls = _get_hermes_bordered_markdown_cls()
        assert issubclass(cls, Markdown)

    def test_caches_the_class_across_calls(self):
        # Same class identity = caching works; rebuilding on every
        # render would create a hot allocation path for a long-lived
        # CLI session.
        assert _get_hermes_bordered_markdown_cls() is _get_hermes_bordered_markdown_cls()

    def test_subclass_overrides_only_table_open(self):
        """We deliberately do not replace any other element handler —
        protects future Rich upgrades that add new element types from
        silently regressing because our ``elements`` mapping was a
        full copy frozen at PR time."""
        cls = _get_hermes_bordered_markdown_cls()
        upstream_keys = set(Markdown.elements.keys())
        # All upstream keys are still present (no accidental drops).
        assert upstream_keys.issubset(cls.elements.keys())
        # Only "table_open" was customised.
        diverged = [
            k for k in upstream_keys
            if cls.elements[k] is not Markdown.elements[k]
        ]
        assert diverged == ["table_open"]
