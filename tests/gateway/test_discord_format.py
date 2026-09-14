"""Discord Markdown rendering, including block-aware table conversion."""

import types
import sys


def _make_discord_adapter():
    """Construct a DiscordAdapter with discord.py stubbed out."""
    fake_discord = types.ModuleType("discord")
    fake_discord.Intents = type("Intents", (), {"default": classmethod(lambda cls: cls())})
    fake_discord.Message = object
    fake_ext = types.ModuleType("discord.ext")
    fake_commands = types.ModuleType("discord.ext.commands")
    fake_ext.commands = fake_commands
    fake_discord.ext = fake_ext
    sys.modules.setdefault("discord", fake_discord)
    sys.modules.setdefault("discord.ext", fake_ext)
    sys.modules.setdefault("discord.ext.commands", fake_commands)

    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = object.__new__(DiscordAdapter)
    return adapter


class TestDiscordFormatMessage:

    def test_table_becomes_aligned_code_block(self):
        adapter = _make_discord_adapter()
        text = (
            "Results:\n\n"
            "| Name | Score |\n"
            "|------|-------|\n"
            "| Alice | 95   |\n"
            "| Bob   | 80   |\n"
            "\nDone."
        )
        out = adapter.format_message(text)
        assert "```text" in out
        assert "Alice" in out and "95" in out
        assert "Bob" in out and "80" in out
        assert out.startswith("Results:")
        assert out.rstrip().endswith("Done.")
        assert "|------|" not in out

    def test_table_like_text_inside_fence_is_unchanged(self):
        adapter = _make_discord_adapter()
        text = "Before\n```markdown\n| Name | Score |\n| --- | --- |\n| Alice | 95 |\n```\nAfter"

        assert adapter.format_message(text) == text

    def test_existing_discord_markdown_is_preserved(self):
        adapter = _make_discord_adapter()
        text = "## Heading\n- **bold** [link](https://example.com)\n> quoted"

        assert adapter.format_message(text) == text

    def test_wide_cells_align_by_display_width(self):
        from wcwidth import wcswidth

        adapter = _make_discord_adapter()
        out = adapter.format_message(
            "| Name | Score |\n| --- | --- |\n| 日本語 | 95 |\n| A | 80 |"
        )

        rows = [line for line in out.splitlines() if line.startswith("|")]
        assert len(rows) == 4
        assert wcswidth(rows[0][:rows[0].index("|", 1)]) == wcswidth(rows[2][:rows[2].index("|", 1)])


class TestDiscordMarkdownRenderer:
    def test_extracts_explicit_status_field_and_keeps_other_columns(self):
        from plugins.platforms.discord.markdown_renderer import render_discord_markdown

        rendered = render_discord_markdown(
            "| Service | Status | Latency |\n| --- | --- | --- |\n| API | healthy | 20 ms |"
        )

        assert rendered.fields == [("Status", "✅ healthy")]
        assert "API" in rendered.content
        assert "20 ms" in rendered.content
        assert "healthy" not in rendered.content

    def test_unknown_status_stays_in_pseudo_table(self):
        from plugins.platforms.discord.markdown_renderer import render_discord_markdown

        rendered = render_discord_markdown(
            "| Service | Status |\n| --- | --- |\n| API | calibrating |"
        )

        assert rendered.fields == []
        assert "calibrating" in rendered.content

    def test_explicit_statuses_use_expected_emoji(self):
        from plugins.platforms.discord.markdown_renderer import render_discord_markdown

        rendered = render_discord_markdown(
            "| Health |\n| --- |\n| healthy |\n| degraded |\n| failed |\n| critical |"
        )

        assert rendered.fields == [
            ("Health", "✅ healthy"),
            ("Health", "⚠️ degraded"),
            ("Health", "❌ failed"),
            ("Health", "🔴 critical"),
        ]
        assert "healthy" in rendered.content

    def test_multiple_tables_are_converted(self):
        from plugins.platforms.discord.markdown_renderer import render_discord_markdown

        rendered = render_discord_markdown(
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |"
        )

        assert rendered.content.count("```text") == 2
        assert all(value in rendered.content for value in ("1", "2", "3", "4"))

    def test_conversion_failure_returns_original_text(self, monkeypatch):
        from plugins.platforms.discord import markdown_renderer

        source = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        monkeypatch.setattr(markdown_renderer, "_render_table", lambda *_: (_ for _ in ()).throw(ValueError()))

        assert markdown_renderer.render_discord_markdown(source).content == source

    def test_long_pseudo_table_splits_as_complete_fences(self):
        adapter = _make_discord_adapter()
        rendered = adapter.format_message(
            "| Name | Note |\n| --- | --- |\n" + "\n".join(
                f"| item-{index} | {'x' * 40} |" for index in range(80)
            )
        )

        chunks = adapter.truncate_message(rendered, 300)
        assert len(chunks) > 1
        assert all(chunk.count("```") % 2 == 0 for chunk in chunks)

    def test_embed_builder_carries_status_fields(self):
        adapter = _make_discord_adapter()

        assert adapter._embed_for_status_fields([("Status", "✅ healthy")]) is not None


class TestDiscordToolPreviewFormatting:
    def test_truncated_url_keeps_full_click_target(self):
        from agent.display import ToolPreview

        adapter = _make_discord_adapter()
        url = "https://hermes-agent.nousresearch.com/docs/gateway/discord/tool-progress"
        visible = "https://hermes-agent.nousresearch..."

        out = adapter.format_tool_preview(ToolPreview(visible, truncated=True, url=url))

        assert out == f"[hermes-agent.nousresearch...](<{url}>)"

    def test_truncated_url_label_is_not_a_second_url_target(self):
        from agent.display import ToolPreview

        adapter = _make_discord_adapter()
        url = "https://centaur.run/secrets/advanced-permissioning"
        visible = "https://centaur.run/secrets/advanced-..."

        out = adapter.format_tool_preview(ToolPreview(visible, truncated=True, url=url))

        assert out == (
            "[centaur.run/secrets/advanced-...]"
            "(<https://centaur.run/secrets/advanced-permissioning>)"
        )

    def test_link_escapes_discord_markdown_delimiters(self):
        from agent.display import ToolPreview

        adapter = _make_discord_adapter()
        preview = ToolPreview(
            r"https://example.com/docs/[beta]...",
            truncated=True,
            url="https://example.com/docs/_(beta)",
        )

        assert adapter.format_tool_preview(preview) == (
            r"[example.com/docs/\[beta\]...]"
            r"(<https://example.com/docs/_%28beta%29>)"
        )

    def test_structured_tool_event_uses_clickable_truncated_url(self):
        from gateway.stream_events import ToolCallChunk

        adapter = _make_discord_adapter()
        url = "https://hermes-agent.nousresearch.com/docs/gateway/discord/tool-progress"
        visible = url[:37] + "..."

        out = adapter.format_tool_event(
            ToolCallChunk("web_extract", preview=url, args={"urls": [url]}),
            mode="all",
            preview_max_len=40,
        )

        assert out is not None
        assert f"[{visible.removeprefix('https://')}](<{url}>)" in out

    def test_untruncated_url_remains_plain(self):
        from agent.display import ToolPreview

        adapter = _make_discord_adapter()
        url = "https://example.com/page"

        out = adapter.format_tool_preview(ToolPreview(url))

        assert out == url

    def test_truncated_non_url_remains_plain(self):
        from agent.display import ToolPreview

        adapter = _make_discord_adapter()
        visible = "a long search query that was trunc..."

        out = adapter.format_tool_preview(ToolPreview(visible, truncated=True))

        assert out == visible
