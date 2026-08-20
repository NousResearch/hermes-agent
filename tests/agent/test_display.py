"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import base64
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import agent.display as display_module
from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    get_cute_tool_message,
    prepare_tool_preview,
    redact_tool_args_for_display,
    set_tool_preview_max_len,
    _render_inline_unified_diff,
    _summarize_rendered_diff_sections,
    render_edit_diff_with_delta,
)


@pytest.fixture(autouse=True)
def reset_tool_preview_max_len():
    set_tool_preview_max_len(0)
    yield
    set_tool_preview_max_len(0)


def test_cute_tool_message_falls_back_when_renderer_raises(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("cosmetic failure")

    monkeypatch.setattr(display_module, "_get_cute_tool_message", _boom)

    assert get_cute_tool_message("web_extract", {"urls": []}, 0.25) == (
        "┊ ⚡ web_extra completed  0.2s"
    )


class TestBuildToolPreview:
    """Tests for build_tool_preview defensive handling and normal operation."""

    def test_none_args_returns_none(self):
        """PR #453: None args should not crash, should return None."""
        assert build_tool_preview("terminal", None) is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no keys to preview."""
        assert build_tool_preview("terminal", {}) is None








    def test_browser_type_preview_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": secret})
        assert result is not None
        assert secret not in result
        assert "sk-pro" in result and "..." in result

    def test_browser_type_preview_keeps_normal_text(self):
        text = "hello world search query"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": text})
        assert result is not None
        assert text in result

    def test_browser_type_display_args_redact_api_key(self):
        secret = "ghp_ABCDEFGHIJ1234567890"
        safe_args = redact_tool_args_for_display(
            "browser_type", {"ref": "@e3", "text": secret}
        )
        assert secret not in str(safe_args)
        assert safe_args["ref"] == "@e3"
        assert safe_args["text"].startswith("ghp_AB")















    def test_delegate_task_batch_preview_respects_max_len(self):
        result = build_tool_preview(
            "delegate_task",
            {"tasks": [{"goal": "A" * 80}, {"goal": "B" * 80}]},
            max_len=30,
        )
        assert result == "2 tasks: AAAAAAAAAAAAAAAAAA..."
        assert len(result) == 30

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


class TestPrepareToolPreview:
    def test_recovers_and_describes_truncated_url(self):
        url = "https://example.com/a/very/long/path/to/a/page"
        set_tool_preview_max_len(20)

        preview = prepare_tool_preview(
            "web_extract",
            {"urls": [url]},
            fallback=url[:17] + "...",
            max_len=20,
        )

        assert preview.text == url[:17] + "..."
        assert preview.truncated is True
        assert preview.url == url

    def test_untruncated_url_has_no_link_target(self):
        url = "https://example.com/page"
        preview = prepare_tool_preview(
            "browser_navigate", None, fallback=url, max_len=40
        )

        assert preview.text == url
        assert preview.truncated is False
        assert preview.url is None

    def test_truncated_non_url_has_no_link_target(self):
        preview = prepare_tool_preview(
            "web_search",
            {"query": "how to parse a URL"},
            fallback="how to parse a URL",
            max_len=12,
        )

        assert preview.truncated is True
        assert preview.url is None


class TestCuteToolMessagePreviewLength:


    def test_search_files_preview_uses_positive_configured_limit_not_default(self):
        set_tool_preview_max_len(80)
        pattern = "function.formatToolCall.context.preview.compactPreview.maxLength.truncate"

        line = get_cute_tool_message("search_files", {"pattern": pattern}, 0.1)

        assert pattern in line
        assert "..." not in line





    def test_browser_type_cute_message_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@password", "text": secret},
            0.1,
            result='{"success": true, "typed": "sk-pro...EFGH"}',
        )

        assert secret not in line
        assert "sk-pro" in line

    def test_browser_type_cute_message_keeps_normal_text(self):
        text = "hello world"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@search", "text": text},
            0.1,
            result='{"success": true, "typed": "hello world"}',
        )

        assert text in line


class TestEditDiffPreview:



    def test_extract_edit_diff_uses_local_snapshot_for_write_file(self, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("old\n", encoding="utf-8")

        snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})

        target.write_text("new\n", encoding="utf-8")

        diff = extract_edit_diff(
            "write_file",
            '{"bytes_written": 4}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert diff is not None
        assert "--- a/" in diff
        assert "+++ b/" in diff
        assert "-old" in diff
        assert "+new" in diff



    def test_render_edit_diff_with_delta_handles_renderer_errors(self, monkeypatch):
        printer = MagicMock()

        monkeypatch.setattr("agent.display._summarize_rendered_diff_sections", MagicMock(side_effect=RuntimeError("boom")))

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n"}',
            print_fn=printer,
        )

        assert rendered is False
        assert printer.call_count == 0


    def test_summarize_rendered_diff_sections_limits_file_count(self):
        diff = "".join(
            f"--- a/file{i}.py\n+++ b/file{i}.py\n+line{i}\n"
            for i in range(8)
        )

        rendered = _summarize_rendered_diff_sections(diff, max_files=3, max_lines=50)

        assert any("a/file0.py" in line for line in rendered)
        assert any("a/file1.py" in line for line in rendered)
        assert any("a/file2.py" in line for line in rendered)
        assert not any("a/file7.py" in line for line in rendered)
        assert "additional file" in rendered[-1]


class TestBuildToolLabel:
    """Friendly human-phrased tool labels for built-in tools."""

    @pytest.fixture(autouse=True)
    def _enable_friendly(self):
        from agent.display import set_friendly_tool_labels
        set_friendly_tool_labels(True)
        yield
        set_friendly_tool_labels(True)

    def test_web_search_uses_for_connector(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_search", {"query": "weather in NYC"})
        assert label == 'Searching the web for weather in NYC'

    def test_web_extract_reads_url(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_extract", {"urls": ["https://example.com/page"]})
        assert label is not None
        assert label.startswith("Reading ")
        assert "example.com/page" in label







    def test_disabled_falls_back_to_preview(self):
        from agent.display import (
            build_tool_label,
            build_tool_preview,
            set_friendly_tool_labels,
        )
        set_friendly_tool_labels(False)
        args = {"query": "weather in NYC"}
        label = build_tool_label("web_search", args)
        # With the feature off, must match the raw preview exactly
        assert label == build_tool_preview("web_search", args)
        assert "Searching the web" not in (label or "")



class TestBuildStatusPhrase:
    """build_status_phrase — live working-state text for Slack's status line."""



    def test_verb_only_when_args_none(self):
        # live_status: "verb" mode passes args=None to suppress previews.
        from agent.display import build_status_phrase
        assert build_status_phrase("terminal", None) == "is running…"
        assert build_status_phrase("read_file", None) == "is reading…"



    def test_caps_length_for_slack_status_line(self):
        from agent.display import build_status_phrase
        phrase = build_status_phrase(
            "terminal", {"command": "x" * 300}, max_len=49
        )
        assert phrase is not None and len(phrase) <= 49
        assert phrase.endswith("…")


    def test_respects_friendly_labels_toggle(self):
        from agent.display import build_status_phrase, set_friendly_tool_labels
        set_friendly_tool_labels(False)
        try:
            assert build_status_phrase("terminal", {"command": "ls"}) is None
        finally:
            set_friendly_tool_labels(True)


# =========================================================================
# Inline image preview (display.image_preview) — issue #6675
# =========================================================================


@pytest.fixture(autouse=True)
def reset_image_preview_latches():
    """Restore the default image-preview latches around every test."""
    display_module.set_image_preview(True, 0)
    yield
    display_module.set_image_preview(True, 0)


def _make_image(tmp_path, name="shot.png", size=64) -> str:
    img = tmp_path / name
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * size)
    return str(img)


class TestImagePreviewLatches:
    def test_defaults_enabled_natural_size(self):
        assert display_module._image_preview_enabled is True
        assert display_module._image_preview_max_width == 0

    def test_setter_coerces_values(self):
        # Deliberately wrong-typed inputs: the setter is defensive against
        # YAML-typed config values (bool()/int() coercion).
        display_module.set_image_preview(0, -5)  # type: ignore[arg-type]
        assert display_module.get_image_preview_enabled() is False
        assert display_module.get_image_preview_max_width() == 0
        display_module.set_image_preview("yes", "400")  # type: ignore[arg-type]
        assert display_module.get_image_preview_enabled() is True
        assert display_module.get_image_preview_max_width() == 400

    def test_setter_malformed_width_keeps_explicit_disable(self):
        # cli.py passes raw YAML values through; a malformed width must not
        # abort the latch update or re-enable previews (int(_ipw) at the call
        # site previously raised before the setter ever ran).
        display_module.set_image_preview(False, "400px")  # type: ignore[arg-type]
        assert display_module.get_image_preview_enabled() is False
        assert display_module.get_image_preview_max_width() == 0


class TestExtractImagePaths:
    def test_quoted_absolute_path(self, tmp_path):
        img = _make_image(tmp_path)
        assert display_module._extract_image_paths(f"Screenshot saved to '{img}'") == [img]

    def test_json_embedded_path(self, tmp_path):
        img = _make_image(tmp_path)
        assert display_module._extract_image_paths(f'{{"path": "{img}", "w": 800}}') == [img]

    def test_bare_absolute_path(self, tmp_path):
        img = _make_image(tmp_path)
        assert display_module._extract_image_paths(f"see {img} for details") == [img]

    def test_hyphenated_filename(self, tmp_path):
        img = _make_image(tmp_path, "screenshot-2026-01-15.png")
        assert display_module._extract_image_paths(f"Saved {img}") == [img]

    def test_uppercase_extension(self, tmp_path):
        img = _make_image(tmp_path, "photo.JPEG")
        assert display_module._extract_image_paths(f"Saved {img}") == [img]

    def test_relative_existing_resolves_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rel.png").write_bytes(b"x")
        assert display_module._extract_image_paths("wrote rel.png") == [str(tmp_path / "rel.png")]

    def test_quoted_path_with_spaces(self, tmp_path):
        img = tmp_path / "my shot.png"
        img.write_bytes(b"x")
        assert display_module._extract_image_paths(f"Saved '{img}'") == [str(img)]
        assert display_module._extract_image_paths(f'Saved "{img}"') == [str(img)]

    def test_json_path_with_spaces(self, tmp_path):
        img = tmp_path / "my shot.png"
        img.write_bytes(b"x")
        assert display_module._extract_image_paths(f'{{"path": "{img}"}}') == [str(img)]

    def test_quoted_sentence_with_path_uses_bare_fallback(self, tmp_path):
        img = _make_image(tmp_path)
        assert display_module._extract_image_paths(f'"Saved screenshot to {img}"') == [img]
        assert display_module._extract_image_paths(f"'Result image: {img}'") == [img]

    def test_bare_token_boundary_avoids_longer_filenames(self, tmp_path):
        real = _make_image(tmp_path, "image.png")
        (tmp_path / "image.png.bak").write_bytes(b"x")
        (tmp_path / "image.pngzip").write_bytes(b"x")
        text = f"touched {tmp_path / 'image.png.bak'} and {tmp_path / 'image.pngzip'}"
        assert display_module._extract_image_paths(text) == []
        assert display_module._extract_image_paths(f"real one: {real}") == [real]

    def test_sentence_period_after_path_still_extracts(self, tmp_path):
        img = _make_image(tmp_path)
        assert display_module._extract_image_paths(f"see {img}. next") == [img]
        assert display_module._extract_image_paths(f"saved to '{img}'.") == [img]
        # Period INSIDE the closing quote — the bare boundary lookahead must
        # allow a trailing '.' when it is not a filename continuation.
        assert display_module._extract_image_paths(f'"Saved screenshot to {img}."') == [img]

    def test_http_url_excluded(self, tmp_path):
        assert display_module._extract_image_paths("see https://example.com/img.png") == []

    def test_data_uri_excluded(self):
        assert display_module._extract_image_paths("data:image/png;base64,AAAA") == []

    def test_missing_file_excluded(self, tmp_path):
        assert display_module._extract_image_paths(str(tmp_path / "gone.png")) == []

    def test_dedupe_preserves_order(self, tmp_path):
        img = _make_image(tmp_path)
        assert display_module._extract_image_paths(f"one {img} two {img}") == [img]

    def test_cap_at_three(self, tmp_path):
        imgs = [_make_image(tmp_path, f"a{i}.png") for i in range(4)]
        found = display_module._extract_image_paths(" ".join(imgs))
        assert found == imgs[:3]

    def test_does_not_match_directory_named_like_image(self, tmp_path):
        d = tmp_path / "dir.png"
        d.mkdir()
        assert display_module._extract_image_paths(str(d)) == []

    def test_quoted_run_of_two_paths_both_found(self, tmp_path):
        # Regression: a quoted string wrapping two image paths was captured
        # as ONE token that did not exist as a file, hiding both paths.
        a = _make_image(tmp_path, "a.png")
        b = _make_image(tmp_path, "b.png")
        assert display_module._extract_image_paths(f"compare '{a} and {b}'") == [a, b]

    def test_relative_resolves_against_base_dir(self, tmp_path):
        img = tmp_path / "out.png"
        img.write_bytes(b"x")
        assert display_module._extract_image_paths("wrote out.png", base_dir=str(tmp_path)) == [str(img)]

    def test_base_dir_ignored_when_not_absolute(self, tmp_path, monkeypatch):
        img = tmp_path / "out.png"
        img.write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        assert display_module._extract_image_paths("wrote out.png", base_dir="relative/base") == [str(img)]


class TestITerm2Escape:
    def test_structure_and_base64_roundtrip(self, tmp_path):
        img = _make_image(tmp_path)
        esc = display_module._iterm2_image_escape(Path(img), 0)
        assert esc.startswith("\033]1337;File=inline=1;preserveAspectRatio=1:")
        assert esc.endswith("\a")
        payload = esc.split(":", 1)[1][:-1]
        assert base64.b64decode(payload) == Path(img).read_bytes()

    def test_width_clause_only_when_positive(self, tmp_path):
        img = _make_image(tmp_path)
        assert ";width=" not in display_module._iterm2_image_escape(Path(img), 0)
        assert ";width=400px:" in display_module._iterm2_image_escape(Path(img), 400)


class TestKittyEscapes:
    def test_multi_chunk_framing_and_roundtrip(self, tmp_path):
        img = _make_image(tmp_path, "big.png", size=20000)
        chunks = display_module._kitty_image_escapes(Path(img))
        assert len(chunks) > 1
        assert chunks[0].startswith("\033_Ga=T,f=100,t=d,m=1;")
        assert all(c.startswith("\033_Gm=1;") for c in chunks[1:-1])
        assert chunks[-1].startswith("\033_Gm=0;")
        payloads = [c.split(";", 1)[1].rsplit("\033\\", 1)[0] for c in chunks]
        assert all(len(p) <= display_module._KITTY_CHUNK_SIZE for p in payloads)
        assert base64.b64decode("".join(payloads)) == Path(img).read_bytes()

    def test_single_chunk_is_final(self, tmp_path):
        img = _make_image(tmp_path)
        chunks = display_module._kitty_image_escapes(Path(img))
        assert len(chunks) == 1
        assert chunks[0].startswith("\033_Ga=T,f=100,t=d,m=0;")

    def test_empty_file_single_final_chunk(self, tmp_path):
        img = tmp_path / "empty.png"
        img.write_bytes(b"")
        chunks = display_module._kitty_image_escapes(Path(img))
        assert chunks == ["\033_Ga=T,f=100,t=d,m=0;\033\\"]


class TestTerminalDetection:
    def test_iterm2(self, monkeypatch):
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        assert display_module._terminal_supports_inline_images() == "iterm2"

    def test_kitty_term(self, monkeypatch):
        # Hermetic: must clear TERM_PROGRAM/KITTY_WINDOW_ID or this fails
        # when the suite runs from inside iTerm2 (checked first).
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.setenv("TERM", "xterm-kitty")
        assert display_module._terminal_supports_inline_images() == "kitty"

    def test_kitty_window_id(self, monkeypatch):
        monkeypatch.setenv("KITTY_WINDOW_ID", "1")
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert display_module._terminal_supports_inline_images() == "kitty"

    def test_wezterm(self, monkeypatch):
        monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
        assert display_module._terminal_supports_inline_images() == "kitty"

    def test_wezterm_term_env(self, monkeypatch):
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.setenv("TERM", "xterm-wezterm")
        assert display_module._terminal_supports_inline_images() == "kitty"

    def test_unsupported_terminal(self, monkeypatch):
        monkeypatch.setenv("TERM_PROGRAM", "")
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        assert display_module._terminal_supports_inline_images() == ""


class TestRenderImagePreview:
    def _tty_and_terminal(self, monkeypatch, protocol="iterm2"):
        monkeypatch.setattr(display_module, "_stdout_is_tty", lambda: True)
        monkeypatch.setattr(display_module, "_terminal_supports_inline_images", lambda: protocol)

    def test_disabled_latch_emits_nothing(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch)
        display_module.set_image_preview(False, 0)
        calls = []
        assert display_module.render_image_preview(_make_image(tmp_path), print_fn=calls.append) is False
        assert calls == []

    def test_non_tty_emits_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(display_module, "_stdout_is_tty", lambda: False)
        calls = []
        assert display_module.render_image_preview(_make_image(tmp_path), print_fn=calls.append) is False
        assert calls == []

    def test_unsupported_terminal_without_chafa_emits_nothing(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="")
        monkeypatch.setenv("PATH", str(tmp_path))  # no chafa binary anywhere
        calls = []
        assert display_module.render_image_preview(_make_image(tmp_path), print_fn=calls.append) is False
        assert calls == []

    def test_iterm2_emits_label_and_wrapped_escape(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="iterm2")
        img = _make_image(tmp_path)
        calls = []
        assert display_module.render_image_preview(f"Saved {img}", print_fn=calls.append) is True
        escape = display_module._iterm2_image_escape(Path(img), 0)
        assert calls == ["  ┊ image", "\001" + escape + "\002"]

    def test_kitty_emits_label_and_single_wrapped_fragment(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="kitty")
        img = _make_image(tmp_path)
        calls = []
        assert display_module.render_image_preview(f"Saved {img}", print_fn=calls.append) is True
        joined = "".join(display_module._kitty_image_escapes(Path(img)))
        # All chunks in ONE print_fn call: _cprint adds a newline per call,
        # and inter-chunk newlines would drift the cursor before render.
        assert calls == ["  ┊ image", "\001" + joined + "\002"]

    def test_size_cap_skips_large_file(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="iterm2")
        big = tmp_path / "big.png"
        big.write_bytes(b"x" * (display_module._MAX_IMAGE_PREVIEW_BYTES + 1))
        small = _make_image(tmp_path, "small.png")
        calls = []
        assert display_module.render_image_preview(f"{big} then {small}", print_fn=calls.append) is True
        assert calls == ["  ┊ image", "\001" + display_module._iterm2_image_escape(Path(small), 0) + "\002"]

    def test_kitty_renders_file_under_cap(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="kitty")
        img = _make_image(tmp_path, "medium.png", size=1_500_000)  # 1.5MB < 5MB cap
        calls = []
        assert display_module.render_image_preview(f"Saved {img}", print_fn=calls.append) is True
        assert calls[0] == "  ┊ image"
        assert len(calls) == 2  # label + one joined fragment (no per-chunk newlines)

    def test_kitty_non_png_falls_back_to_chafa(self, tmp_path, monkeypatch):
        # kitty graphics protocol only accepts PNG (f=100) or raw RGB/RGBA —
        # JPEG must fall back to chafa when available.
        self._tty_and_terminal(monkeypatch, protocol="kitty")
        jpg = _make_image(tmp_path, "photo.jpg")
        fake = tmp_path / "chafa"
        fake.write_text("#!/bin/sh\nprintf '\\033[31m█\\033[0m\\n'\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        calls = []
        assert display_module.render_image_preview(f"Saved {jpg}", print_fn=calls.append) is True
        assert calls == ["  ┊ image", "\x1b[31m█\x1b[0m\n"]

    def test_kitty_non_png_without_chafa_emits_nothing(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="kitty")
        jpg = _make_image(tmp_path, "photo.jpg")
        monkeypatch.setenv("PATH", str(tmp_path))  # no chafa binary
        calls = []
        assert display_module.render_image_preview(f"Saved {jpg}", print_fn=calls.append) is False
        assert calls == []

    def test_max_width_plumbed_to_iterm2_escape(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="iterm2")
        display_module.set_image_preview(True, 400)
        img = _make_image(tmp_path)
        calls = []
        assert display_module.render_image_preview(f"Saved {img}", print_fn=calls.append) is True
        assert ";width=400px:" in calls[1]

    def test_print_fn_none_emits_nothing(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch)
        assert display_module.render_image_preview(_make_image(tmp_path), print_fn=None) is False

    def test_chafa_fallback_uses_path_binary(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="")
        img = _make_image(tmp_path)
        fake = tmp_path / "chafa"
        fake.write_text("#!/bin/sh\nprintf '\\033[31m█\\033[0m\\n'\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        calls = []
        assert display_module.render_image_preview(f"Saved {img}", print_fn=calls.append) is True
        assert calls == ["  ┊ image", "\x1b[31m█\x1b[0m\n"]

    def test_chafa_skips_file_over_cap(self, tmp_path, monkeypatch):
        # The 5 MiB cap applies to the chafa path too, not just the
        # iTerm2/kitty escape protocols — a huge file must not be handed to
        # chafa for a potentially long decode.
        self._tty_and_terminal(monkeypatch, protocol="")
        big = tmp_path / "big.png"
        big.write_bytes(b"x" * (display_module._MAX_IMAGE_PREVIEW_BYTES + 1))
        fake = tmp_path / "chafa"
        fake.write_text("#!/bin/sh\nprintf 'x\\n'\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        calls = []
        assert display_module.render_image_preview(str(big), print_fn=calls.append) is False
        assert calls == []

    def test_time_budget_breaks_loop(self, tmp_path, monkeypatch):
        # A pathological set of previews must not stall the CLI main thread
        # for the full 3 × 15s worst case: once the cumulative budget is
        # exceeded, remaining images are skipped.
        self._tty_and_terminal(monkeypatch, protocol="iterm2")
        imgs = [_make_image(tmp_path, f"a{i}.png") for i in range(3)]
        calls = []
        vals = [0.0, 0.1, 25.0]

        def fake_monotonic():
            return vals.pop(0) if vals else 25.0

        monkeypatch.setattr(display_module.time, "monotonic", fake_monotonic)
        assert display_module.render_image_preview(" ".join(imgs), print_fn=calls.append) is True
        assert calls == ["  ┊ image", "\001" + display_module._iterm2_image_escape(Path(imgs[0]), 0) + "\002"]

    def test_render_uses_base_dir_for_relative_path(self, tmp_path, monkeypatch):
        # Relative paths in tool output resolve against the tool call's
        # workdir when one was provided — not the CLI process cwd.
        self._tty_and_terminal(monkeypatch, protocol="iterm2")
        img = tmp_path / "out.png"
        img.write_bytes(b"x")
        monkeypatch.chdir(tmp_path.parent)
        calls = []
        assert display_module.render_image_preview("wrote out.png", base_dir=str(tmp_path), print_fn=calls.append) is True
        assert calls == ["  ┊ image", "\001" + display_module._iterm2_image_escape(img, 0) + "\002"]

    def test_no_result_or_empty_result_emits_nothing(self, monkeypatch):
        self._tty_and_terminal(monkeypatch)
        calls = []
        assert display_module.render_image_preview(None, print_fn=calls.append) is False
        assert display_module.render_image_preview("", print_fn=calls.append) is False
        assert calls == []

    def test_non_str_result_emits_nothing(self, monkeypatch):
        self._tty_and_terminal(monkeypatch)
        calls = []
        # Deliberately wrong-typed input — the gate is isinstance()-based.
        assert display_module.render_image_preview({"path": "/tmp/x.png"}, print_fn=calls.append) is False  # type: ignore[arg-type]
        assert calls == []

    def test_never_raises_when_escape_builder_fails(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="iterm2")
        img = _make_image(tmp_path)
        monkeypatch.setattr(display_module, "_iterm2_image_escape", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        calls = []
        assert display_module.render_image_preview(f"Saved {img}", print_fn=calls.append) is False
        assert calls == []

    def test_one_bad_image_does_not_kill_the_rest(self, tmp_path, monkeypatch):
        self._tty_and_terminal(monkeypatch, protocol="iterm2")
        good = _make_image(tmp_path, "good.png")
        bad = _make_image(tmp_path, "bad.png")
        real = display_module._iterm2_image_escape

        def flaky(path, max_width):
            if Path(path).name == "bad.png":
                raise RuntimeError("boom")
            return real(path, max_width)

        monkeypatch.setattr(display_module, "_iterm2_image_escape", flaky)
        calls = []
        assert display_module.render_image_preview(f"{bad} {good}", print_fn=calls.append) is True
        assert calls == ["  ┊ image", "\001" + real(Path(good), 0) + "\002"]


class TestImagePreviewConfig:
    def test_display_config_defaults(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        display = DEFAULT_CONFIG["display"]
        assert display["image_preview"] is True
        assert display["image_preview_max_width"] == 0
