"""Tests for gateway/platforms/base.py — MessageEvent, media extraction, message truncation."""

import os
import time
from unittest.mock import patch

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE,
    MessageEvent,
    SendResult,
    cache_audio_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
    safe_url_for_log,
    utf16_len,
    validate_inbound_media_size,
    _log_safe_path,
    _prefix_within_utf16_limit,
    cache_audio_from_bytes,
)


def test_media_delivery_denies_encrypted_bitwarden_cache(tmp_path, monkeypatch):
    """Encrypted Bitwarden cache is covered by the media credential guard."""
    import gateway.platforms.base as base

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(base, "_HERMES_HOME", hermes_home)
    monkeypatch.setattr(base, "_HERMES_ROOT", hermes_home)
    path = hermes_home / "cache" / "bws_cache.enc.json"
    path.parent.mkdir()
    path.write_text("encrypted-secret-cache")

    assert path in base._media_delivery_denied_paths()
    assert base.validate_media_delivery_path(str(path)) is None


class TestInboundMediaSizeCap:
    """gateway.max_inbound_media_bytes caps inbound media buffered into RAM (#13145)."""

    _PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64

    def test_default_cap_is_128_mib(self, monkeypatch):
        # No config override -> default. Patch loader to return empty config.
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: base.DEFAULT_INBOUND_MEDIA_MAX_BYTES)
        assert base.DEFAULT_INBOUND_MEDIA_MAX_BYTES == 128 * 1024 * 1024

    def test_image_bytes_rejected_when_oversized(self, monkeypatch):
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 16)
        with pytest.raises(ValueError, match="Inbound image payload is too large"):
            cache_image_from_bytes(self._PNG, ext=".png")


class TestSecretCaptureGuidance:
    def test_gateway_secret_capture_message_points_to_local_setup(self):
        message = GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
        assert "local cli" in message.lower()
        assert "~/.hermes/.env" in message


class TestSafeUrlForLog:
    def test_strips_query_fragment_and_userinfo(self):
        url = (
            "https://user:pass@example.com/private/path/image.png"
            "?X-Amz-Signature=supersecret&token=abc#frag"
        )
        result = safe_url_for_log(url)
        assert result == "https://example.com/.../image.png"
        assert "supersecret" not in result
        assert "token=abc" not in result
        assert "user:pass@" not in result


class TestCacheAudioFromBytes:
    def test_sniffs_mp4_quicktime_audio_even_when_ext_is_ogg(self, tmp_path):
        payload = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 32
        with patch("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path):
            result = cache_audio_from_bytes(payload, ext=".ogg")

        saved = tmp_path / os.path.basename(result)
        assert saved.suffix == ".m4a"
        assert saved.read_bytes() == payload


# ---------------------------------------------------------------------------
# MessageEvent — command parsing
# ---------------------------------------------------------------------------


class TestMessageEventIsCommand:
    def test_slash_command(self):
        event = MessageEvent(text="/new")
        assert event.is_command() is True


class TestMessageEventGetCommand:
    def test_simple_command(self):
        event = MessageEvent(text="/new")
        assert event.get_command() == "new"

    def test_command_with_args(self):
        event = MessageEvent(text="/reset session")
        assert event.get_command() == "reset"

    def test_not_a_command(self):
        event = MessageEvent(text="hello")
        assert event.get_command() is None


class TestMessageEventGetCommandArgs:
    def test_command_with_args(self):
        event = MessageEvent(text="/new session id 123")
        assert event.get_command_args() == "session id 123"


# ---------------------------------------------------------------------------
# extract_images
# ---------------------------------------------------------------------------


class TestExtractImages:
    def test_no_images(self):
        images, cleaned = BasePlatformAdapter.extract_images("Just regular text.")
        assert images == []
        assert cleaned == "Just regular text."

    def test_markdown_image_with_image_ext(self):
        content = "Here is a photo: ![cat](https://example.com/cat.png)"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/cat.png"
        assert images[0][1] == "cat"
        assert "![cat]" not in cleaned


    def test_fal_media_cdn(self):
        content = "![gen](https://fal.media/files/abc123/output.png)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://fal.media/files/abc123/output.png"
        assert images[0][1] == "gen"


    def test_replicate_delivery(self):
        content = "![](https://replicate.delivery/pbxt/abc/output)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://replicate.delivery/pbxt/abc/output"
        assert images[0][1] == ""


    def test_html_img_tag(self):
        content = 'Check this: <img src="https://example.com/photo.png">'
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/photo.png"
        assert images[0][1] == ""  # HTML images have no alt text
        assert "<img" not in cleaned


    def test_non_image_link_preserved_when_mixed_with_images(self):
        """Regression: non-image markdown links must not be silently removed
        when the response also contains real images."""
        content = (
            "Here is the image: ![photo](https://fal.media/cat.png)\n"
            "And a doc: ![report](https://example.com/report.pdf)"
        )
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://fal.media/cat.png"
        # The PDF link must survive in cleaned content
        assert "![report](https://example.com/report.pdf)" in cleaned


# ---------------------------------------------------------------------------
# extract_media
# ---------------------------------------------------------------------------


class TestExtractMedia:
    def test_no_media(self):
        media, cleaned = BasePlatformAdapter.extract_media("Just text.")
        assert media == []
        assert cleaned == "Just text."

    def test_single_media_tag(self):
        content = "MEDIA:/path/to/audio.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/path/to/audio.ogg"
        assert media[0][1] is False  # no voice tag


    def test_voice_directive_only_taints_audio_files(self):
        """[[audio_as_voice]] is message-global but must only flag audio files.

        A non-audio file marked is_voice is excluded from the embedded-photo
        batch and falls through to send_document, so an image sharing a
        message with a voice note used to arrive as a file attachment
        (#44826).
        """
        content = "[[audio_as_voice]]\nMEDIA:/tmp/pic.png\nMEDIA:/tmp/voice.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        flags = dict(media)
        assert flags["/tmp/pic.png"] is False
        assert flags["/tmp/voice.ogg"] is True
        assert "[[audio_as_voice]]" not in cleaned

    def test_voice_directive_skips_video_and_documents(self):
        content = "[[audio_as_voice]]\nMEDIA:/tmp/clip.mp4\nMEDIA:/tmp/report.pdf\nMEDIA:/tmp/note.opus"
        media, _ = BasePlatformAdapter.extract_media(content)
        flags = dict(media)
        assert flags["/tmp/clip.mp4"] is False
        assert flags["/tmp/report.pdf"] is False
        assert flags["/tmp/note.opus"] is True

    def test_multiple_media_tags(self):
        content = "MEDIA:/a.ogg\nMEDIA:/b.ogg"
        media, _ = BasePlatformAdapter.extract_media(content)
        assert len(media) == 2


    def test_cleaned_content_trims_excess_newlines(self):
        content = "Before\n\nMEDIA:/audio.ogg\n\n\n\nAfter"
        _, cleaned = BasePlatformAdapter.extract_media(content)
        assert "\n\n\n" not in cleaned


    def test_duplicate_media_tags_are_deduplicated(self):
        content = "MEDIA:/tmp/test.png\nMEDIA:/tmp/test.png\nMEDIA:/tmp/other.png"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [
            ("/tmp/test.png", False),
            ("/tmp/other.png", False),
        ]
        assert cleaned == ""


    def test_dedup_uses_expanded_path_so_tilde_and_absolute_collapse(self):
        import os
        home = os.path.expanduser("~")
        content = f"MEDIA:~/foo.png\nMEDIA:{home}/foo.png"
        media, _ = BasePlatformAdapter.extract_media(content)
        assert media == [(f"{home}/foo.png", False)]

    def test_as_document_directive_stripped_from_cleaned_text(self):
        """[[as_document]] is a routing directive — strip it from
        user-visible text just like [[audio_as_voice]]. Callers detect the
        directive on the original content (before extract_media)."""
        content = "Here is your infographic:\n[[as_document]]\nMEDIA:/tmp/x.jpg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/x.jpg", False)]
        assert "[[as_document]]" not in cleaned
        assert "Here is your infographic" in cleaned


    def test_both_directives_can_coexist(self):
        """A response could (rarely) contain both [[audio_as_voice]] for an
        ogg file AND [[as_document]] for an attached image. The voice flag
        propagates per-tuple; [[as_document]] is detected at dispatch."""
        content = "[[audio_as_voice]]\n[[as_document]]\nMEDIA:/tmp/x.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        # Voice flag is propagated to every media tuple (this matches the
        # existing extract_media contract)
        assert media == [("/tmp/x.ogg", True)]
        # Both directives stripped from cleaned text
        assert "[[audio_as_voice]]" not in cleaned
        assert "[[as_document]]" not in cleaned

    # Windows path support — regression coverage for #34632


    def test_relative_path_still_ignored(self):
        """Relative Windows-style paths (no drive letter) must not match."""
        media, _ = BasePlatformAdapter.extract_media(
            r"MEDIA:Users\kotsu\file.pdf"
        )
        assert media == []

    # --- Code block / inline code / blockquote false-positive guards (#35695) ---


    def test_media_mixed_code_and_prose(self):
        """Real MEDIA: in prose + example in code block: only prose extracted,
        and the code block survives verbatim in the delivered text."""
        content = (
            "Here is your file:\n"
            "MEDIA:/output/report.pdf\n"
            "Example usage:\n"
            "```text\nMEDIA:/example/path.pdf\n```\n"
            "Done."
        )
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/output/report.pdf"
        assert "Done." in cleaned
        # The real tag is stripped from the delivered text...
        assert "MEDIA:/output/report.pdf" not in cleaned
        # ...but the fenced code block (incl. its example MEDIA: line) must
        # survive verbatim — masking is a locator, not a text rewrite.
        assert "```text\nMEDIA:/example/path.pdf\n```" in cleaned

    def test_inline_code_survives_when_real_media_present(self):
        """When a real MEDIA: tag is delivered, an inline-code example in the
        same reply must not be blanked to whitespace."""
        content = "See MEDIA:/r/a.png and `MEDIA:/ex/b.png` inline"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert [p for p, _ in media] == ["/r/a.png"]
        assert "`MEDIA:/ex/b.png`" in cleaned

    # --- Markdown emphasis wrapping tolerance ---
    # Models routinely present a file as **MEDIA:/path** / *MEDIA:/path* /
    # _MEDIA:/path_. The old pattern only tolerated a single quote/backtick, so
    # the emphasis prevented the match and the file was silently never
    # delivered (the literal MEDIA: text leaked into the chat instead).


class TestMediaInsideSerializedJson:
    """Regression coverage for #34375 — MEDIA: embedded in serialized JSON
    string values (e.g. a stored previous reply inside a tool result) must not
    be re-delivered as a real attachment, while legitimate MEDIA: tags in prose,
    at line start, indented, or as quoted-path tags keep working.
    """


    def test_media_in_embedded_serialized_reply_not_extracted(self):
        """A serialized tool result that embeds a prior reply's MEDIA: tag."""
        content = (
            '{"content":"previous reply MEDIA:/Users/ex/.hermes/media/'
            'generated/stale.png and more text"}'
        )
        media, _ = BasePlatformAdapter.extract_media(content)
        assert media == [], f"embedded serialized reply leaked: {media}"

    # --- Legitimate tags must still extract (no regression vs line-start anchor) ---


    def test_quoted_path_media_still_extracted(self):
        """MEDIA:"..." quoted-path form (a real LLM output) is not JSON-masked."""
        media, _ = BasePlatformAdapter.extract_media(
            'MEDIA:"/path/with space/file.png"'
        )
        assert len(media) == 1 and media[0][0] == "/path/with space/file.png"

    def test_tts_two_line_still_extracted(self):
        media, _ = BasePlatformAdapter.extract_media(
            "[[audio_as_voice]]\nMEDIA:/tmp/v.ogg"
        )
        assert len(media) == 1 and media[0][0] == "/tmp/v.ogg"
        assert media[0][1] is True  # voice flag

    # --- cleaned-text invariants: real tags stripped, JSON data kept verbatim ---

    def test_json_embedded_media_kept_verbatim_in_cleaned_text(self):
        """A real tag is delivered+stripped; a JSON-embedded MEDIA: stays as
        literal text (stored data must read back unchanged)."""
        content = 'MEDIA:/real/r.png\nlog: {"old":"MEDIA:/stale/s.png"}'
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert [p for p, _ in media] == ["/real/r.png"]
        # The JSON-embedded path must survive verbatim — not blanked to spaces.
        assert '{"old":"MEDIA:/stale/s.png"}' in cleaned


class TestMediaExtensionAllowlistParity:
    """Regression coverage for issue #34517 — the MEDIA: extension black hole.

    extract_media used to carry a narrow extension allowlist that omitted
    .md/.json/.yaml/.xml/.html etc., while extract_local_files had a broad one.
    Combined with an unconditional ``MEDIA:\\s*\\S+`` strip at the dispatch
    sites, an unmatched MEDIA: tag for one of those extensions was deleted from
    the body before extract_local_files could pick up the bare path — the file
    was silently dropped. Both extractors now derive from the single
    MEDIA_DELIVERY_EXTS source of truth, and the strip is anchored to that set.
    """

    DROPPED_BEFORE = ["md", "json", "yaml", "yml", "xml", "html", "htm",
                      "tsv", "svg"]


    def test_unknown_extension_not_black_holed_by_cleanup(self):
        """A MEDIA: tag with an unknown extension is NOT stripped from the
        body by the extension-anchored cleanup — and when the path does not
        validate (nonexistent file here), it is not delivered either, so the
        tag survives visibly instead of vanishing (the core of issue #34517).
        Validated unknown-extension paths DO deliver — see
        TestUniversalMediaEgress (#36060)."""
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE
        text = "Saved to MEDIA:/tmp/data.weirdext done"
        media, _ = BasePlatformAdapter.extract_media(text)
        assert media == []  # nonexistent path fails validation, not delivered
        stripped = MEDIA_TAG_CLEANUP_RE.sub("", text)
        assert "/tmp/data.weirdext" in stripped  # path preserved, not dropped


class TestExtensionlessMediaDelivery:
    """Regression: MEDIA: tags for extension-less files (Caddyfile, Makefile)."""

    def _patch_allow_root(self, monkeypatch, root):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (str(root),),
        )
        monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)

    def test_extensionless_media_extracted_when_file_validates(self, tmp_path, monkeypatch):
        root = tmp_path / "output"
        root.mkdir()
        caddy = root / "Caddyfile"
        caddy.write_text("localhost {}", encoding="utf-8")
        self._patch_allow_root(monkeypatch, root)

        content = f"Here is your config:\nMEDIA:{caddy}\nDone."
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == str(caddy.resolve())
        assert "MEDIA:" not in cleaned
        assert "Done." in cleaned


class TestUniversalMediaEgress:
    """#36060: every MEDIA: path is deliverable regardless of file type.

    Known extensions extract unconditionally (MEDIA_TAG_CLEANUP_RE); unknown
    extensions and extension-less files extract via the validated pass —
    delivered when validate_media_delivery_path accepts them, left visible
    when it does not (nonexistent, denylisted).
    """

    def _patch_allow_root(self, monkeypatch, root):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (str(root),),
        )
        monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)

    @pytest.mark.parametrize("name", [
        "script.py", "server.log", "notes.weirdext", "app.ts", "run.sh",
        "config.toml", "styles.css", "contract.sol",
    ])
    def test_unknown_extension_delivered_when_file_validates(
        self, tmp_path, monkeypatch, name,
    ):
        root = tmp_path / "output"
        root.mkdir()
        f = root / name
        f.write_text("content", encoding="utf-8")
        self._patch_allow_root(monkeypatch, root)

        content = f"Here you go:\nMEDIA:{f}\nDone."
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == str(f.resolve())
        assert "MEDIA:" not in cleaned
        assert "Done." in cleaned


    def test_known_extension_still_unconditional(self):
        # Known extensions keep the pre-#36060 behavior: extracted (and the
        # tag stripped) even when the file does not exist — downstream
        # delivery surfaces the failure.
        content = "MEDIA:/nonexistent/report.pdf"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/nonexistent/report.pdf", False)]
        assert "MEDIA:" not in cleaned


class TestMediaDeliveryPathValidation:
    def _patch_roots(self, monkeypatch, *roots):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            tuple(roots),
        )
        # All tests in this class cover strict-mode behavior (allowlist +
        # recency window + denylist). Force strict on so they keep
        # exercising the legacy path even though the public default
        # flipped to off in 2026-05.
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
        # Disable recency-based trust by default so the original allowlist
        # tests continue to exercise the strict-allowlist path. Tests that
        # specifically cover recency trust re-enable it themselves.
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")


    def test_rejects_symlink_escape_from_safe_root(self, tmp_path, monkeypatch):
        root = tmp_path / "media-cache"
        root.mkdir()
        secret = tmp_path / "outside.png"
        secret.write_bytes(b"secret")
        link = root / "safe-looking.png"
        try:
            link.symlink_to(secret)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        self._patch_roots(monkeypatch, root)

        assert BasePlatformAdapter.validate_media_delivery_path(str(link)) is None

    def test_filter_keeps_safe_media_and_drops_unsafe(self, tmp_path, monkeypatch):
        root = tmp_path / "media-cache"
        safe = root / "speech.ogg"
        unsafe = tmp_path / "outside.ogg"
        safe.parent.mkdir(parents=True)
        safe.write_bytes(b"OggS")
        unsafe.write_bytes(b"OggS")
        self._patch_roots(monkeypatch, root)

        filtered = BasePlatformAdapter.filter_media_delivery_paths([
            (str(unsafe), False),
            (str(safe), True),
        ])

        assert filtered == [(str(safe.resolve()), True)]


    def test_allows_stale_kanban_attachment_but_not_neighboring_workspace(
        self, tmp_path, monkeypatch,
    ):
        """Strict mode trusts durable attachments without trusting scratch."""
        self._patch_roots(monkeypatch)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
        board_root = tmp_path / "hermes" / "kanban" / "boards" / "research"
        board_root.mkdir(parents=True)
        (board_root / "kanban.db").touch()
        attachment = board_root / "attachments" / "t_12345678" / "report.pdf"
        scratch = board_root / "workspaces" / "t_12345678" / "notes.txt"
        attachment.parent.mkdir(parents=True)
        scratch.parent.mkdir(parents=True)
        attachment.write_bytes(b"%PDF")
        scratch.write_text("private", encoding="utf-8")

        assert BasePlatformAdapter.validate_media_delivery_path(str(attachment)) == str(
            attachment.resolve()
        )
        assert BasePlatformAdapter.validate_media_delivery_path(str(scratch)) is None


    def test_recency_trust_denies_system_paths_even_when_fresh(self, tmp_path, monkeypatch):
        """A freshly-touched file under /etc must NOT be uploaded.

        Belt-and-braces: even if an attacker rewrites the file's mtime
        (e.g. via a separately compromised tool result that touches a system
        file), the denylist refuses to deliver paths under /etc, /proc, /sys,
        ~/.ssh, ~/.aws, etc.
        """
        self._patch_roots(monkeypatch)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        # Simulate $HOME so ~/.ssh resolves into our tmp dir.
        fake_home = tmp_path / "home"
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        secret = ssh_dir / "id_rsa.txt"
        secret.write_bytes(b"-----BEGIN ...")  # mtime = now
        monkeypatch.setenv("HOME", str(fake_home))

        assert BasePlatformAdapter.validate_media_delivery_path(str(secret)) is None


class TestMediaDeliveryDefaultMode:
    """Default (non-strict) mode — denylist gates delivery, nothing else.

    Symmetric with inbound delivery: Telegram/Discord/Slack accept any
    document type the user uploads, and the agent can hand back any file
    that isn't a credential. Strict mode is opt-in for operators running
    public-facing gateways.
    """

    def _patch_roots(self, monkeypatch, *roots):
        # Empty cache allowlist so the only positive path through
        # validate_media_delivery_path in these tests is the
        # default-mode "anything not denied" branch.
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            tuple(roots),
        )
        # Pin strict OFF — the public default. Tests that exercise the
        # strict path live in TestMediaDeliveryPathValidation.
        monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)

    def test_accepts_stale_file_outside_allowlist(self, tmp_path, monkeypatch):
        """The motivating case — agent says ``MEDIA:/home/user/notes.md``
        for an .md it has been working with for hours. Strict mode would
        reject this (outside allowlist, outside recency window). Default
        mode delivers it.
        """
        self._patch_roots(monkeypatch)

        notes = tmp_path / "notes.md"
        notes.write_text("# Old notes\n")
        old_mtime = time.time() - 7200  # 2 hours ago — far outside any window
        os.utime(notes, (old_mtime, old_mtime))

        assert BasePlatformAdapter.validate_media_delivery_path(str(notes)) == str(notes.resolve())


    @pytest.mark.parametrize(
        "rel",
        [
            "mcp-tokens/github.json",
            "mcp-tokens/github.client.json",
            "mcp-tokens/github.meta.json",
        ],
    )
    def test_denylist_blocks_mcp_oauth_tokens(self, tmp_path, monkeypatch, rel):
        """Live MCP OAuth tokens/client creds under ~/.hermes/mcp-tokens/ must
        never deliver as native media — same exfil class as auth.json/.env.
        Sibling to the pairing/ directory denylist entry.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        (hermes_dir / "mcp-tokens").mkdir(parents=True)
        secret = hermes_dir / rel
        secret.write_text('{"access_token": "live-bearer-abc123"}')
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_HOME",
            hermes_dir,
        )
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_ROOT",
            hermes_dir,
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(secret)) is None


    def test_denylist_blocks_google_token_default_mode(self, tmp_path, monkeypatch):
        """Integration credentials at the HERMES_HOME root (google_token.json)
        must never be deliverable, even though they aren't the historically
        enumerated .env/auth.json/config.yaml files. Regression for a
        refreshed google_token.json being auto-attached to a Slack reply
        (#50912).
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        token = hermes_dir / "google_token.json"
        token.write_text('{"access_token": "***", "refresh_token": "***"}')
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)
        monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(token)) is None


    def test_denylist_blocks_non_cache_file_under_hermes_home(self, tmp_path, monkeypatch):
        """A non-credential file the agent wrote directly under ~/.hermes
        (not in a cache subdir) is still deliverable via recency trust — we
        did NOT blanket-deny the tree (per #32090/#34425). This guards against
        accidentally re-introducing the rejected whole-tree deny.
        """
        self._patch_roots(monkeypatch)  # strict mode on
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        artifact = hermes_dir / "adhoc_report.pdf"
        artifact.write_bytes(b"%PDF-1.4")  # fresh mtime
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)
        monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(artifact)) == str(artifact.resolve())

    def test_strict_mode_envvar_restores_legacy_behavior(self, tmp_path, monkeypatch):
        """Setting HERMES_MEDIA_DELIVERY_STRICT=1 reactivates the older
        allowlist+recency logic. A stale file outside the allowlist is
        rejected.
        """
        self._patch_roots(monkeypatch)
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")

        stale = tmp_path / "old.pdf"
        stale.write_bytes(b"%PDF-1.4")
        old_mtime = time.time() - 7200
        os.utime(stale, (old_mtime, old_mtime))

        assert BasePlatformAdapter.validate_media_delivery_path(str(stale)) is None


    def test_root_home_deliverable_is_accepted(self, tmp_path, monkeypatch):
        """The motivating bug (#38106): a root-run gateway has ``$HOME=/root``,
        which is on the system-prefix denylist. A plain deliverable the agent
        produced in its working dir (``/root/work/proposal.docx``) must still
        deliver — the home itself is not a credential location.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "root"
        workdir = fake_home / "work"
        workdir.mkdir(parents=True)
        doc = workdir / "proposal.docx"
        doc.write_bytes(b"PK\x03\x04")
        monkeypatch.setenv("HOME", str(fake_home))
        # $HOME is itself on the denied-prefix list, mirroring /root.
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(fake_home),),
        )

        assert (
            BasePlatformAdapter.validate_media_delivery_path(str(doc))
            == str(doc.resolve())
        )


    def test_profile_scoped_cache_delivers_under_symlinked_root(self, tmp_path, monkeypatch):
        """Reopened #31733: a profile gateway whose HERMES_HOME is symlinked
        under a denied prefix (e.g. /opt/data -> /root/.hermes) emits
        profile-scoped paths (``<root>/profiles/<name>/cache/images/x.png``)
        that resolve under ``/root``. ``$HOME`` is NOT that prefix, so the
        root-home exception doesn't fire, and the top-level cache allowlist
        doesn't cover the profile subdir — the file was silently dropped.
        Per-profile cache roots must be allowlisted so it delivers.
        """
        self._patch_roots(monkeypatch)  # strict on, zero top-level cache roots

        # Stand-in for the literal /root deny prefix in the deployment.
        denied_root = tmp_path / "root"
        hermes_root = denied_root / ".hermes"
        prof_cache = hermes_root / "profiles" / "myprof" / "cache" / "images"
        prof_cache.mkdir(parents=True)
        image = prof_cache / "gen.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        # $HOME is NOT the denied prefix (mirrors HOME=/opt/data/home).
        fake_home = tmp_path / "opt" / "data" / "home"
        fake_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(denied_root),),
        )
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_ROOT", hermes_root
        )

        assert (
            BasePlatformAdapter.validate_media_delivery_path(str(image))
            == str(image.resolve())
        )


    def test_root_home_workdir_symlink_to_credential_blocked(self, tmp_path, monkeypatch):
        """A symlink in the workdir pointing at a credential is rejected on its
        resolved target, even under the $HOME exception.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "root"
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        key = ssh_dir / "id_rsa"
        key.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----")
        workdir = fake_home / "work"
        workdir.mkdir()
        link = workdir / "innocent.pdf"
        link.symlink_to(key)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(fake_home),),
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(link)) is None


class TestDockerContainerMediaPathTranslation:
    """MEDIA:/workspace (and configured mounts) must resolve to host paths."""

    def test_configured_workspace_mount_translates(self, tmp_path, monkeypatch):
        import json

        host_ws = tmp_path / "host-ws"
        host_ws.mkdir()
        media = host_ws / "shot.png"
        media.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setenv(
            "TERMINAL_DOCKER_VOLUMES",
            json.dumps([f"{host_ws}:/workspace"]),
        )
        monkeypatch.delenv("TERMINAL_ENV", raising=False)

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/shot.png"
        ) == str(media.resolve())

    def test_configured_output_mount_translates(self, tmp_path, monkeypatch):
        import json

        host_out = tmp_path / "documents"
        host_out.mkdir()
        media = host_out / "report.pdf"
        media.write_bytes(b"%PDF-1.4")
        monkeypatch.setenv(
            "TERMINAL_DOCKER_VOLUMES",
            json.dumps([f"{host_out}:/output"]),
        )

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/output/report.pdf"
        ) == str(media.resolve())

    def test_longest_prefix_wins(self, tmp_path, monkeypatch):
        import json

        host_a = tmp_path / "a"
        host_b = tmp_path / "b"
        host_a.mkdir()
        host_b.mkdir()
        nested = host_b / "file.png"
        nested.write_bytes(b"png")
        monkeypatch.setenv(
            "TERMINAL_DOCKER_VOLUMES",
            json.dumps([
                f"{host_a}:/data",
                f"{host_b}:/data/nested",
            ]),
        )

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/data/nested/file.png"
        ) == str(nested.resolve())

    def test_default_persistent_workspace_fallback(self, tmp_path, monkeypatch):
        sandbox = tmp_path / "sandboxes"
        ws = sandbox / "docker" / "default" / "workspace"
        ws.mkdir(parents=True)
        media = ws / "out.png"
        media.write_bytes(b"png")
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
        monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
        monkeypatch.delenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", raising=False)

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/out.png"
        ) == str(media.resolve())

    def test_unmapped_container_path_fails(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        assert BasePlatformAdapter.validate_media_delivery_path("/workspace/nope.png") is None

    def test_persistent_home_root_write_translates(self, tmp_path, monkeypatch):
        """An agent writing /root/out.png in a persistent container produced a
        real host file under <sandbox>/docker/default/home — deliver it."""
        sandbox = tmp_path / "sandboxes"
        home = sandbox / "docker" / "default" / "home"
        home.mkdir(parents=True)
        media = home / "out.png"
        media.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
        monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/root/out.png"
        ) == str(media.resolve())

    def test_cache_dir_container_path_translates_to_host_cache(self, tmp_path, monkeypatch):
        """MEDIA:/root/.hermes/cache/images/... (the agent_visible_image path
        under docker) must translate to the HOST cache file, not the sandbox
        home copy."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "cache" / "images"
        cache.mkdir(parents=True)
        media = cache / "generated.png"
        media.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/root/.hermes/cache/images/generated.png"
        ) == str(media.resolve())

    def test_container_credential_path_never_translates_through_home(self, tmp_path, monkeypatch):
        """/root/.hermes/* outside a cache mount (the sandbox's credential
        surface: .env, auth.json) must NOT resolve through the persistent
        home mount — those host-side copies sit outside the credential
        denylist prefixes and would otherwise deliver."""
        sandbox = tmp_path / "sandboxes"
        home = sandbox / "docker" / "default" / "home"
        secret = home / ".hermes"
        secret.mkdir(parents=True)
        (secret / "auth.json").write_text('{"token": "SECRET"}')
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
        monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/root/.hermes/auth.json"
        ) is None


# ---------------------------------------------------------------------------
# should_send_media_as_audio
# ---------------------------------------------------------------------------

class TestShouldSendMediaAsAudio:
    """Audio-routing policy shared by gateway + scheduler + send_message."""

    def test_unknown_extension_returns_false(self):
        from gateway.platforms.base import should_send_media_as_audio
        assert should_send_media_as_audio(None, ".png") is False
        assert should_send_media_as_audio("telegram", ".pdf") is False


    def test_telegram_ogg_opus_only_when_voice_flagged(self):
        from gateway.platforms.base import should_send_media_as_audio
        assert should_send_media_as_audio("telegram", ".ogg", is_voice=True) is True
        assert should_send_media_as_audio("telegram", ".opus", is_voice=True) is True
        assert should_send_media_as_audio("telegram", ".ogg") is False
        assert should_send_media_as_audio("telegram", ".opus") is False


# ---------------------------------------------------------------------------
# truncate_message
# ---------------------------------------------------------------------------


class TestIsSenderAuthorized:
    """``_is_sender_authorized`` is a tri-state: True / False / unknown.

    Callers gate credentialed side effects on an explicit ``is True``, so a
    truthy non-boolean must resolve to unknown rather than being coerced
    into an authorization by ``bool()``.
    """

    def _adapter(self):
        class StubAdapter(BasePlatformAdapter):
            async def connect(self, *, is_reconnect: bool = False):
                return True

            async def disconnect(self):
                pass

            async def send(self, *a, **kw):
                pass

            async def get_chat_info(self, *a):
                return {}

        from gateway.config import Platform, PlatformConfig

        return StubAdapter(config=PlatformConfig(enabled=True, token="test"),
                           platform=Platform.TELEGRAM)

    def test_no_check_registered_is_unknown(self):
        assert self._adapter()._is_sender_authorized("user") is None

    def test_empty_user_id_is_unknown(self):
        adapter = self._adapter()
        adapter.set_authorization_check(lambda *_a: True)
        assert adapter._is_sender_authorized("") is None

    def test_true_and_false_propagate(self):
        adapter = self._adapter()
        adapter.set_authorization_check(lambda *_a: True)
        assert adapter._is_sender_authorized("user") is True
        adapter.set_authorization_check(lambda *_a: False)
        assert adapter._is_sender_authorized("user") is False

    @pytest.mark.parametrize("result", ["allowed", 1, object(), [1]])
    def test_truthy_non_boolean_is_unknown(self, result):
        adapter = self._adapter()
        adapter.set_authorization_check(lambda *_a: result)
        assert adapter._is_sender_authorized("user") is None

    def test_raising_check_is_unknown(self):
        def boom(*_a):
            raise RuntimeError("auth backend down")

        adapter = self._adapter()
        adapter.set_authorization_check(boom)
        assert adapter._is_sender_authorized("user") is None

    def test_check_receives_chat_context(self):
        seen = []
        adapter = self._adapter()
        adapter.set_authorization_check(
            lambda user_id, chat_type, chat_id: seen.append((user_id, chat_type, chat_id)) or True
        )
        adapter._is_sender_authorized("user", "group", "chan")
        assert seen == [("user", "group", "chan")]


class TestTruncateMessage:
    def _adapter(self):
        """Create a minimal adapter instance for testing static/instance methods."""

        class StubAdapter(BasePlatformAdapter):
            async def connect(self, *, is_reconnect: bool = False):
                return True

            async def disconnect(self):
                pass

            async def send(self, *a, **kw):
                pass

            async def get_chat_info(self, *a):
                return {}

        from gateway.config import Platform, PlatformConfig

        config = PlatformConfig(enabled=True, token="test")
        return StubAdapter(config=config, platform=Platform.TELEGRAM)

    def test_short_message_single_chunk(self):
        adapter = self._adapter()
        chunks = adapter.truncate_message("Hello world", max_length=100)
        assert chunks == ["Hello world"]

    def test_exact_length_single_chunk(self):
        adapter = self._adapter()
        msg = "x" * 100
        chunks = adapter.truncate_message(msg, max_length=100)
        assert chunks == [msg]


    @staticmethod
    def _truncate_with_timeout(content, max_length, *, len_fn=None, timeout=3.0):
        """Run truncate_message on a worker thread; fail if it doesn't return.

        Guards against the regression where a pathologically small max_length
        made the split loop never consume any input and spin forever.
        """
        import threading

        box: dict = {}

        def _run():
            box["result"] = BasePlatformAdapter.truncate_message(
                content, max_length, len_fn=len_fn
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)
        assert not t.is_alive(), (
            f"truncate_message hung (infinite loop) for max_length={max_length}"
        )
        return box["result"]

    def test_pathological_small_max_length_terminates(self):
        # max_length 0 and 1 previously drove the split loop into an unbounded
        # hang (headroom -> 0, split_at -> 0, remaining never shrinks). It must
        # terminate and preserve every character across the chunks.
        import re

        for max_length in (0, 1, 2):
            chunks = self._truncate_with_timeout("abcdefghij", max_length)
            assert chunks, f"no chunks for max_length={max_length}"
            reassembled = "".join(
                re.sub(r"\s*\(\d+/\d+\)$", "", c) for c in chunks
            )
            for ch in "abcdefghij":
                assert ch in reassembled, f"char {ch!r} lost at max_length={max_length}"


    def test_code_block_language_tag_carried(self):
        adapter = self._adapter()
        msg = "Start\n```javascript\n" + "console.log('x');\n" * 80 + "```\nEnd"
        chunks = adapter.truncate_message(msg, max_length=300)
        if len(chunks) > 1:
            # At least one continuation chunk should reopen with ```javascript
            reopened_with_lang = any("```javascript" in chunk for chunk in chunks[1:])
            assert reopened_with_lang, (
                "No continuation chunk reopened with language tag"
            )


# ---------------------------------------------------------------------------
# _get_human_delay
# ---------------------------------------------------------------------------


class TestGetHumanDelay:


    def test_natural_mode_ignores_malformed_custom_env_vars(self):
        env = {
            "HERMES_HUMAN_DELAY_MODE": "natural",
            "HERMES_HUMAN_DELAY_MIN_MS": "oops",
            "HERMES_HUMAN_DELAY_MAX_MS": "still-bad",
        }
        with patch.dict(os.environ, env):
            delay = BasePlatformAdapter._get_human_delay()
            assert 0.8 <= delay <= 2.5


    def test_custom_mode_tolerates_malformed_env_vars(self):
        env = {
            "HERMES_HUMAN_DELAY_MODE": "custom",
            "HERMES_HUMAN_DELAY_MIN_MS": "oops",
            "HERMES_HUMAN_DELAY_MAX_MS": "still-bad",
        }
        with patch.dict(os.environ, env):
            # falls back to the custom-mode defaults instead of crashing
            delay = BasePlatformAdapter._get_human_delay()
            assert 0.8 <= delay <= 2.5


# ---------------------------------------------------------------------------
# utf16_len / _prefix_within_utf16_limit / truncate_message with len_fn
# ---------------------------------------------------------------------------
# Ported from nearai/ironclaw#2304 — Telegram counts message length in UTF-16
# code units, not Unicode code-points.  Astral-plane characters (emoji, CJK
# Extension B) are surrogate pairs: 1 Python char but 2 UTF-16 units.


class TestUtf16Len:
    """Verify the UTF-16 length helper."""

    def test_ascii(self):
        assert utf16_len("hello") == 5

    def test_bmp_cjk(self):
        # CJK ideographs in the BMP are 1 code unit each
        assert utf16_len("你好") == 2


class TestPrefixWithinUtf16Limit:
    """Verify UTF-16-aware prefix truncation."""

    def test_fits_entirely(self):
        assert _prefix_within_utf16_limit("hello", 10) == "hello"


    def test_all_emoji(self):
        msg = "😀" * 10  # 20 UTF-16 units
        result = _prefix_within_utf16_limit(msg, 6)
        assert result == "😀😀😀"
        assert utf16_len(result) == 6


class TestTruncateMessageUtf16:
    """Verify truncate_message respects UTF-16 lengths when len_fn=utf16_len."""

    def test_short_emoji_message_no_split(self):
        """A short message under the UTF-16 limit should not be split."""
        msg = "Hello 😀 world"
        chunks = BasePlatformAdapter.truncate_message(msg, 4096, len_fn=utf16_len)
        assert len(chunks) == 1
        assert chunks[0] == msg

    def test_emoji_near_limit_triggers_split(self):
        """A message at 4096 codepoints but >4096 UTF-16 units must split."""
        # 2049 emoji = 2049 codepoints but 4098 UTF-16 units → exceeds 4096
        msg = "😀" * 2049
        assert len(msg) == 2049  # Python len sees 2049 chars
        assert utf16_len(msg) == 4098  # but it's 4098 UTF-16 units

        # Without UTF-16 awareness, this would NOT split (2049 < 4096)
        chunks_naive = BasePlatformAdapter.truncate_message(msg, 4096)
        assert len(chunks_naive) == 1, "Without len_fn, no split expected"

        # With UTF-16 awareness, it MUST split
        chunks = BasePlatformAdapter.truncate_message(msg, 4096, len_fn=utf16_len)
        assert len(chunks) > 1, "With utf16_len, message should be split"

        # Each chunk must fit within the UTF-16 limit
        for i, chunk in enumerate(chunks):
            assert utf16_len(chunk) <= 4096, (
                f"Chunk {i} exceeds 4096 UTF-16 units: {utf16_len(chunk)}"
            )


class TestProxyKwargsForAiohttp:
    """Verify proxy_kwargs_for_aiohttp routes all schemes through ProxyConnector."""


    def test_http_proxy_uses_connector_when_aiohttp_socks_available(self):
        pytest.importorskip("aiohttp_socks")
        from unittest.mock import MagicMock
        from gateway.platforms.base import proxy_kwargs_for_aiohttp

        sentinel = MagicMock(name="ProxyConnector")
        with patch("aiohttp_socks.ProxyConnector.from_url", return_value=sentinel):
            sess_kw, req_kw = proxy_kwargs_for_aiohttp("http://proxy:8080")
        assert sess_kw.get("connector") is sentinel, (
            "HTTP proxy must use ProxyConnector so libraries that don't "
            "forward per-request proxy= kwargs still route through the proxy"
        )
        assert req_kw == {}


class TestMediaDeliveryDiagnosability:
    """Diagnosable rejection logging + crafted-path robustness (#33251)."""

    def test_rejected_path_appears_in_log(self, tmp_path, caplog):
        outside = tmp_path / "outside.ogg"
        outside.write_bytes(b"OggS")
        with patch.dict(os.environ, {"HERMES_MEDIA_DELIVERY_STRICT": "1",
                                     "HERMES_MEDIA_TRUST_RECENT_FILES": "0"}), \
                patch("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", ()):
            with caplog.at_level("WARNING"):
                out = BasePlatformAdapter.filter_media_delivery_paths([(str(outside), False)])
        assert out == []
        # The dropped path must be in the log so operators can diagnose it.
        assert str(outside) in caplog.text

    def test_crafted_null_path_does_not_abort_batch(self, tmp_path, monkeypatch):
        """One crafted ~\\x00 path must not drop every other attachment."""
        good = tmp_path / "good.png"
        good.write_bytes(b"\x89PNG")
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")
        out = BasePlatformAdapter.filter_media_delivery_paths([
            ("~\x00evil.png", False),
            (str(good), False),
        ])
        assert out == [(str(good.resolve()), False)]


# ---------------------------------------------------------------------------
# Media-send fallback must not leak host filesystem paths into chat
# ---------------------------------------------------------------------------


class _CapturingAdapter(BasePlatformAdapter):
    """Minimal concrete BasePlatformAdapter that records what send() sees.

    The four media-send fallbacks (send_voice, send_video, send_document,
    send_image_file) historically forwarded their *_path argument into the
    chat text. That argument is a host filesystem path inside the Hermes
    cache, so any subclass that fell back to super() — like the Telegram
    adapter on a rejected video — would leak the host's directory layout
    into the user's chat.
    """

    def __init__(self):
        from gateway.config import Platform, PlatformConfig
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent: list[dict] = []

    async def connect(self) -> bool:  # pragma: no cover - not exercised
        return True

    async def disconnect(self) -> None:  # pragma: no cover - not exercised
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - not exercised
        return {"name": chat_id, "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id="m1")


class TestMediaFallbackDoesNotLeakHostPath:
    """Regression: the four base-class media fallbacks must not echo *_path.

    Telegram, Discord, and Slack adapters all fall back to these base
    implementations on native-send failure. When they did, the user saw
    a chat message like ``🎬 Video: /home/.../hermes/cache/video/abc.mp4``
    — a host filesystem path with no actionable information.
    """

    SENSITIVE_PATH = "/home/jayne/.hermes/cache/media/sensitive_host_path_abc123.bin"


    @pytest.mark.asyncio
    async def test_send_document_fallback_includes_explicit_filename_only(self):
        """A caller-supplied file_name is user-facing and may be shown — but
        the host file_path argument must still be suppressed."""
        adapter = _CapturingAdapter()
        result = await adapter.send_document(
            chat_id="123",
            file_path=self.SENSITIVE_PATH,
            file_name="report.pdf",
        )
        assert result.success
        sent_text = adapter.sent[0]["content"]
        assert self.SENSITIVE_PATH not in sent_text
        assert "/home/" not in sent_text
        assert "report.pdf" in sent_text


    @pytest.mark.asyncio
    async def test_caption_is_preserved_in_fallback(self):
        """The user-supplied caption is still shown — only the path is suppressed."""
        adapter = _CapturingAdapter()
        await adapter.send_video(
            chat_id="123",
            video_path=self.SENSITIVE_PATH,
            caption="Here's the daily summary.",
        )
        sent_text = adapter.sent[0]["content"]
        assert "Here's the daily summary." in sent_text
        assert self.SENSITIVE_PATH not in sent_text


class TestDockerProfileSandboxMediaTranslation:
    """MEDIA from persistent Docker sandboxes must resolve to the host
    directory the profile's container actually bind-mounts (#93950).

    Contract: persistent Docker is PROFILE-scoped — the default profile (and
    CLI) uses the literal ``default`` sandbox, other profiles use
    ``sanitize_task_id_for_path("profile:<name>")``. Legacy per-session
    sandboxes (``session:<key>``) created during the a270c4ade bug window
    remain resolvable as a fallback so their files still deliver.
    """

    SESSION_KEY = "agent:main:telegram:dm:123456"

    @staticmethod
    def _sandbox_dir(task_id: str = "default"):
        from tools.environments.base import get_sandbox_dir, sanitize_task_id_for_path

        name = task_id if task_id == "default" else sanitize_task_id_for_path(task_id)
        return get_sandbox_dir() / "docker" / name

    def _enable_docker(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")

    def test_default_profile_workspace_media_translates(self, monkeypatch):
        """A MEDIA tag pointing at the container's /workspace resolves to the
        default profile's shared host sandbox — with or without a session
        key, since every session of the profile shares one container."""
        self._enable_docker(monkeypatch)
        workspace = self._sandbox_dir() / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        produced = workspace / "chart.png"
        produced.write_bytes(b"png")

        with_key = BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/chart.png", session_key=self.SESSION_KEY
        )
        without_key = BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/chart.png"
        )

        assert with_key == without_key == str(produced.resolve())

    def test_legacy_session_sandbox_still_resolves(self, monkeypatch):
        """Self-heal for the a270c4ade bug window: files produced in a legacy
        per-session sandbox still deliver via the fallback candidate."""
        self._enable_docker(monkeypatch)
        workspace = self._sandbox_dir(f"session:{self.SESSION_KEY}") / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        produced = workspace / "out.png"
        produced.write_bytes(b"png")

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/out.png", session_key=self.SESSION_KEY
        ) == str(produced.resolve())

    def test_profile_and_legacy_sandboxes_both_searched(self, monkeypatch):
        """When the profile sandbox exists but the file was produced in a
        legacy per-session container, translation still finds it — the dir
        existing must not mask the fallback (#93950 follow-up)."""
        self._enable_docker(monkeypatch)
        (self._sandbox_dir() / "workspace").mkdir(parents=True, exist_ok=True)
        legacy_ws = self._sandbox_dir(f"session:{self.SESSION_KEY}") / "workspace"
        legacy_ws.mkdir(parents=True, exist_ok=True)
        produced = legacy_ws / "old.png"
        produced.write_bytes(b"png")

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/old.png", session_key=self.SESSION_KEY
        ) == str(produced.resolve())

    def test_home_mount_translates_stray_root_writes(self, monkeypatch):
        """/root/<file> lands in the profile sandbox's home mount."""
        self._enable_docker(monkeypatch)
        home = self._sandbox_dir() / "home"
        home.mkdir(parents=True, exist_ok=True)
        produced = home / "note.txt"
        produced.write_text("hi")

        assert BasePlatformAdapter.validate_media_delivery_path(
            "/root/note.txt", session_key=self.SESSION_KEY
        ) == str(produced.resolve())

    def test_home_credential_surface_still_refused(self, monkeypatch):
        """The /root/.hermes exclusion survives profile scoping: translating
        the home mount must never expose the container's secret surface —
        in the profile layout AND the legacy session layout."""
        self._enable_docker(monkeypatch)
        for task in ("default", f"session:{self.SESSION_KEY}"):
            secrets = self._sandbox_dir(task) / "home" / ".hermes"
            secrets.mkdir(parents=True, exist_ok=True)
            (secrets / "auth.json").write_text("{}")

        assert (
            BasePlatformAdapter.validate_media_delivery_path(
                "/root/.hermes/auth.json", session_key=self.SESSION_KEY
            )
            is None
        )

    def test_filter_passes_session_key_through(self, monkeypatch):
        """The adapter filter used by _process_message_background forwards the
        key, so legacy-sandbox MEDIA tags survive filtering in one hop."""
        self._enable_docker(monkeypatch)
        workspace = self._sandbox_dir(f"session:{self.SESSION_KEY}") / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        produced = workspace / "clip.mp4"
        produced.write_bytes(b"mp4")

        kept = BasePlatformAdapter.filter_media_delivery_paths(
            [("/workspace/clip.mp4", False)], session_key=self.SESSION_KEY
        )

        assert kept == [(str(produced.resolve()), False)]

    def test_unresolved_docker_media_names_the_cause(self, monkeypatch, caplog):
        """A container path that resolves in no candidate sandbox must log
        WHY it was dropped (sandbox mismatch), not just the generic unsafe-
        path line — the silent-drop mode reported in #93950."""
        import logging

        self._enable_docker(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="gateway.platforms.base"):
            resolved = BasePlatformAdapter.validate_media_delivery_path(
                "/workspace/ghost.png", session_key=self.SESSION_KEY
            )

        assert resolved is None
        assert any(
            "did not resolve" in r.message and f"session_key={self.SESSION_KEY}" in r.message
            for r in caplog.records
        )


class _LockGovernanceProbeAdapter(BasePlatformAdapter):
    """Minimal concrete adapter for platform-lock takeover governance tests."""

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *args, **kwargs) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": "probe", "type": "dm"}


class TestPlatformLockTakeoverGovernance:
    """A supervised (non-``--replace``) gateway must never evict a live holder.

    Regression for #79048: launchd services with ``KeepAlive=true`` used to be
    generated with ``--replace``, re-arming takeover authority on every
    respawn. Two profiles sharing one platform token (e.g. the same Discord
    bot) would then terminate each other in an endless mutual-eviction loop.
    The runtime guard is the adapter's ``_platform_lock_takeover_allowed``
    bit — only an explicit ``gateway run --replace`` startup arms it. Without
    that authority a live cross-home holder must be left alone and reported
    as a retryable failure, never terminated.
    """

    def _make_adapter(self, *, takeover_allowed: bool):
        from gateway.config import Platform, PlatformConfig

        adapter = _LockGovernanceProbeAdapter(
            config=PlatformConfig(), platform=Platform.DISCORD
        )
        adapter._platform_lock_takeover_allowed = takeover_allowed
        return adapter

    def test_no_takeover_without_replace_authority(self, tmp_path, monkeypatch):
        from gateway import status

        existing = {
            "pid": 4242,
            "start_time": 123,
            "home": str(tmp_path / "other-profile-home"),
        }
        monkeypatch.setattr(
            status,
            "acquire_scoped_lock",
            lambda scope, identity, metadata=None: (False, existing),
        )
        takeover_calls = []
        monkeypatch.setattr(
            status,
            "take_over_scoped_lock_holder",
            lambda existing: takeover_calls.append(existing) or 4242,
        )
        monkeypatch.setattr(status, "write_runtime_status", lambda **kw: None)

        adapter = self._make_adapter(takeover_allowed=False)
        # Ordinary (supervised) start: the live cross-home holder must be
        # left untouched — the gateway fails retryably instead of killing it.
        assert adapter._acquire_platform_lock("discord-token", "tok", "Discord") is False
        assert takeover_calls == []
        assert adapter.fatal_error_retryable is True
        assert adapter._fatal_error_code == "discord-token_lock"

    def test_takeover_authority_consumed_once(self, tmp_path, monkeypatch):
        from gateway import status

        existing = {
            "pid": 4242,
            "start_time": 123,
            "home": str(tmp_path / "other-profile-home"),
        }
        monkeypatch.setattr(
            status,
            "acquire_scoped_lock",
            lambda scope, identity, metadata=None: (False, existing),
        )
        takeover_calls = []
        monkeypatch.setattr(
            status,
            "take_over_scoped_lock_holder",
            lambda existing: takeover_calls.append(existing) or 4242,
        )
        monkeypatch.setattr(status, "write_runtime_status", lambda **kw: None)

        adapter = self._make_adapter(takeover_allowed=True)
        # An explicit --replace start may attempt exactly one takeover...
        assert adapter._acquire_platform_lock("discord-token", "tok", "Discord") is False
        assert len(takeover_calls) == 1
        # ...but the authority is consumed, so a reconnect can never evict a
        # healthy holder (this is what stops the supervised respawn loop).
        assert adapter._acquire_platform_lock("discord-token", "tok", "Discord") is False
        assert len(takeover_calls) == 1
        assert adapter._platform_lock_takeover_attempted is True


class TestDockerExplicitProfileOverridesAmbientLookup:
    """An explicitly-passed ``profile`` must win over the ambient
    ``get_active_profile_name()`` lookup (#64889, #94441).

    Delivery code (post-stream MEDIA rescan, cron result delivery) commonly
    runs after the turn's ``_profile_runtime_scope`` has already exited, so
    the ambient active-profile lookup can silently reflect a DIFFERENT
    profile (or the default) than the one that actually produced the file.
    Passing the delivering source's own profile explicitly must resolve the
    correct sandbox regardless of what the ambient lookup would say.
    """

    @staticmethod
    def _sandbox_dir(task_id: str = "default"):
        from tools.environments.base import get_sandbox_dir, sanitize_task_id_for_path

        name = task_id if task_id == "default" else sanitize_task_id_for_path(task_id)
        return get_sandbox_dir() / "docker" / name

    def _enable_docker(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")

    def test_explicit_profile_wins_when_ambient_disagrees(self, monkeypatch):
        """Ambient lookup says one profile; the explicit param names another.
        The explicit one must be the one that resolves."""
        self._enable_docker(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "stale_profile"
        )
        workspace = self._sandbox_dir("profile:real_profile") / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        produced = workspace / "chart.png"
        produced.write_bytes(b"png")
        # The ambient-named profile's sandbox does NOT exist at all — if the
        # explicit param were ignored, translation would fall through to
        # "default" (also absent here) and return None.
        assert not (self._sandbox_dir("profile:stale_profile")).exists()

        result = BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/chart.png", profile="real_profile"
        )

        assert result == str(produced.resolve())

    def test_omitted_profile_still_falls_back_to_ambient(self, monkeypatch):
        """Backward compatibility: no explicit profile behaves exactly as
        before — ambient lookup decides."""
        self._enable_docker(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "ambient_profile"
        )
        workspace = self._sandbox_dir("profile:ambient_profile") / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        produced = workspace / "chart.png"
        produced.write_bytes(b"png")

        result = BasePlatformAdapter.validate_media_delivery_path("/workspace/chart.png")

        assert result == str(produced.resolve())

    def test_explicit_profile_and_session_key_compose(self, monkeypatch):
        """Both new-style profile scoping and the legacy session_key fallback
        can be passed together — profile still wins when its sandbox exists,
        session_key remains available as a fallback candidate."""
        self._enable_docker(monkeypatch)
        legacy_ws = self._sandbox_dir("session:legacy-key") / "workspace"
        legacy_ws.mkdir(parents=True, exist_ok=True)
        legacy_file = legacy_ws / "old.png"
        legacy_file.write_bytes(b"png")

        # Only the legacy sandbox exists — the named profile's does not — so
        # passing both must still fall through to the legacy candidate.
        result = BasePlatformAdapter.validate_media_delivery_path(
            "/workspace/old.png", session_key="legacy-key", profile="no_such_profile"
        )

        assert result == str(legacy_file.resolve())
