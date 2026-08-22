"""Seam-identity tests for the SlackFileDownloadMixin extraction
(adapter.py god-file slice R5-C5, epic #78647, target #78638).

Verifies the mixin-first class line keeps every C5 file-download member
bound through the MRO with zero test edits, plus the SSRF-guard behavior
smoke for the two download entry points.
"""

import ast
import asyncio
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.platforms.slack.adapter as adapter_mod
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.slack_file_download_mixin import (
    SlackFileDownloadMixin,
)

C5_MEMBERS = [
    "_is_slack_cdn_url",
    "_resolve_download_token",
    "_download_slack_file",
    "_download_slack_file_bytes",
]


def _fake_adapter():
    """Adapter-shaped object: only the instance attrs the mixin touches."""
    self = SlackAdapter.__new__(SlackAdapter)
    self.config = SimpleNamespace(token="«redacted:xox…»")
    self._team_clients = {}
    return self


class TestSeamIdentity:
    """The class line must resolve every C5 member to the mixin (MRO)."""

    def test_adapter_subclasses_mixin(self):
        assert issubclass(SlackAdapter, SlackFileDownloadMixin)

    def test_all_4_members_identity_bound(self):
        # _is_slack_cdn_url is a @classmethod: getattr() on a class returns a
        # freshly-created bound-method object per access, so `is` must
        # compare the underlying __func__ (which IS the mixin's function).
        for name in C5_MEMBERS:
            adapter_attr = getattr(SlackAdapter, name)
            mixin_attr = getattr(SlackFileDownloadMixin, name)
            if name == "_is_slack_cdn_url":
                assert adapter_attr.__func__ is mixin_attr.__func__, name
            else:
                assert adapter_attr is mixin_attr, name

    def test_mro_mixin_first(self):
        assert SlackAdapter.__mro__[1] is SlackFileDownloadMixin

    def test_mixin_has_no_module_level_adapter_import(self):
        # Circular-import guard: the mixin must not import adapter at module
        # level (no lazy imports are needed -- the moved methods import
        # httpx / gateway helpers / urllib inside the functions).
        path = pathlib.Path(adapter_mod.__file__).with_name(
            "slack_file_download_mixin.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "slack.adapter" in (
                node.module or ""
            ):
                assert node.col_offset > 0, (
                    f"module-level adapter import at {node.lineno}"
                )
        assert True


class TestResolveDownloadToken:
    """Token selection: explicit team -> URL-embedded team -> primary."""

    def test_explicit_team_id_wins(self):
        self = _fake_adapter()
        self._team_clients = {"T111": SimpleNamespace(token="tok-1")}
        assert self._resolve_download_token("https://x", "T111") == "tok-1"

    def test_url_embedded_team_id(self):
        self = _fake_adapter()
        self._team_clients = {"T222": SimpleNamespace(token="tok-2")}
        assert (
            self._resolve_download_token(
                "https://files.slack.com/files-pri/T222-F99/name.jpg"
            )
            == "tok-2"
        )

    def test_falls_back_to_primary_token(self):
        self = _fake_adapter()
        assert self._resolve_download_token("https://files.slack.com/x") == (
            "«redacted:xox…»"
        )


class TestIsSlackCdnUrl:
    """CDN allowlist classification (classmethod, token-exfiltration gate)."""

    def test_accepts_slack_cdn_hosts(self):
        assert SlackAdapter._is_slack_cdn_url(
            "https://files.slack.com/files-pri/T111-F99/x.jpg"
        )
        assert SlackAdapter._is_slack_cdn_url(
            "https://files.slack-files.com/x.jpg"
        )
        assert SlackAdapter._is_slack_cdn_url("https://slack.com/x")

    def test_rejects_foreign_public_host(self):
        assert not SlackAdapter._is_slack_cdn_url(
            "https://evil.example.com/steal"
        )

    def test_rejects_non_https_and_garbage(self):
        assert not SlackAdapter._is_slack_cdn_url(
            "http://files.slack.com/x.jpg"
        )
        assert not SlackAdapter._is_slack_cdn_url("not-a-url")
        assert not SlackAdapter._is_slack_cdn_url("")


class TestDownloadSsrfBehavior:
    """SSRF-guard smoke through the mixin's download entry points."""

    def test_unsafe_url_raises_value_error_before_network(self, monkeypatch):
        import tools.url_safety as url_safety

        calls = {"checked": []}

        def fake_is_safe_url(url, *a, **k):
            calls["checked"].append(url)
            return False

        monkeypatch.setattr(url_safety, "is_safe_url", fake_is_safe_url)

        self = _fake_adapter()
        with pytest.raises(ValueError):
            asyncio.run(self._download_slack_file("http://169.254.169.254/", ".jpg"))
        with pytest.raises(ValueError):
            asyncio.run(
                self._download_slack_file_bytes("http://169.254.169.254/")
            )
        assert len(calls["checked"]) == 2, (
            "both download methods must call is_safe_url before fetching"
        )


class TestImportabilityWithoutSlackSdk:
    """The mixin module must import with the slack SDK blocked."""

    def test_mixin_imports_without_slack_sdk(self, monkeypatch):
        for modname in list(sys.modules):
            if modname == "slack_sdk" or modname.startswith("slack_sdk."):
                monkeypatch.setitem(sys.modules, modname, None)
            if modname == "slack_bolt" or modname.startswith("slack_bolt."):
                monkeypatch.setitem(sys.modules, modname, None)
        import importlib

        module = importlib.import_module(
            "plugins.platforms.slack.slack_file_download_mixin"
        )
        assert module.SlackFileDownloadMixin is SlackFileDownloadMixin


def test_golden_window_sha_unchanged_at_pin():
    """The extracted window bytes still match the golden receipt.

    The golden sha was captured at pin aaf9688519 (window 8034-8229 of
    plugins/platforms/slack/adapter.py). The window now lives verbatim in
    slack_file_download_mixin.py (lines 43-238, byte-identical), so the
    receipt is verified against the SHIPPED module instead of the git
    object -- this also works in CI's shallow checkout, where the parent
    commit object may be absent.
    """
    import hashlib

    module_text = pathlib.Path(
        __file__
    ).resolve().parent.parent.parent.parent / "plugins" / "platforms" / "slack" / "slack_file_download_mixin.py"
    lines = module_text.read_text(encoding="utf-8").split("\n")
    # 196-line window: 0-indexed lines 43..238 (1-indexed 44..239) + trailing newline
    window = "\n".join(lines[43:239]) + "\n"
    assert hashlib.sha256(window.encode("utf-8")).hexdigest() == (
        "2a7edfc04118cf5c3300f5d25ae70e029f3c02ce7a9709263e2ce78cf6ee6e34"
    )
