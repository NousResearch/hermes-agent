"""Buzz CLI sha256 pin (item 8)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.gateway.buzz_forward_support import _buzz_mod, _make_adapter

pytest_plugins = ["tests.gateway.buzz_forward_support"]


class TestCliSha256Pin:
    def test_path_search_skipped_when_pin_set(self, tmp_path, monkeypatch):
        fake_on_path = tmp_path / "buzz"
        fake_on_path.write_text("path", encoding="utf-8")
        fake_on_path.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("BUZZ_CLI_SHA256", "abc")
        adapter = _make_adapter()
        assert adapter.cli_path == ""

    @pytest.mark.asyncio
    async def test_mismatch_refuses_start(self, tmp_path):
        binary = tmp_path / "buzz"
        binary.write_text("hello", encoding="utf-8")
        adapter = _make_adapter({"cli_path": str(binary), "cli_sha256": "0" * 64})
        ok = await adapter.connect()
        assert ok is False
        assert adapter._fatal_error_code == "config_invalid"

    def test_matching_pin_resolves_explicit_path(self, tmp_path):
        binary = tmp_path / "buzz"
        binary.write_text("hello", encoding="utf-8")
        digest = hashlib.sha256(b"hello").hexdigest()
        adapter = _make_adapter({"cli_path": str(binary), "cli_sha256": digest})
        assert adapter.cli_path == str(binary)
        assert _buzz_mod._sha256_file(Path(adapter.cli_path)) == digest

    @pytest.mark.asyncio
    async def test_rehash_before_exec_refuses_replaced_file(self, tmp_path):
        binary = tmp_path / "real-buzz"
        binary.write_text("hello", encoding="utf-8")
        link = tmp_path / "buzz"
        link.symlink_to(binary)
        digest = hashlib.sha256(b"hello").hexdigest()
        adapter = _make_adapter({"cli_path": str(link), "cli_sha256": digest})
        resolved = _buzz_mod._pinned_cli_exec_path(adapter.cli_path, adapter._extra)
        assert Path(resolved).resolve() == binary.resolve()
        binary.write_text("tampered", encoding="utf-8")
        code, _out, err = await adapter._run_cli(["users", "get"])
        assert code != 0
        assert "sha256" in err.lower()
