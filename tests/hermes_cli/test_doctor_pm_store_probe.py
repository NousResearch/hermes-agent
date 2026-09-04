"""Doctor reports the tools Hermes would actually run — the pm store first.

Hermes runs pinned tools out of the pm store. Nothing puts the store on an
interactive shell's PATH, so a PATH-only probe reports a perfectly healthy
managed install as "not found", and silently skips the checks that depend
on the tool being present.
"""
from __future__ import annotations

from hermes_cli import doctor


class TestPmToolPath:
    """_pm_tool_path answers from facts.json + the pm store, nothing else."""

    def _store(self, tmp_path, monkeypatch, facts: dict):
        import json

        store = tmp_path / "tools"
        store.mkdir(parents=True, exist_ok=True)
        (store / "facts.json").write_text(
            json.dumps({"schema": 1, "packages": facts}), encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
        return store

    def test_staged_tool_resolves_to_its_store_binary(self, tmp_path, monkeypatch):
        import sys

        store = self._store(
            tmp_path, monkeypatch,
            {"ripgrep": {"entry": "ripgrep-15.0.0-test", "version": "15.0.0", "env": {}}},
        )
        entry = store / "ripgrep-15.0.0-test"
        entry.mkdir()
        binary = entry / ("rg.exe" if sys.platform == "win32" else "rg")
        binary.write_text("", encoding="utf-8")

        assert doctor._pm_tool_path("ripgrep") == binary

    def test_unstaged_tool_resolves_to_none(self, tmp_path, monkeypatch):
        self._store(tmp_path, monkeypatch, {})
        assert doctor._pm_tool_path("ripgrep") is None

    def test_recorded_but_deleted_binary_resolves_to_none(self, tmp_path, monkeypatch):
        self._store(
            tmp_path, monkeypatch,
            {"ripgrep": {"entry": "ripgrep-15.0.0-test", "version": "15.0.0", "env": {}}},
        )
        assert doctor._pm_tool_path("ripgrep") is None
