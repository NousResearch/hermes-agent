import importlib
import sys
import types
from pathlib import Path


def test_gateway_session_imports_with_stale_utils_without_atomic_replace(
    monkeypatch, tmp_path: Path
) -> None:
    stale_utils = types.ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", stale_utils)
    monkeypatch.delitem(sys.modules, "gateway", raising=False)
    monkeypatch.delitem(sys.modules, "gateway.session", raising=False)

    session = importlib.import_module("gateway.session")

    target = tmp_path / "sessions.json"
    target.write_text("old", encoding="utf-8")
    link = tmp_path / "sessions-link.json"
    link.symlink_to(target)
    tmp = tmp_path / ".sessions.tmp"
    tmp.write_text("new", encoding="utf-8")

    replaced_path = session.atomic_replace(tmp, link)

    assert Path(replaced_path) == target
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "new"
