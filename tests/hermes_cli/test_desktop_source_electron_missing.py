"""A current renderer must not hide a source build whose linked Electron is gone.

A Windows ZIP update drops workspace node_modules (pnpm's links do not survive the
copy). If the reinstall then fails, apps/desktop/dist and its freshness receipt are
still current, and `hermes desktop --source` would skip the build and fail to launch
the missing runtime instead of relinking it.
"""
from hermes_agent.hermes_cli import main_desktop, source_build


def _source_tree(tmp_path, monkeypatch, *, electron: bool):
    desktop = tmp_path / "apps/desktop"
    (desktop / "dist").mkdir(parents=True)
    (desktop / "dist/index.html").write_text("<html></html>", encoding="utf-8")
    if electron:
        (desktop / "node_modules/electron").mkdir(parents=True)
        (desktop / "node_modules/electron/package.json").write_text("{}", encoding="utf-8")
    # The compiler owns receipts; hold that check current to isolate the runtime boundary.
    monkeypatch.setattr(source_build, "source_product_current", lambda *_: True)
    return tmp_path, desktop


def test_missing_linked_electron_needs_a_source_build(tmp_path, monkeypatch):
    root, desktop = _source_tree(tmp_path, monkeypatch, electron=False)

    assert main_desktop._desktop_build_needed(desktop, root, source_mode=True)


def test_current_source_build_with_its_electron_is_reused(tmp_path, monkeypatch):
    root, desktop = _source_tree(tmp_path, monkeypatch, electron=True)

    assert not main_desktop._desktop_build_needed(desktop, root, source_mode=True)
