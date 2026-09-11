"""Reusing a bundle cache must not ship packages removed from its selection."""
from types import SimpleNamespace

from pm.lock import Facts, Lockfile
from scripts.bundles import native


def test_cached_bundle_prunes_unselected_facts_and_entries(tmp_path, monkeypatch):
    output = tmp_path / "payload"
    staged_store = output / "tools"
    user_store = tmp_path / "user-tools"
    selected = {"agent-browser", "chromium", "uv", "python"}
    stale = {"chromium-headless-shell", "retired-tool"}
    for root in (staged_store, user_store):
        root.mkdir(parents=True)
        facts = Facts(root / "facts.json")
        for name in selected | stale:
            entry = f"cached-{name}"
            (root / entry).mkdir()
            (root / entry / "payload").write_text(name, encoding="utf-8")
            facts.record(name, "fixture", entry, {}, root)
        (root / "orphaned-version").mkdir()
        (root / ".partials").mkdir()
    user_facts_before = (user_store / "facts.json").read_bytes()
    user_entries_before = {p.name for p in user_store.iterdir()}

    # Chromium is a dependency, not a selected root; Python is supplied by
    # bundle selection and uv must survive despite being internal.
    lock = Lockfile(tmp_path / "lock.json")
    for name in ("agent-browser", "uv"):
        lock.set_pin(name, "fixture", {})
    lock.save()
    monkeypatch.setattr(native, "_lockfile", lambda: lock)
    from pathlib import Path
    import shutil
    staged_lock = output / "hermes-agent/pm/lock.json"
    staged_lock.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parents[2] / "pm/lock.json", staged_lock)
    monkeypatch.setattr("scripts.bundles.payload.snapshot", lambda *args: None)
    monkeypatch.setattr(native, "_install_names", lambda names: 0)
    # Stop after cleanup, before invoking any venv/build subprocesses.
    monkeypatch.setattr(native, "pm_uv", lambda: (None, {}))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(user_store))

    assert native._stage_native(SimpleNamespace(out=str(output), ref="HEAD")) == 1

    remaining = Facts(staged_store / "facts.json")
    for name in stale:
        assert remaining.get(name) is None
        assert not (staged_store / f"cached-{name}").exists()
    for name in selected:
        fact = remaining.get(name)
        assert fact is not None
        assert fact["entry"] == f"cached-{name}"
        assert (staged_store / f"cached-{name}" / "payload").read_text(encoding="utf-8") == name
    assert not (staged_store / "orphaned-version").exists()
    assert (staged_store / ".partials").is_dir()
    assert native._store().root == user_store
    assert (user_store / "facts.json").read_bytes() == user_facts_before
    assert {p.name for p in user_store.iterdir()} == user_entries_before
