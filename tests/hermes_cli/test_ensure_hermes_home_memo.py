"""ensure_hermes_home is memoized per home path (perf: it runs on every
load_config), but a deleted home must still be recreated on the next call."""

import shutil

from hermes_cli import config as cfg


def test_repeat_calls_are_memoized_but_deleted_home_is_recreated(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    cfg.ensure_hermes_home()
    assert (home / "sessions").is_dir()

    # Memoized: a second call must not recreate a removed SUBDIR (the fast
    # path only re-checks the home root)…
    shutil.rmtree(home / "sessions")
    cfg.ensure_hermes_home()
    assert not (home / "sessions").exists()

    # …but a vanished HOME re-runs the full walk and restores the skeleton.
    shutil.rmtree(home)
    cfg.ensure_hermes_home()
    assert (home / "sessions").is_dir()


def test_distinct_home_paths_each_get_the_skeleton(tmp_path, monkeypatch):
    first = tmp_path / "a" / ".hermes"
    second = tmp_path / "b" / ".hermes"

    monkeypatch.setenv("HERMES_HOME", str(first))
    cfg.ensure_hermes_home()

    # Profile switch: HERMES_HOME moves → the new path is ensured too.
    monkeypatch.setenv("HERMES_HOME", str(second))
    cfg.ensure_hermes_home()

    assert (first / "logs").is_dir()
    assert (second / "logs").is_dir()
