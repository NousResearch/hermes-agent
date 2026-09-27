"""A config.yaml write must survive a writer that read the file earlier and saves afterwards.

Regression for #96571 (``save_config`` with a stale dict reverted a ``save_config_value`` write) and
the concurrency scope of #66752: every read -> mutate -> whole-document save lost any write that
landed between its read and its save, in one process or across processes (gateway + CLI).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path

import pytest

import hermes_yaml

_KEY = "approvals.destructive_slash_confirm"  # schema default True, so a stale dict carries True
_A_SENTINEL = "a-value-7f3e"  # value written by the stale saver; must never reach a log line


def _disk(path: Path) -> dict:
    return hermes_yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _interleaved_write(writer: str) -> None:
    """Writer B: set ``_KEY`` to False through one real config write path."""
    if writer == "save_config_value":
        from cli import save_config_value
        assert save_config_value(_KEY, False)
    elif writer == "tui_write_config_key":
        from tui_gateway import server
        server._write_config_key(_KEY, False)
    else:
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        cfg["approvals"]["destructive_slash_confirm"] = False
        save_config(cfg)


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from tui_gateway import server
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_cfg_cache", None)
    return home


@pytest.mark.parametrize("on_disk", [
    "approvals:\n  mode: smart\n  destructive_slash_confirm: true\n",  # the #96571 script
    "approvals:\n  mode: smart\n",  # key absent: the stale dict's True is only the merged default
], ids=["key-on-disk", "key-absent"])
@pytest.mark.parametrize("writer", ["save_config_value", "tui_write_config_key", "save_config"])
@pytest.mark.parametrize("stale_saver", ["save_config", "tui_save_cfg"])
def test_stale_whole_document_save_keeps_a_write_that_landed_after_its_read(
        home, caplog, stale_saver, writer, on_disk):
    from hermes_cli.config import load_config, save_config
    from tui_gateway import server

    cfg_path = home / "config.yaml"
    cfg_path.write_text("# user comment\n" + on_disk, encoding="utf-8")
    stale = load_config() if stale_saver == "save_config" else server._load_cfg_raw()

    _interleaved_write(writer)
    assert _disk(cfg_path)["approvals"]["destructive_slash_confirm"] is False

    stale.setdefault("a_writer", {})["key"] = _A_SENTINEL
    caplog.set_level(logging.INFO)
    caplog.clear()
    (save_config if stale_saver == "save_config" else server._save_cfg)(stale)

    after = _disk(cfg_path)
    assert after["approvals"]["destructive_slash_confirm"] is False, "the interleaved write was reverted"
    assert after["approvals"]["mode"] == "smart"
    assert after["a_writer"]["key"] == _A_SENTINEL
    assert "# user comment" in cfg_path.read_text(encoding="utf-8")
    # One audit line for the stale save: it names the path the caller changed, never a value, and
    # never claims the key it did not touch.
    audit = [r.getMessage() for r in caplog.records if str(cfg_path) in r.getMessage()]
    assert len(audit) == 1 and "a_writer.key" in audit[0], audit
    assert _KEY not in audit[0]
    assert not any(_A_SENTINEL in r.getMessage() for r in caplog.records)


def _write_keys(home: str, writer: str, barrier) -> None:
    os.environ["HERMES_HOME"] = home
    from cli import save_config_value
    from tui_gateway import server

    barrier.wait(timeout=60)
    for i in range(40):
        if writer == "tui":
            server._write_config_key(f"tui_keys.k{i}", i)
        else:
            assert save_config_value(f"cli_keys.k{i}", i)


def test_two_processes_writing_config_concurrently_lose_no_keys(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [context.Process(target=_write_keys, args=(str(home), writer, barrier))
                 for writer in ("tui", "cli")]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    after = _disk(home / "config.yaml")
    assert after.get("tui_keys") == {f"k{i}": i for i in range(40)}
    assert after.get("cli_keys") == {f"k{i}": i for i in range(40)}
    assert after["approvals"]["mode"] == "manual"


def test_a_stale_dict_keeps_its_snapshot_while_many_newer_loaded_dicts_are_alive(home):
    """Provenance must not depend on how many other loaded configs are alive: a registry that evicted
    past a count cap saved the oldest caller's dict whole again, reverting a later write."""
    from hermes_cli.config import load_config, save_config

    cfg_path = home / "config.yaml"
    cfg_path.write_text("approvals:\n  destructive_slash_confirm: true\n", encoding="utf-8")
    stale = load_config()
    alive = [load_config() for _ in range(200)]

    _interleaved_write("save_config_value")
    stale.setdefault("a_writer", {})["key"] = _A_SENTINEL
    save_config(stale)

    after = _disk(cfg_path)
    assert after["approvals"]["destructive_slash_confirm"] is False, "the interleaved write was reverted"
    assert after["a_writer"]["key"] == _A_SENTINEL
    assert len(alive) == 200


def test_a_save_based_on_a_missing_config_keeps_a_first_write_made_after_the_read(home):
    """Reading an absent config.yaml served an untracked {}, so saving it back deleted a config
    another writer had created after that read."""
    from hermes_cli.config import read_raw_config, save_config

    cfg_path = home / "config.yaml"
    assert not cfg_path.exists()
    stale = read_raw_config()

    _interleaved_write("save_config_value")
    stale.setdefault("a_writer", {})["key"] = _A_SENTINEL
    save_config(stale)

    after = _disk(cfg_path)
    assert after["approvals"]["destructive_slash_confirm"] is False, "the first write was deleted"
    assert after["a_writer"]["key"] == _A_SENTINEL


def test_update_model_restore_keeps_a_config_write_that_lands_during_it(home, monkeypatch):
    """The post-update model restore read config.yaml with its own loader and wrote that whole dict
    back, reverting a config write that landed after its read."""
    from hermes_cli import backup

    cfg_path = home / "config.yaml"
    cfg_path.write_text("model:\n  default: changed-model\napprovals:\n  destructive_slash_confirm: true\n",
                        encoding="utf-8")
    snapshot = backup._quick_snapshot_root(home) / "pre-update"
    snapshot.mkdir(parents=True)
    (snapshot / "config.yaml").write_text("model:\n  default: original-model\n", encoding="utf-8")

    read = backup._read_raw_yaml_dict
    interleaved = []

    def read_then_interleave(path):
        loaded = read(path)
        if Path(path) == cfg_path and not interleaved:
            interleaved.append(True)
            _interleaved_write("save_config_value")
        return loaded

    monkeypatch.setattr(backup, "_read_raw_yaml_dict", read_then_interleave)
    result = backup.restore_config_model_settings_if_rewritten("pre-update", home)

    after = _disk(cfg_path)
    assert interleaved
    assert result and result["keys"] == ["model.default"]
    assert after["model"]["default"] == "original-model"
    assert after["approvals"]["destructive_slash_confirm"] is False, "the interleaved write was reverted"
