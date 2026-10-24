"""``load_config_readonly()`` is read-only toward the FILESYSTEM too.

The defect this pins against (eager-tool-discovery audit, 2026-08-09):
``_load_config_impl`` called ``ensure_hermes_home()`` unconditionally, so a
function named *readonly* scaffolded eleven directories and wrote ``SOUL.md``
into whatever home the environment resolved. Because the config read sits on
the tool tree's import chain, that made pytest COLLECTION with an exported
``HERMES_HOME`` materialize a home skeleton before any fixture ran, and made
every "look at a setting" hot path a potential writer.

The contract now: reading config must leave a nonexistent home nonexistent.
``load_config()`` — the mutate-then-``save_config`` path — still ensures the
home, as do the explicit write paths.
"""

from __future__ import annotations


def test_load_config_readonly_does_not_materialize_the_home(tmp_path, monkeypatch):
    home = tmp_path / "never_materialized"
    assert not home.exists()

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HEAD_HOME", raising=False)

    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()

    assert isinstance(config, dict) and config, "readonly load must still answer"
    assert not home.exists(), (
        "load_config_readonly() materialized the home directory — a reader "
        "scaffolding the filesystem is the eager-tool-discovery defect class; "
        f"created: {sorted(p.name for p in home.iterdir()) if home.exists() else []}"
    )


def test_load_config_still_ensures_the_home(tmp_path, monkeypatch):
    """The mutable path keeps its guarantee — the split must not overshoot.

    Callers of ``load_config()`` mutate the result and hand it to
    ``save_config``; the home existing afterward is part of that path's
    long-standing contract and every write path relies on it.
    """

    home = tmp_path / "materialized"
    assert not home.exists()

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HEAD_HOME", raising=False)

    from hermes_cli.config import load_config

    config = load_config()

    assert isinstance(config, dict) and config
    assert home.is_dir(), "load_config() must keep ensuring the home"


def _existing_home_with_config(tmp_path, monkeypatch, name):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: probe-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HEAD_HOME", raising=False)
    return home


def test_load_config_readonly_writes_nothing_into_an_existing_home(tmp_path, monkeypatch):
    """With the home present and config.yaml parsing, a read-only load still writes nothing
    (no last-known-good copy under ``backups/``)."""
    home = _existing_home_with_config(tmp_path, monkeypatch, "present_ro")
    before = sorted(p.relative_to(home) for p in home.rglob("*"))

    from hermes_cli.config import load_config_readonly

    assert load_config_readonly()["model"]["default"] == "probe-model"
    assert sorted(p.relative_to(home) for p in home.rglob("*")) == before


def test_load_config_still_keeps_a_last_known_good_copy(tmp_path, monkeypatch):
    """Positive control for the test above: the same home under ``load_config()`` does get
    its ``backups/`` copy, so the read-only case is not passing for want of a backup path."""
    home = _existing_home_with_config(tmp_path, monkeypatch, "present_rw")

    from hermes_cli.config import load_config
    from hermes_cli.config_backups import list_config_backups

    assert load_config()["model"]["default"] == "probe-model"
    assert list_config_backups(home / "config.yaml", "good")
