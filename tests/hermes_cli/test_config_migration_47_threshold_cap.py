"""Config v47 (drop the briefly-shipped ``threshold_tokens: 256000`` default) must also run on an
unversioned config.yaml. The cap was copied into user files by the template seeder and targeted
writers while #115986's 256K default was live, and those files carry no ``_config_version``, so
without v47 in LEGACY_KEY_STEPS the stale cap survives every upgrade and keeps compacting
large-window models at 256K regardless of ``compression.threshold`` (#136350)."""

import pytest


def _raw(home):
    import hermes_yaml as yaml

    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write(home, body):
    (home / "config.yaml").write_text(body, encoding="utf-8")


def _run(current_ver, unversioned):
    from hermes_cli.config_migrations import run_migrations

    results = {"config_added": [], "warnings": []}
    run_migrations(current_ver, results, quiet=True, unversioned=unversioned)
    return results


def test_unversioned_seeded_cap_is_dropped(hermes_home):
    _write(
        hermes_home,
        (
            "compression:\n"
            "  enabled: true\n"
            "  threshold: 0.5\n"
            "  threshold_tokens: 256000\n"
        ),
    )

    _run(0, unversioned=True)

    compression = _raw(hermes_home)["compression"]
    assert "threshold_tokens" not in compression
    # Sibling keys — and the section — survive the rewrite untouched.
    assert compression["enabled"] is True
    assert compression["threshold"] == 0.5


def test_unversioned_explicit_cap_is_kept(hermes_home):
    _write(hermes_home, "compression:\n  threshold_tokens: 200000\n")

    _run(0, unversioned=True)

    assert _raw(hermes_home)["compression"]["threshold_tokens"] == 200000


def test_unversioned_explicit_null_is_kept(hermes_home):
    _write(hermes_home, "compression:\n  threshold_tokens: null\n")

    _run(0, unversioned=True)

    assert _raw(hermes_home)["compression"]["threshold_tokens"] is None


def test_versioned_still_drops_the_seeded_cap(hermes_home):
    _write(
        hermes_home, ("_config_version: 46\ncompression:\n  threshold_tokens: 256000\n")
    )

    _run(46, unversioned=False)

    assert "threshold_tokens" not in _raw(hermes_home)["compression"]
