"""A transient sharing violation on config.yaml must not degrade the config to defaults.

Live evidence (TONY, 2026-09-12): a read denied with ``[Errno 13] Permission denied`` while
another process held config.yaml reached the parse-failure funnel, warned "Failed to parse …
Fix the YAML", fell back to ``DEFAULT_CONFIG`` (dropping the user's fallback chain and model
routing) and snapshotted a "corrupt" copy that was byte-identical to the live file.

Contract asserted here:

  1. A lock that clears on retry is not a parse failure — the user's overrides survive and no
     snapshot is taken.
  2. A genuinely unparseable file still takes the parse-failure path (loud warning + snapshot),
     because that is what tells the user what is actually wrong.
"""

from __future__ import annotations

import builtins
import errno

import pytest

from hermes_cli import config as config_mod


@pytest.fixture
def config_home(tmp_path):
    """Temp config.yaml plus a clean loader cache/state for this path."""
    path = config_mod.get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config_mod._CONFIG_PARSE_FAILURES.clear()
    config_mod._CONFIG_PARSE_WARNED.clear()
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    return path


def _corrupt_snapshots(path) -> list:
    backups = path.parent / "backups" / "config"
    return sorted(backups.glob("*.corrupt.*")) if backups.exists() else []


def test_transient_share_violation_is_retried_not_treated_as_broken_yaml(
    config_home, monkeypatch, caplog,
):
    config_home.write_text("display:\n  skin: usertheme\n", encoding="utf-8")
    real_open = builtins.open
    denied = {"count": 0}

    def flaky_open(file, *args, **kwargs):
        # Only the first read of config.yaml is denied; the file itself is perfectly valid.
        if str(file) == str(config_home) and denied["count"] == 0:
            denied["count"] += 1
            raise PermissionError(errno.EACCES, "Permission denied", str(config_home))
        return real_open(file, *args, **kwargs)

    # Patch the BUILTIN, not ``utils.open``: the point is that the loader's read is denied and the
    # retry has to absorb it — a module-attribute seam would only prove the helper exists.
    monkeypatch.setattr(builtins, "open", flaky_open)

    with caplog.at_level("WARNING"):
        cfg = config_mod.load_config()

    assert denied["count"] == 1, "the blocked read was not retried"
    assert cfg["display"]["skin"] == "usertheme", "user override lost to DEFAULT_CONFIG"
    assert config_mod._CONFIG_PARSE_FAILURES == {}, "a cleared lock was recorded as a parse failure"
    assert "Failed to parse" not in caplog.text
    assert _corrupt_snapshots(config_home) == [], "a valid config was snapshotted as corrupt"


def test_unparseable_yaml_still_reports_a_parse_failure(config_home, caplog):
    config_home.write_text("display: [unclosed\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        config_mod.load_config()

    assert "Failed to parse" in caplog.text, "a real YAML error must stay loud"
    assert config_mod._CONFIG_PARSE_FAILURES, "provider auto-resolution relies on this record"
    assert _corrupt_snapshots(config_home), "a real parse failure still snapshots the file"
