"""load_env() must not cache a failed read as the new truth.

Regression: when ``stat`` on ``~/.hermes/.env`` raised ``FileNotFoundError``
(e.g. the file is momentarily absent during an atomic rewrite), ``load_env``
stored an *empty* result under the failure cache key and returned it — and the
whole resolve path (``load_env`` → ``_load_hermes_env_vars`` → docker env
forwarding) logged nothing. A long-running gateway could silently lose
forwarded credentials for entire sessions (observed as ``GH_TOKEN`` vanishing
from ``docker exec -e`` injection with zero log lines).

The fix: failures never overwrite the memo; callers get the last successful
values back (with one warning per failure streak), a fresh install with no
.env stays silent, and the writer path (``invalidate_env_cache``) still
forgets everything so intentional edits behave as before.
"""

import logging

import pytest

import hermes_cli.config as cfg


@pytest.fixture()
def env_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    monkeypatch.setattr(cfg, "get_env_path", lambda: env)
    monkeypatch.setattr(cfg, "_env_cache", None)
    monkeypatch.setattr(cfg, "_env_load_failing", False)
    return env


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_missing_env_with_no_history_is_empty_and_quiet(env_file, caplog):
    caplog.set_level(logging.WARNING, logger=cfg.__name__)
    assert cfg.load_env() == {}
    assert _warnings(caplog) == []  # fresh install is a legitimate state


def test_transient_removal_falls_back_to_last_good_and_warns_once(env_file, caplog):
    caplog.set_level(logging.WARNING, logger=cfg.__name__)
    env_file.write_text("GH_TOKEN=abc\n", encoding="utf-8")
    assert cfg.load_env() == {"GH_TOKEN": "abc"}

    env_file.unlink()
    assert cfg.load_env() == {"GH_TOKEN": "abc"}  # last good, not empty
    assert cfg.load_env() == {"GH_TOKEN": "abc"}  # still cached, still quiet
    assert len(_warnings(caplog)) == 1  # one warning per streak, not per call


def test_recovery_and_second_streak_warns_again(env_file, caplog):
    caplog.set_level(logging.WARNING, logger=cfg.__name__)
    env_file.write_text("GH_TOKEN=abc\n", encoding="utf-8")
    cfg.load_env()
    env_file.unlink()
    cfg.load_env()
    assert len(_warnings(caplog)) == 1

    # File comes back with different content (different size => new cache key).
    env_file.write_text("GH_TOKEN=xyz-longer\n", encoding="utf-8")
    assert cfg.load_env() == {"GH_TOKEN": "xyz-longer"}

    env_file.unlink()
    assert cfg.load_env() == {"GH_TOKEN": "xyz-longer"}
    assert len(_warnings(caplog)) == 2  # new streak logs once more


def test_writer_invalidation_still_forgets_the_fallback(env_file, caplog):
    caplog.set_level(logging.WARNING, logger=cfg.__name__)
    env_file.write_text("GH_TOKEN=abc\n", encoding="utf-8")
    cfg.load_env()
    env_file.unlink()

    cfg.invalidate_env_cache()
    # Intentional writer-path edits must not resurrect stale values.
    assert cfg.load_env() == {}
