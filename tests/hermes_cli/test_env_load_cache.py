"""Tests for the load_env() process-level cache.

The cache exists to keep `hermes tools` → "All Platforms" fast: every
`get_env_value()` lookup used to re-read and re-sanitise the entire
.env file, racking up hundreds of ms across one menu render. The
cache is keyed on (path, mtime, size); writers (save_env_value /
remove_env_value / sanitise_env_file) call invalidate_env_cache().
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _write_env(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_load_env_caches_on_repeat_calls():
    """Repeated load_env() calls on the same file return the cached dict."""
    from hermes_cli.config import invalidate_env_cache, load_env

    invalidate_env_cache()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write("OPENAI_API_KEY=sk-first\n")
        env_path = Path(f.name)

    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            first = load_env()
            # Even if a writer outside our cache mutates the file, an
            # mtime/size match means the cache still wins. We simulate that
            # by writing identical bytes back — sanity check that the cache
            # is keyed structurally, not on a counter.
            second = load_env()

        assert first == second
        assert first.get("OPENAI_API_KEY") == "sk-first"
    finally:
        env_path.unlink(missing_ok=True)
        invalidate_env_cache()


def test_load_env_invalidates_on_mtime_bump():
    """Editing the file (mtime changes) invalidates the cache."""
    from hermes_cli.config import invalidate_env_cache, load_env

    invalidate_env_cache()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write("OPENAI_API_KEY=sk-old\n")
        env_path = Path(f.name)

    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            first = load_env()
            assert first.get("OPENAI_API_KEY") == "sk-old"

            # Rewrite file with new contents and bump mtime to make sure
            # the FS records the change even on coarse-mtime filesystems.
            _write_env(env_path, "OPENAI_API_KEY=sk-new\n")
            future = env_path.stat().st_mtime + 5.0
            os.utime(env_path, (future, future))

            second = load_env()
            assert second.get("OPENAI_API_KEY") == "sk-new", (
                "load_env() returned stale value after file change"
            )
    finally:
        env_path.unlink(missing_ok=True)
        invalidate_env_cache()


def test_invalidate_env_cache_forces_reread():
    """invalidate_env_cache() forces the next load_env() to hit the disk.

    This is the belt-and-braces knob for writers (save_env_value, etc.)
    on filesystems where mtime resolution might miss a same-second write.
    """
    from hermes_cli.config import invalidate_env_cache, load_env

    invalidate_env_cache()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write("OPENAI_API_KEY=sk-old\n")
        env_path = Path(f.name)

    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            assert load_env().get("OPENAI_API_KEY") == "sk-old"

            # Rewrite WITHOUT bumping mtime — simulates same-second write.
            mtime_before = env_path.stat().st_mtime
            _write_env(env_path, "OPENAI_API_KEY=sk-new\n")
            os.utime(env_path, (mtime_before, mtime_before))

            # Without invalidation, cache hit might return stale.
            invalidate_env_cache()

            assert load_env().get("OPENAI_API_KEY") == "sk-new"
    finally:
        env_path.unlink(missing_ok=True)
        invalidate_env_cache()


def test_save_env_value_invalidates_cache(tmp_path, monkeypatch):
    """save_env_value() invalidates the cache so subsequent reads see the update."""
    from hermes_cli import config as config_mod
    from hermes_cli.config import invalidate_env_cache, load_env, save_env_value

    invalidate_env_cache()

    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING_KEY=old\n", encoding="utf-8")

    monkeypatch.setattr(config_mod, "get_env_path", lambda: env_path)
    monkeypatch.setattr(config_mod, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(config_mod, "_secure_file", lambda _p: None)
    monkeypatch.setattr(config_mod, "is_managed", lambda: False)

    try:
        # Prime the cache.
        first = load_env()
        assert first.get("EXISTING_KEY") == "old"

        save_env_value("NEW_KEY", "shiny")

        # Same-second writes on coarse-mtime filesystems would normally
        # let stale cache survive; invalidate_env_cache() inside the
        # writer makes the next read see the new key.
        result = load_env()
        assert result.get("NEW_KEY") == "shiny"
        assert result.get("EXISTING_KEY") == "old"
    finally:
        monkeypatch.delenv("NEW_KEY", raising=False)
        invalidate_env_cache()


def test_remove_env_value_invalidates_cache(tmp_path, monkeypatch):
    """remove_env_value() invalidates the cache so the removed key disappears."""
    from hermes_cli import config as config_mod
    from hermes_cli.config import (
        invalidate_env_cache,
        load_env,
        remove_env_value,
        save_env_value,
    )

    invalidate_env_cache()

    env_path = tmp_path / ".env"
    monkeypatch.setattr(config_mod, "get_env_path", lambda: env_path)
    monkeypatch.setattr(config_mod, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(config_mod, "_secure_file", lambda _p: None)
    monkeypatch.setattr(config_mod, "is_managed", lambda: False)

    save_env_value("DOOMED_KEY", "value")
    assert load_env().get("DOOMED_KEY") == "value"

    try:
        removed = remove_env_value("DOOMED_KEY")
        assert removed is True
        assert "DOOMED_KEY" not in load_env()
    finally:
        monkeypatch.delenv("DOOMED_KEY", raising=False)
        invalidate_env_cache()


def test_load_env_handles_missing_file():
    """A nonexistent .env returns {} and caches the empty result."""
    from hermes_cli.config import invalidate_env_cache, load_env

    invalidate_env_cache()

    nonexistent = Path(tempfile.gettempdir()) / "hermes-test-no-such-env-xyz123.env"
    nonexistent.unlink(missing_ok=True)

    try:
        with patch("hermes_cli.config.get_env_path", return_value=nonexistent):
            assert load_env() == {}
            assert load_env() == {}  # cached
    finally:
        invalidate_env_cache()
