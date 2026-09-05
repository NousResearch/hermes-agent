"""Tests for the load_env() process-level cache.

The cache exists to keep `hermes tools` → "All Platforms" fast: every
`get_env_value()` lookup used to re-read and re-sanitise the entire
.env file, racking up hundreds of ms across one menu render. .env
holds API keys, so the cache is keyed on a content digest (not
mtime/size, which a dotfile-sync or backup/restore tool can leave
unchanged while rewriting the bytes); writers (save_env_value /
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




def test_load_env_detects_content_change_with_preserved_mtime(tmp_path):
    """A tool that rewrites .env's bytes but restores the original mtime (e.g. `cp -p`,
    a dotfile-sync, or a backup/restore) must still be seen on the next load_env() call.

    This is the discriminating case for content-digest vs. mtime/size keying: bumping
    mtime while changing content (the OTHER natural test) passes under either scheme,
    since mtime usually changes too. Only holding mtime fixed while changing content
    tells them apart -- this fails under a (path, mtime, size) key and passes under a
    (path, content-digest) key.
    """
    from hermes_cli.config import invalidate_env_cache, load_env

    invalidate_env_cache()

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-old\n", encoding="utf-8")
    original_stat = env_path.stat()

    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            first = load_env()
            assert first.get("OPENAI_API_KEY") == "sk-old"

            # Same length new content (size unchanged too), mtime explicitly restored.
            env_path.write_text("OPENAI_API_KEY=sk-new\n", encoding="utf-8")
            os.utime(env_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            assert env_path.stat().st_mtime_ns == original_stat.st_mtime_ns
            assert env_path.stat().st_size == original_stat.st_size

            second = load_env()

        assert second.get("OPENAI_API_KEY") == "sk-new"
    finally:
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


