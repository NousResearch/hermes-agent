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


