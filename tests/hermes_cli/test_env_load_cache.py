"""Tests for the load_env() process-level cache.

The cache exists to keep `hermes tools` → "All Platforms" fast: every
`get_env_value()` lookup used to re-read and re-sanitise the entire
.env file, racking up hundreds of ms across one menu render. The
cache is keyed on (path, mtime, size); writers (save_env_value /
remove_env_value / sanitise_env_file) call invalidate_env_cache().
"""

from __future__ import annotations

from pathlib import Path


def _write_env(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")






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


