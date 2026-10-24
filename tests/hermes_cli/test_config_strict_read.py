"""Regression for #141856: the strict pre-write config read re-parsed config.yaml on every call
(~31 ms/call on a real profile; tool hooks call it several times per tool call, stalling the
gateway's event loop under load). The parse is now cached on the file signature; failures are
never cached, and a signature inside the racy-mtime window is never trusted."""

import os
import time
from pathlib import Path

import pytest

import hermes_cli.config as config_mod
from hermes_cli import config_strict_read
from hermes_cli.config import require_readable_config_before_write
from hermes_cli.config_strict_read import _STRICT_READ_CACHE
from utils import RACY_MTIME_WINDOW_NS

_CONFIG = "model:\n  default: my-org/custom-model\n"


@pytest.fixture(autouse=True)
def _clear_strict_cache():
    _STRICT_READ_CACHE.clear()
    yield
    _STRICT_READ_CACHE.clear()


class _ParseCounter:
    """Counts strict parses of one config path through the fast_safe_load patch seam."""

    def __init__(self, monkeypatch, path: Path):
        self.parses = 0
        real = config_mod.fast_safe_load

        def counting(stream):
            if getattr(stream, "name", None) == str(path):
                self.parses += 1
            return real(stream)

        monkeypatch.setattr(config_mod, "fast_safe_load", counting)


def _write_config(path: Path, text: str = _CONFIG) -> None:
    path.write_text(text, encoding="utf-8")
    _age_out_of_racy_window(path)


def _age_out_of_racy_window(path: Path) -> None:
    aged = time.time_ns() - 2 * RACY_MTIME_WINDOW_NS
    os.utime(path, ns=(aged, aged))


def test_unchanged_file_parses_once_and_each_call_gets_its_own_copy(tmp_path, monkeypatch):
    """The contract the cache exists for: an unchanged file is parsed once, and callers that
    mutate the returned mapping before writing it back cannot corrupt the next caller's read."""
    path = tmp_path / "config.yaml"
    _write_config(path)
    counter = _ParseCounter(monkeypatch, path)

    first = require_readable_config_before_write(path)
    second = require_readable_config_before_write(path)

    assert counter.parses == 1
    assert second == first
    first["model"]["default"] = "mutated-by-caller"
    assert require_readable_config_before_write(path)["model"]["default"] == "my-org/custom-model"


def test_rewritten_file_is_reparsed_without_explicit_invalidation(tmp_path, monkeypatch):
    """The writers' atomic replace must invalidate by itself: the next guard reads the new file,
    never the cached old parse."""
    path = tmp_path / "config.yaml"
    _write_config(path)
    counter = _ParseCounter(monkeypatch, path)
    assert require_readable_config_before_write(path)["model"]["default"] == "my-org/custom-model"

    replacement = tmp_path / ".config.yaml.new"
    replacement.write_text("model:\n  default: other-org/other-model\n", encoding="utf-8")
    os.replace(replacement, path)
    _age_out_of_racy_window(path)

    assert require_readable_config_before_write(path)["model"]["default"] == "other-org/other-model"
    assert counter.parses == 2


def test_failures_are_never_cached(tmp_path, monkeypatch):
    """Every refusal re-validates the file: a malformed or non-mapping config refuses on every
    call, and the guard picks the file's recovery back up the moment it parses again."""
    path = tmp_path / "config.yaml"
    counter = _ParseCounter(monkeypatch, path)
    path.write_text("model: [unclosed\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="has a formatting error"):
            require_readable_config_before_write(path)
    assert counter.parses == 2

    path.write_text("- just\n- a list\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="must start with settings names"):
            require_readable_config_before_write(path)
    assert counter.parses == 4

    _write_config(path, _CONFIG)
    assert require_readable_config_before_write(path)["model"]["default"] == "my-org/custom-model"


def test_racy_signature_is_never_stored(tmp_path, monkeypatch):
    """A parse whose mtime sits in the racy window is not cached: a same-tick rewrite could hide
    behind an unchanged signature, so those reads keep re-parsing until the window clears."""
    path = tmp_path / "config.yaml"
    _write_config(path)
    counter = _ParseCounter(monkeypatch, path)

    monkeypatch.setattr(config_strict_read, "mtime_is_racy", lambda mtime_ns: True)
    require_readable_config_before_write(path)
    require_readable_config_before_write(path)
    assert counter.parses == 2, "a racy signature must not produce a trusted cache entry"

    monkeypatch.setattr(config_strict_read, "mtime_is_racy", lambda mtime_ns: False)
    require_readable_config_before_write(path)
    require_readable_config_before_write(path)
    assert counter.parses == 3, "once the window clears, the parse is stored and reused"


def test_default_path_resolves_the_profile_config(tmp_path, monkeypatch):
    """config_path=None reads the profile's own config.yaml (get_config_path), not a cached or
    foreign path."""
    from hermes_constants import get_hermes_home
    _write_config(get_hermes_home() / "config.yaml", _CONFIG)
    assert require_readable_config_before_write() == {"model": {"default": "my-org/custom-model"}}
