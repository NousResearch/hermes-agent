"""Regression: mini_swe_runner loads latin-1/.cp1252 Hermes .env on import."""

import importlib
import os
import sys

import pytest


def test_mini_swe_runner_import_loads_latin1_hermes_env(tmp_path, monkeypatch):
    """Invalid UTF-8 in ~/.hermes/.env must not crash entrypoint dotenv load."""
    home = tmp_path / "hermes"
    home.mkdir()
    # 0xe9 is latin-1 'é'; not valid as a lone UTF-8 continuation.
    (home / ".env").write_bytes(b"MINI_SWE_LATIN1_PROBE=caf\xe9\n")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("MINI_SWE_LATIN1_PROBE", raising=False)

    sys.modules.pop("mini_swe_runner", None)
    importlib.import_module("mini_swe_runner")

    assert os.getenv("MINI_SWE_LATIN1_PROBE") == "café"
