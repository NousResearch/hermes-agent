"""Regression tests for the shared package-shaped ``dotenv`` stub.

``gateway.run`` loads ``<repo>/.env`` at import time, so the gateway/provider
tests patch ``sys.modules["dotenv"]`` to keep a developer checkout's launch env
out of the test process.  A bare ``types.ModuleType("dotenv")`` exposing only
``load_dotenv`` is not enough once that ``.env`` exists: the load falls through
to ``hermes_cli.env_loader._load_dotenv_with_fallback``, which re-parses the file
with ``from dotenv.main import DotEnv`` and needs ``dotenv.variables`` too.

A fresh clone has no ``.env``, so CI never reaches that path — which is exactly
why ``tests/dotenv_stub`` exists.  Keep it package-shaped and keep it inert.
"""

from __future__ import annotations

import os
import sys

from tests.dotenv_stub import install


def test_stub_resolves_the_fallback_parser_imports(monkeypatch):
    """``env_loader``'s fallback path must import against the stub."""
    install(monkeypatch)

    from dotenv.main import DotEnv
    from dotenv.variables import parse_variables

    assert list(DotEnv("/nonexistent").parse()) == []
    assert parse_variables("A=1") == []


def test_stub_keeps_the_environment_inert(monkeypatch):
    """The stub must not publish anything into ``os.environ``."""
    install(monkeypatch)
    monkeypatch.delenv("HERMES_DOTENV_STUB_CANARY", raising=False)

    import dotenv

    dotenv.load_dotenv("/nonexistent")

    assert "HERMES_DOTENV_STUB_CANARY" not in os.environ


def test_stub_registers_every_layer_without_monkeypatch():
    """The module-level provider tests rely on the bare ``install()`` form."""
    previous = {
        name: sys.modules.get(name)
        for name in ("dotenv", "dotenv.main", "dotenv.variables")
    }
    try:
        fake = install()
        assert sys.modules["dotenv"] is fake
        assert sys.modules["dotenv.main"] is fake.main
        assert sys.modules["dotenv.variables"] is fake.variables
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
