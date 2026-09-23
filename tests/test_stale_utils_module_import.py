"""Regression for the stale-``utils``-module ImportError after a hot ``git pull``.

Real incident (gateway session 1518671026962174144)::

    Sorry, I encountered an error (ImportError).
    cannot import name 'env_float' from 'utils' (~/.hermes/hermes-agent/utils.py)

Mechanism:

1. A long-running gateway/agent process imported ``utils`` BEFORE ``env_float``
   existed (added in 06ca1e99, 2026-06-20 14:00). The cached module object in
   ``sys.modules`` therefore has no ``env_float`` attribute.
2. ``hermes update`` ran ``git pull``, updating ``utils.py`` (now defining
   ``env_float``) and ~22 consumer modules (now doing ``from utils import
   env_float``) on disk -- WITHOUT restarting the process.
3. Switching the live session's model (anthropic/opus -> opencode/glm) forced the
   FIRST import of a consumer module on the new provider's code path. Its
   top-level ``from utils import env_float`` resolved against the STALE cached
   ``utils`` -> ImportError. The path in parentheses is the consumer-reported
   ``utils.__file__`` on disk (which *does* define ``env_float``), which is why
   the error is so confusing: the file on disk is fine, the in-memory module is not.

``hermes_cli/main.py`` (the ``hermes update`` flow, ~line 9326) already
acknowledges this exact hazard -- "source files on disk are newer than cached
Python modules in this process" -- and reloads ``hermes_constants`` after the
pull, but NOT ``utils``. Any ``utils`` consumer added in the same release stays
exposed until the process restarts.

The messaging client (Discord/Telegram/Feishu/...) is incidental: the trigger is
a fresh import on a stale process, not the platform. We assert that below by
reproducing the failure with the Discord adapter's exact import line.
"""

import sys
import types

import pytest


def _import_fresh_consumer(name: str, source: str) -> types.ModuleType:
    """Import a brand-new module whose body runs ``source`` -- mimicking a
    consumer module being imported for the first time on the model-switch path."""
    mod = types.ModuleType(name)
    mod.__file__ = f"{name}.py"
    sys.modules.pop(name, None)
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod


class TestStaleUtilsModuleImport:
    def test_fresh_consumer_import_fails_against_stale_utils(self, monkeypatch):
        """The bug: stale in-memory ``utils`` + fresh ``from utils import env_float``."""
        import utils

        # Sanity: today's on-disk source is healthy.
        assert hasattr(utils, "env_float")

        # Simulate the pre-06-20 cached module (monkeypatch auto-restores after).
        monkeypatch.delattr(utils, "env_float")

        with pytest.raises(ImportError, match=r"cannot import name 'env_float' from 'utils'"):
            _import_fresh_consumer("stale_switch_path_consumer", "from utils import env_float\n")

    def test_client_is_incidental_discord_import_line_fails_identically(self, monkeypatch):
        """Same failure via the Discord adapter's exact import line -- the client
        does not determine the bug, the stale process does."""
        import utils

        monkeypatch.delattr(utils, "env_float")

        # plugins/platforms/discord/adapter.py:106
        with pytest.raises(ImportError, match=r"cannot import name 'env_float' from 'utils'"):
            _import_fresh_consumer(
                "stale_discord_consumer",
                "from utils import atomic_json_write, env_float\n",
            )

    def test_healthy_process_imports_consumer_fine(self):
        """Control: when the cached ``utils`` matches disk (env_float present),
        the same consumer import succeeds -- proving the harness isolates the
        staleness, not an unrelated import error."""
        import utils

        assert hasattr(utils, "env_float")
        mod = _import_fresh_consumer(
            "healthy_consumer",
            "from utils import env_float\nVALUE = env_float('UNSET_FOR_TEST', 1.5)\n",
        )
        assert mod.VALUE == 1.5
