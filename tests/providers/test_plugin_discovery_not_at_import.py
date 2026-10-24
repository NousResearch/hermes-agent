"""Importing a module that model-provider plugins themselves import must not run plugin discovery.

``list_providers()`` imports every ``HERMES_HOME/plugins`` model-provider plugin. When that happens inside the
body of ``hermes_cli.models`` (or a module it imports on the way in), a plugin doing
``from hermes_cli.models import _PROVIDER_MODELS`` meets a partially initialised module, raises a circular
``ImportError`` and registers nothing. Measured on a fleet home: ``import hermes_cli.models`` as the first
import left 0 of 52 plugin providers registered. Two module bodies did it: ``agent.model_metadata``
(``_PROVIDER_PREFIXES`` and the URL-map auto-extend) and ``hermes_cli.models_catalog_static`` (the
``CANONICAL_PROVIDERS`` plugin pass). Each case runs in a fresh interpreter, because import order is the bug.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME = "import-order-probe"

_PLUGIN = textwrap.dedent(
    """
    import os

    with open(os.environ["PROBE_LOG"], "a", encoding="utf-8") as log:
        log.write("imported\\n")
    # what real provider plugins read while they import
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_MODELS  # noqa: F401
    from agent.model_metadata import get_model_context_length  # noqa: F401
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="%s", display_name="Import Order Probe", auth_type="api_key",
                                      base_url="https://import-order-probe.invalid/v1", api_mode="chat_completions"))
    """ % NAME
)

_SCRIPT = textwrap.dedent(
    """
    import importlib, json, os, sys

    importlib.import_module(sys.argv[1])
    with open(os.environ["PROBE_LOG"], encoding="utf-8") as log:
        imported_during_import = bool(log.read())

    import providers
    from hermes_cli.models import CANONICAL_PROVIDERS, _KNOWN_PROVIDER_NAMES
    from agent import model_metadata

    print(json.dumps({
        "imported_during_import": imported_during_import,
        "registered": sys.argv[2] in {p.name for p in providers.list_providers()},
        "canonical_rows": [p.slug for p in CANONICAL_PROVIDERS].count(sys.argv[2]),
        "known_name": sys.argv[2] in _KNOWN_PROVIDER_NAMES,
        "prefix": sys.argv[2] in model_metadata._PROVIDER_PREFIXES,
        "url_provider": model_metadata._infer_provider_from_url("https://import-order-probe.invalid/v1"),
    }))
    """
)


@pytest.mark.parametrize("entry", ["hermes_cli.models", "hermes_cli.models_catalog_static", "agent.model_metadata"])
def test_importing_a_plugin_facing_module_first_does_not_run_discovery(entry, tmp_path):
    home = tmp_path / "home"
    plugin = home / "plugins" / NAME
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {NAME}\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    log = tmp_path / "probe.log"
    log.write_text("", encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(home), "PROBE_LOG": str(log), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}

    run = subprocess.run([sys.executable, "-c", _SCRIPT, entry, NAME], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=120)

    assert run.returncode == 0, run.stderr[-3000:]
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert "circular import" not in run.stderr, run.stderr[-3000:]
    assert result == {
        "imported_during_import": False,  # the module body did not import plugins
        "registered": True,  # ... so the plugin imported cleanly once discovery ran
        "canonical_rows": 1,  # plugin rows reach the catalog on first read, exactly once
        "known_name": True,
        "prefix": True,
        "url_provider": NAME,
    }
