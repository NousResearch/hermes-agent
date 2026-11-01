"""The provider entry-point gate must not drag the plugin manager into every process.

``hermes_cli.config`` runs ``_inject_profile_env_vars()`` at import time, which reaches
``providers.list_providers()`` → ``_discover_entry_point_providers()``. That gate only needs
``plugins.enabled`` / ``plugins.disabled``, both owned by ``hermes_cli.plugins_discovery``;
importing the ``hermes_cli.plugins`` facade for them pulled loader, dispatch, ledger and
middleware into every CLI invocation, cron tick and `-z` subprocess (~170ms) — and then threw the
work away on the opt-in default, where nothing is enabled.

A subprocess is the only honest check: in-process, another test may already have imported the
manager.
"""

import subprocess
import sys

import pytest

_PROBE = (
    "import sys\n"
    "import hermes_cli.config\n"
    "heavy = [m for m in ('hermes_cli.plugins_loader', 'hermes_cli.plugins_dispatch',\n"
    "                     'hermes_cli.plugins_ledger', 'hermes_cli.middleware')\n"
    "         if m in sys.modules]\n"
    "print('HEAVY:' + '|'.join(heavy))\n"  # marker: the clean case prints an otherwise empty line
)


def _run_probe(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)


def test_importing_config_does_not_load_the_plugin_manager():
    result = _run_probe(_PROBE)

    assert result.returncode == 0, result.stderr
    marker = next((line for line in result.stdout.splitlines() if line.startswith("HEAVY:")), None)
    assert marker is not None, f"probe printed no result: {result.stdout!r} / {result.stderr!r}"
    loaded = [m for m in marker[len("HEAVY:"):].split("|") if m]
    assert loaded == [], f"importing hermes_cli.config pulled in the plugin manager: {loaded}"


def test_gate_still_reads_both_plugin_lists(monkeypatch):
    """The cheap import must expose the same two gates the facade re-exported."""
    from hermes_cli import plugins_discovery
    import providers

    seen = {}
    monkeypatch.setattr(plugins_discovery, "_get_enabled_plugins", lambda: seen.setdefault("enabled", set()))
    monkeypatch.setattr(plugins_discovery, "_get_disabled_plugins", lambda: seen.setdefault("disabled", set()))

    providers._discover_entry_point_providers()

    assert seen == {"enabled": set(), "disabled": set()}


@pytest.mark.parametrize("gate", ["_get_enabled_plugins", "_get_disabled_plugins"])
def test_facade_and_owner_expose_the_same_gate(gate):
    """`hermes_cli.plugins` only re-exports these; both names must stay in sync."""
    from hermes_cli import plugins, plugins_discovery

    assert getattr(plugins, gate) is getattr(plugins_discovery, gate)
