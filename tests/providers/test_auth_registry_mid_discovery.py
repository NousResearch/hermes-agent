"""Regression tests for #102123: plugins discovered after ``hermes_cli.auth`` is
first imported must still reach ``PROVIDER_REGISTRY``.

``hermes_cli.auth`` mirrors provider-plugin profiles into ``PROVIDER_REGISTRY``
when it is imported.  If a plugin's own imports pull ``hermes_cli.auth`` in
while ``providers._discover_providers()`` is still iterating the plugin
directories, that mirror runs against a partial profile list (the discovery
guard is already set, so ``list_providers()`` returns whatever has been
registered so far).  Every plugin discovered afterwards was invisible to
``resolve_provider()`` and failed with "Unknown provider".
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import providers
import hermes_cli.auth as auth_mod
from providers.base import ProviderProfile

REPO_ROOT = Path(__file__).resolve().parents[2]

_PLAIN_PROFILE = (
    "from providers import register_provider\n"
    "from providers.base import ProviderProfile\n"
    "register_provider(ProviderProfile(\n"
    "    name={name!r},\n"
    "    aliases=({alias!r},),\n"
    "    env_vars=('{env}',),\n"
    "    base_url='https://{name}.example/v1',\n"
    "    auth_type='api_key',\n"
    "))\n"
)


# ---------------------------------------------------------------------------
# End-to-end: a real plugin tree, a fresh interpreter, the public CLI gate.
# ---------------------------------------------------------------------------

def _write_plugin(root: Path, name: str, body: str) -> None:
    plugin_dir = root / "plugins" / "model-providers" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\nkind: model-provider\nversion: 0.0.1\ndescription: probe\n",
        encoding="utf-8",
    )


def _run_probe(hermes_home: Path, code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_PROFILE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_plugins_discovered_after_auth_import_resolve(tmp_path):
    hermes_home = tmp_path / ".hermes"
    # Sorted first: a plugin whose imports drag hermes_cli.auth in mid-discovery
    # (any plugin importing agent.credential_pool or similar does this).
    _write_plugin(
        hermes_home,
        "aaa-early-probe",
        "import hermes_cli.auth  # noqa: F401 — simulate a core-importing plugin\n"
        + _PLAIN_PROFILE.format(
            name="aaa-early-probe", alias="aaa-alias", env="AAA_EARLY_PROBE_KEY"
        ),
    )
    # Sorted last: an ordinary plugin discovered after that import.
    _write_plugin(
        hermes_home,
        "zzz-late-probe",
        _PLAIN_PROFILE.format(
            name="zzz-late-probe", alias="zzz-alias", env="ZZZ_LATE_PROBE_KEY"
        ),
    )

    probe = _run_probe(
        hermes_home,
        "import providers\n"
        "names = {p.name for p in providers.list_providers()}\n"
        "assert {'aaa-early-probe', 'zzz-late-probe'} <= names, names\n"
        "from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider\n"
        # Discovery completion must have mirrored the late plugin already;
        # consumers that read PROVIDER_REGISTRY directly rely on this.
        "assert 'zzz-late-probe' in PROVIDER_REGISTRY, sorted(PROVIDER_REGISTRY)\n"
        "assert resolve_provider('aaa-early-probe') == 'aaa-early-probe'\n"
        "assert resolve_provider('zzz-late-probe') == 'zzz-late-probe'\n"
        "assert resolve_provider('zzz-alias') == 'zzz-late-probe'\n"
        "cfg = PROVIDER_REGISTRY['zzz-late-probe']\n"
        "assert cfg.api_key_env_vars == ('ZZZ_LATE_PROBE_KEY',), cfg\n"
        "assert cfg.inference_base_url == 'https://zzz-late-probe.example/v1', cfg\n",
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


# ---------------------------------------------------------------------------
# In-process: the sync hook's contract (partial snapshot reconciled, idempotent,
# never imports hermes_cli.auth on its own).
# ---------------------------------------------------------------------------

EARLY = "probe-102123-early"
LATE = "probe-102123-late"
LATE_ALIAS = "probe-102123-late-alias"


@pytest.fixture()
def _isolated_registries():
    """Snapshot both registries; restore on teardown so nothing leaks."""
    saved_registry = dict(providers._REGISTRY)
    saved_aliases = dict(providers._ALIASES)
    saved_discovered = providers._discovered
    saved_auth_keys = set(auth_mod.PROVIDER_REGISTRY)
    saved_plugin_modules = {
        m for m in sys.modules if m.startswith("plugins.model_providers")
    }
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    providers._discovering = False
    yield
    for key in set(auth_mod.PROVIDER_REGISTRY) - saved_auth_keys:
        del auth_mod.PROVIDER_REGISTRY[key]
    for mod in [
        m
        for m in sys.modules
        if m.startswith("plugins.model_providers")
        and m not in saved_plugin_modules
    ]:
        del sys.modules[mod]
    providers._REGISTRY.clear()
    providers._REGISTRY.update(saved_registry)
    providers._ALIASES.clear()
    providers._ALIASES.update(saved_aliases)
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = saved_discovered
    providers._discovering = False


def test_post_discovery_registration_is_mirrored(_isolated_registries, monkeypatch, tmp_path):
    """A register_provider() call after discovery finished reaches the auth registry at once."""
    monkeypatch.setattr(providers, "_discover_entry_point_providers", lambda: None)
    monkeypatch.setattr(providers, "_BUNDLED_PLUGINS_DIR", tmp_path)
    monkeypatch.setattr(providers, "_user_plugins_dir", lambda: None)
    monkeypatch.setattr(providers, "_installed_plugins_dir", lambda: None)
    providers._discover_providers()

    providers.register_provider(
        ProviderProfile(
            name=LATE,
            display_name="Late",
            base_url="https://late.example/v1",
            env_vars=("PROBE_102123_LATE_KEY",),
            aliases=(LATE_ALIAS,),
        )
    )
    assert LATE in auth_mod.PROVIDER_REGISTRY
    assert auth_mod.PROVIDER_REGISTRY[LATE_ALIAS] is auth_mod.PROVIDER_REGISTRY[LATE]
