"""Provider auto-extend must be LAZY, never at ``hermes_cli.models`` import.

Regression for the circular-import class: ``hermes_cli/models.py`` (and
``agent/model_metadata.py``) used to call ``providers.list_providers()`` in
their module bodies. ``list_providers()`` imports every model-provider plugin,
and plugins routinely do ``from hermes_cli.models import _PROVIDER_MODELS,
CANONICAL_PROVIDERS`` at their own import time — so discovery ran against a
partially initialised ``hermes_cli.models``:

* models-first import order  -> plugin raises ``ImportError: cannot import name
  'CANONICAL_PROVIDERS' from partially initialized module``, and
  ``_PROVIDER_MODELS`` was empty for every pin.
* discovery-first order      -> the plugin's own provider never reached
  ``CANONICAL_PROVIDERS``.

Both orders are exercised here against a real temp ``HERMES_HOME`` plugin root,
in a subprocess (import order is a process-global property, so each case needs
a fresh interpreter). The driver does not merely import the picker — it calls
``list_picker_providers()`` and ``provider_label()`` and asserts on their
output, and a plugin-free baseline run is compared against the plugin run so
"no behaviour change for existing providers" is measured rather than assumed.
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

PLUGIN_NAME = "zz-lazy-canonical-probe"
PROBE_KEY_ENV = "ZZ_LAZY_CANONICAL_PROBE_KEY"
PROBE_MODELS = ["probe-model-a", "probe-model-b"]

# The plugin does exactly what every claude-apx-N / claude-bpx-N clone does:
# imports the two models.py surfaces at its own import time, then registers.
# ``fallback_models`` is what lets the registered provider produce a real
# picker row (rows with no models are filtered out by the picker), so the
# picker assertions below have something to assert about.
PLUGIN_INIT = textwrap.dedent(
    f"""
    import builtins

    _status = {{}}
    try:
        from hermes_cli.models import _PROVIDER_MODELS, CANONICAL_PROVIDERS
        _status["import_ok"] = True
        _status["n_provider_models"] = len(_PROVIDER_MODELS)
    except Exception as exc:
        _status["import_ok"] = False
        _status["error"] = "%s: %s" % (type(exc).__name__, exc)
    builtins._LAZY_CANONICAL_PROBE_STATUS = _status

    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(
        name={PLUGIN_NAME!r},
        display_name="Lazy Canonical Probe",
        description="Lazy Canonical Probe (test fixture)",
        base_url="https://probe.invalid/v1",
        env_vars=({PROBE_KEY_ENV!r},),
        fallback_models={tuple(PROBE_MODELS)!r},
    ))
    """
)

DRIVER = textwrap.dedent(
    f"""
    import builtins, json, sys

    PROBE = {PLUGIN_NAME!r}

    order = sys.argv[1]
    out = {{"order": order}}
    if order == "models-first":
        import hermes_cli.models as models
        import providers
        providers.list_providers()
    else:
        import providers
        providers.list_providers()
        import hermes_cli.models as models

    out["plugin"] = getattr(
        builtins, "_LAZY_CANONICAL_PROBE_STATUS",
        {{"import_ok": None, "error": "plugin was never imported"}},
    )
    out["registry_has_probe"] = any(
        p.name == PROBE for p in __import__("providers").list_providers()
    )
    out["canonical_slugs"] = [p.slug for p in models.CANONICAL_PROVIDERS]
    out["probe_in_canonical"] = PROBE in out["canonical_slugs"]
    out["label"] = models._PROVIDER_LABELS.get(PROBE)
    out["in_known_names"] = PROBE in models._KNOWN_PROVIDER_NAMES

    # Picker + /model label resolution: CALL them, don't just import them.
    from hermes_cli.model_switch import list_picker_providers
    from hermes_cli.models import provider_label

    rows = list_picker_providers(max_models=50)
    by_slug = {{str(r.get("slug", "")).lower(): r for r in rows}}
    probe_row = by_slug.get(PROBE)
    out["picker_slugs"] = sorted(by_slug)
    out["probe_row_name"] = probe_row.get("name") if probe_row else None
    out["probe_row_models"] = (
        list(probe_row.get("models") or []) if probe_row else None
    )

    # /model label resolution: a canonical slug, an ALIAS, the 'custom'
    # special case, and the plugin-registered provider.
    out["probe_label"] = provider_label(PROBE)
    out["nous_label"] = provider_label("nous")
    out["alias_label"] = provider_label("claude")
    out["custom_label"] = provider_label("custom")

    print("PROBE_JSON " + json.dumps(out))
    """
)

# One read per process: whichever conversion is named on argv is the FIRST and
# ONLY read of the lazy container, so a positive result cannot be credited to
# some earlier unrelated read having already triggered the auto-extend.
COPY_DRIVER = textwrap.dedent(
    f"""
    import json, sys

    PROBE = {PLUGIN_NAME!r}
    which = sys.argv[1]

    import hermes_cli.models as models

    C = models.CANONICAL_PROVIDERS
    L = models._PROVIDER_LABELS
    K = models._KNOWN_PROVIDER_NAMES

    if which == "set":
        found = PROBE in set(K)
    elif which == "frozenset":
        found = PROBE in frozenset(K)
    elif which == "set-or":
        found = PROBE in (set() | K)
    elif which == "set-update":
        acc = set()
        acc.update(K)
        found = PROBE in acc
    elif which == "dict":
        found = PROBE in dict(L)
    elif which == "dict-splat":
        found = PROBE in {{**L}}
    elif which == "list":
        found = PROBE in [p.slug for p in list(C)]
    elif which == "list-splat":
        found = PROBE in [p.slug for p in [*C]]
    else:
        raise SystemExit("unknown conversion: " + which)

    print("PROBE_JSON " + json.dumps({{"which": which, "found": found}}))
    """
)


def _run_order(tmp_path: Path, order: str, *, with_plugin: bool = True) -> dict:
    """Run the driver in a fresh interpreter under a temp HERMES_HOME.

    ``with_plugin=False`` gives the baseline: identical environment, no
    plugin installed. Comparing the two is what turns "the picker still
    works" into "the picker is unchanged for every pre-existing provider".
    """
    tag = f"{order}{'' if with_plugin else '-baseline'}"
    home = tmp_path / f"hermes_home_{tag}"
    plugin_root = home / "plugins" / "model-providers"
    plugin_root.mkdir(parents=True)
    if with_plugin:
        plugin_dir = plugin_root / PLUGIN_NAME
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(PLUGIN_INIT)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {PLUGIN_NAME}\nkind: model-provider\nversion: 0.0.1\n"
            "description: lazy-canonical regression fixture\n"
        )

    driver = tmp_path / f"driver_{tag}.py"
    driver.write_text(DRIVER)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Credential for the fixture provider, so the picker emits its row.
    # Set in BOTH arms so the only difference between them is the plugin.
    env[PROBE_KEY_ENV] = "sk-lazy-canonical-probe"

    proc = subprocess.run(
        [sys.executable, str(driver), order],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_JSON ")]
    assert lines, (
        f"driver produced no verdict (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    return json.loads(lines[-1][len("PROBE_JSON "):])


@pytest.mark.parametrize("order", ["models-first", "discovery-first"])
def test_plugin_importing_models_surfaces_does_not_hit_partial_module(tmp_path, order):
    """A provider plugin may import _PROVIDER_MODELS/CANONICAL_PROVIDERS itself."""
    result = _run_order(tmp_path, order)
    plugin = result["plugin"]
    assert plugin["import_ok"] is True, (
        f"[{order}] plugin failed to import models surfaces: {plugin.get('error')}"
    )
    # The bug also emptied _PROVIDER_MODELS for every pin under models-first.
    assert plugin["n_provider_models"] > 0, (
        f"[{order}] plugin saw an EMPTY _PROVIDER_MODELS "
        f"({plugin['n_provider_models']} entries)"
    )


@pytest.mark.parametrize("order", ["models-first", "discovery-first"])
def test_plugin_provider_reaches_canonical_providers(tmp_path, order):
    """Auto-extend still lands the plugin's provider, under either order."""
    result = _run_order(tmp_path, order)
    assert result["registry_has_probe"] is True, (
        f"[{order}] fixture never registered — test is vacuous"
    )
    assert result["probe_in_canonical"] is True, (
        f"[{order}] provider missing from CANONICAL_PROVIDERS "
        f"(tail={result['canonical_slugs'][-5:]})"
    )
    assert result["label"] == "Lazy Canonical Probe", (
        f"[{order}] _PROVIDER_LABELS not extended: {result['label']!r}"
    )
    assert result["in_known_names"] is True, (
        f"[{order}] _KNOWN_PROVIDER_NAMES not extended"
    )


@pytest.mark.parametrize("order", ["models-first", "discovery-first"])
def test_picker_surfaces_the_plugin_provider(tmp_path, order):
    """The picker must actually EMIT a row for the auto-extended provider.

    ``list_picker_providers`` is called for real and its output inspected:
    a plugin provider that reaches ``CANONICAL_PROVIDERS`` but never becomes
    a picker row would satisfy the previous test and still be invisible to
    every ``/model`` picker.
    """
    result = _run_order(tmp_path, order)
    assert PLUGIN_NAME in result["picker_slugs"], (
        f"[{order}] picker emitted no row for the plugin provider; "
        f"rows={result['picker_slugs']}"
    )
    assert result["probe_row_name"] == "Lazy Canonical Probe", (
        f"[{order}] picker row carries the wrong display name: "
        f"{result['probe_row_name']!r}"
    )
    assert result["probe_row_models"] == PROBE_MODELS, (
        f"[{order}] picker row carries the wrong models: "
        f"{result['probe_row_models']!r}"
    )
    # /model label resolution for the plugin provider.
    assert result["probe_label"] == "Lazy Canonical Probe", (
        f"[{order}] provider_label() did not resolve the plugin provider: "
        f"{result['probe_label']!r}"
    )


@pytest.mark.parametrize("order", ["models-first", "discovery-first"])
def test_no_behaviour_change_for_preexisting_providers(tmp_path, order):
    """Measured A/B: installing the plugin must only ADD, never perturb.

    Baseline = identical environment with no plugin installed. The plugin run
    must differ from it by exactly one canonical entry (appended at the tail,
    with every pre-existing slug in its original position) and exactly one
    picker row, and ``/model`` label resolution for canonical slugs, aliases
    and the ``custom`` special case must be byte-identical.
    """
    base = _run_order(tmp_path, order, with_plugin=False)
    withp = _run_order(tmp_path, order, with_plugin=True)

    # Non-vacuity: the baseline must genuinely lack the fixture, otherwise
    # every "unchanged" assertion below is comparing two identical runs.
    assert base["registry_has_probe"] is False, (
        f"[{order}] baseline unexpectedly has the fixture provider registered"
    )
    assert withp["registry_has_probe"] is True, (
        f"[{order}] plugin arm never registered the fixture — A/B is vacuous"
    )

    # Append-only: the baseline list is a strict PREFIX of the extended one.
    # This is the order/identity invariant a count assertion cannot express —
    # it catches reordering and dropped entries, not just a collapsed list.
    n = len(base["canonical_slugs"])
    assert withp["canonical_slugs"][:n] == base["canonical_slugs"], (
        f"[{order}] auto-extend perturbed the canonical list instead of "
        f"appending to it\nbaseline={base['canonical_slugs']}\n"
        f"extended={withp['canonical_slugs'][:n]}"
    )
    assert withp["canonical_slugs"][n:] == [PLUGIN_NAME], (
        f"[{order}] unexpected tail after auto-extend: "
        f"{withp['canonical_slugs'][n:]}"
    )

    # Picker: same rows, plus exactly the plugin's.
    assert (
        sorted(set(withp["picker_slugs"]) - {PLUGIN_NAME})
        == sorted(base["picker_slugs"])
    ), (
        f"[{order}] picker rows changed for pre-existing providers\n"
        f"baseline={base['picker_slugs']}\nwith plugin={withp['picker_slugs']}"
    )

    # /model label resolution for non-plugin inputs is untouched.
    for key, expected in (
        ("nous_label", "Nous Portal"),
        ("alias_label", "Anthropic"),
        ("custom_label", "Custom endpoint"),
    ):
        assert base[key] == expected, (
            f"[{order}] baseline label regressed for {key}: {base[key]!r}"
        )
        assert withp[key] == base[key], (
            f"[{order}] installing a plugin changed {key}: "
            f"{base[key]!r} -> {withp[key]!r}"
        )


def test_models_import_does_not_trigger_provider_discovery(tmp_path):
    """The load-bearing invariant: importing models must not import plugins.

    This is what actually distinguishes the fix from the bug — everything else
    could in principle be satisfied by a different import order.
    """
    home = tmp_path / "hermes_home"
    plugin_dir = home / "plugins" / "model-providers" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    # A plugin that writes a marker file the moment it is imported.
    marker = tmp_path / "plugin_was_imported"
    (plugin_dir / "__init__.py").write_text(
        f"open({str(marker)!r}, 'w').write('imported')\n"
    )
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {PLUGIN_NAME}\nkind: model-provider\nversion: 0.0.1\n"
    )

    driver = tmp_path / "import_only.py"
    driver.write_text("import hermes_cli.models  # noqa: F401\nprint('IMPORTED')\n")

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(driver)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "IMPORTED" in proc.stdout, (
        f"import failed (rc={proc.returncode})\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert not marker.exists(), (
        "importing hermes_cli.models ran provider plugin discovery — the "
        "auto-extend is not lazy"
    )


@pytest.mark.parametrize(
    "which",
    ["set", "frozenset", "set-or", "set-update", "dict", "dict-splat", "list", "list-splat"],
)
def test_copy_constructing_a_lazy_container_still_triggers_the_extend(tmp_path, which):
    """Copying a lazy container must not bypass the auto-extend trigger.

    The lazy surfaces are consumed by copy-construction in real call sites
    (``dict(_PROVIDER_LABELS)`` in ``hermes_cli/main.py``, ``set(...)`` /
    ``| ...`` unions elsewhere). CPython's ``set_update_internal`` takes a
    ``PyAnySet_Check`` fast path that reads a real set's hash table directly
    and NEVER calls ``__iter__`` — so a ``set`` SUBCLASS with a lazy
    ``__iter__`` silently yields a copy missing every plugin provider, while
    a direct ``in`` test on the same object answers correctly. That divergence
    is exactly the bug class this file exists for, re-entering through the
    copy path, and it is invisible to every other test here.

    One conversion per subprocess, so the read under test is the first read.
    """
    home = tmp_path / f"hermes_home_{which}"
    plugin_dir = home / "plugins" / "model-providers" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(PLUGIN_INIT)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {PLUGIN_NAME}\nkind: model-provider\nversion: 0.0.1\n"
    )

    driver = tmp_path / f"copy_{which}.py"
    driver.write_text(COPY_DRIVER)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env[PROBE_KEY_ENV] = "sk-lazy-canonical-probe"

    proc = subprocess.run(
        [sys.executable, str(driver), which],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_JSON ")]
    assert lines, (
        f"driver produced no verdict (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    result = json.loads(lines[-1][len("PROBE_JSON "):])
    assert result["found"] is True, (
        f"{which}(...) of a lazy container did not trigger the plugin "
        f"auto-extend; the copy is missing {PLUGIN_NAME}"
    )
