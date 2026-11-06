"""Every bundled platform module executes under one name: its canonical ``plugins.platforms.<x>.<m>``.

The loader executes a bundled platform package under a synthetic ``hermes_plugins.platforms__<name>``
name. If that package loaded its adapter relatively (``from .adapter import register``), the adapter
would run as ``hermes_plugins.platforms__<name>.adapter`` and, once anything imported
``plugins.platforms.<name>.*`` (the adapter's own absolute sibling imports, core code, other plugins), a
second time: two adapter classes (the registry's and the one the rest of the code sees: Yuanbao's
``get_active()`` singleton, isinstance checks, monkeypatches) and the module's code and objects in
memory twice.
"""

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.config import Platform

_SYNTHETIC = "hermes_plugins.platforms__"
# The loader itself imports ``<package>.tools`` for plugins that declare deferred client tools.
_LOADER_SUBMODULES = {"tools"}
_REPO = Path(__file__).resolve().parents[2]
_BUNDLED = sorted(Platform._scan_bundled_plugin_platforms()[0])


# Declared optional dependencies whose import root differs from the normalized distribution name.
_DIST_IMPORT_ROOTS = {"python_telegram_bot": "telegram", "discord_py": "discord", "google_cloud_pubsub": "google",
                      "google_api_python_client": "googleapiclient", "google_auth": "google",
                      "microsoft_teams_apps": "microsoft_teams", "python_docx": "docx", "python_pptx": "pptx",
                      "python_multipart": "multipart", "firecrawl_py": "firecrawl", "azure_identity": "azure",
                      "agent_client_protocol": "acp", "piper_tts": "piper", "parallel_web": "parallel",
                      "opentelemetry_sdk": "opentelemetry", "opentelemetry_exporter_otlp_proto_http": "opentelemetry"}


def _optional_dependency_roots() -> set[str]:
    """Import roots of the project's declared optional dependencies (pyproject extras)."""
    import tomllib

    extras = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"].get(
        "optional-dependencies", {})
    roots = set()
    for requirements in extras.values():
        for req in requirements:
            dist = re.split(r"[\s\[<>=!~;@]", req.strip(), maxsplit=1)[0].lower().replace("-", "_").replace(".", "_")
            if dist and dist != "hermes_agent":
                roots.add(_DIST_IMPORT_ROOTS.get(dist, dist))
    return roots


def _repo_owned(root: str) -> bool:
    """A top-level module is this repo's own when it resolves inside the checkout (outside its venv), or when
    it does not resolve at all and is no declared optional dependency (a renamed/deleted repo module such as
    ``utils``, ``run_agent``, ``hermes_state_*``). An ImportError naming one is a real bug, never a missing SDK;
    an undeclared absent root fails loudly rather than shrinking coverage."""
    try:
        spec = importlib.util.find_spec(root)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return root not in _optional_dependency_roots()
    origin = spec.origin or next(iter(spec.submodule_search_locations or ()), None)
    path = Path(origin).resolve() if origin and origin not in ("built-in", "frozen") else None
    return bool(path and path.is_relative_to(_REPO)
                and not {".venv", "venv", "site-packages"} & set(path.relative_to(_REPO).parts))


def _missing_sdk(exc_type: str, missing: str | None) -> bool:
    # Only the ROOT decides: a missing submodule inside an installed SDK (``telegram.ext``, an SDK too
    # old for a feature) counts as the optional SDK missing and is skipped deliberately.
    return exc_type == "ModuleNotFoundError" and bool(missing) and not _repo_owned(missing.split(".")[0])


@pytest.mark.parametrize("platform", _BUNDLED)
def test_registry_factory_comes_from_the_canonical_module(platform):
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    entry = platform_registry.get(platform)
    if entry is None:
        try:
            importlib.import_module(f"plugins.platforms.{platform}")
        except ImportError as exc:
            assert _missing_sdk(type(exc).__name__, exc.name), f"{platform}: plugin failed to load: {exc!r}"
            pytest.skip(f"{platform}: optional SDK {exc.name!r} not installed")
        pytest.fail(f"{platform}: package imports, but discover_plugins() registered no entry")
    assert entry is not None

    factory = entry.adapter_factory
    module = factory.__module__
    # A factory defined in the package body itself (a2a's lambda) belongs to the loader-executed __init__;
    # anything deeper must be the canonical module, and the very object that module holds.
    if module != f"{_SYNTHETIC}{platform}":
        assert module.startswith(f"plugins.platforms.{platform}."), module
        if isinstance(factory, type):
            assert getattr(sys.modules[module], factory.__name__) is factory, module
    synthetic = f"{_SYNTHETIC}{platform}."
    extra = sorted(name for name in sys.modules
                   if name.startswith(synthetic) and name[len(synthetic):].split(".")[0] not in _LOADER_SUBMODULES)
    assert not extra, f"{platform}: modules executed under the loader's synthetic name: {extra}"


# Runs in a fresh interpreter: any earlier canonical import in the pytest process would mask (or fake) a duplicate.
_PROBE = r"""
import importlib, json, sys
from hermes_cli.plugins import discover_plugins
from gateway.config import Platform
from gateway.platform_registry import platform_registry

discover_plugins()               # the real loader path: bundled platforms load as hermes_plugins.platforms__<x>
platform_registry.all_entries()  # materialize every deferred adapter
import gateway.run               # noqa: F401  (core modules the gateway pulls in eagerly)

registered = platform_registry.registered_names()
failed = {}
for platform in sorted(Platform._scan_bundled_plugin_platforms()[0]):
    if platform in registered:
        continue
    try:
        importlib.import_module(f"plugins.platforms.{platform}")
    except ImportError as exc:
        failed[platform] = [type(exc).__name__, exc.name, str(exc)]
    else:
        failed[platform] = ["NotRegistered", None, "package imports but registered no entry"]

prefix = "hermes_plugins.platforms__"
synthetic = sorted(n for n in sys.modules if n.startswith(prefix) and "." in n[len(prefix):])
both = [n for n in synthetic if "plugins.platforms.{}.{}".format(*n[len(prefix):].split(".", 1)) in sys.modules]
print(json.dumps({"registered": sorted(registered), "failed": failed, "synthetic": synthetic, "both": both}))
"""


def test_no_platform_module_loaded_under_two_names():
    """After the real loader path (discovery, every deferred adapter materialized, the gateway imported),
    the only bundled platform submodules under the loader's synthetic name are the ones the loader imports
    itself (``tools``), and no module exists both as ``plugins.platforms.<x>.<m>`` and
    ``hermes_plugins.platforms__<x>.<m>``. A plugin that failed to load must be explained by a missing
    optional SDK, so coverage cannot shrink silently."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(_REPO), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", _PROBE], cwd=_REPO, env=env, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])

    genuine = {p: f for p, f in report["failed"].items() if not _missing_sdk(f[0], f[1])}
    assert not genuine, f"bundled platforms failed to load for reasons other than a missing SDK: {genuine}"
    assert set(_BUNDLED) - set(report["failed"]) <= set(report["registered"])

    unexpected = [n for n in report["synthetic"] if n[len(_SYNTHETIC):].partition(".")[2] not in _LOADER_SUBMODULES]
    assert not unexpected, f"modules executed under the loader's synthetic name: {unexpected}"
    assert not report["both"], f"modules executed under two names: {report['both']}"
