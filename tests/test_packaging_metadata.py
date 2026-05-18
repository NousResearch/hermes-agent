from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_faster_whisper_is_not_a_base_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert not any(dep.startswith("faster-whisper") for dep in deps)

    voice_extra = data["project"]["optional-dependencies"]["voice"]
    assert any(dep.startswith("faster-whisper") for dep in voice_extra)


def test_manifest_includes_bundled_skills():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft skills" in manifest
    assert "graft optional-skills" in manifest


def test_pyproject_ships_plugin_yaml_manifests():
    """Regression test for #28149.

    Plugin discovery (``hermes_cli/plugins.py::_scan_directory_level``) reads
    a ``plugin.yaml`` manifest from each bundled plugin directory. Wheels
    only carry files declared in ``[tool.setuptools.package-data]``, so an
    entry for the ``plugins`` package is required, otherwise the installed
    wheel ships every plugin's Python code but none of its manifests and
    ``hermes web_search`` fails with "No web search provider configured".
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]

    assert "plugins" in package_data, (
        "[tool.setuptools.package-data] must declare a 'plugins' entry so "
        "plugin.yaml manifests ship in the wheel (#28149)."
    )
    patterns = package_data["plugins"]
    joined = " ".join(patterns)
    assert "plugin.yaml" in joined, (
        "plugins package-data must include 'plugin.yaml' so wheels carry "
        "the manifests the PluginManager scans for (#28149)."
    )
    assert "plugin.yml" in joined, (
        "plugins package-data must also include 'plugin.yml'. "
        "_scan_directory_level accepts both suffixes, and dropping the "
        "legacy one would silently lose any plugin that uses it."
    )


def test_manifest_includes_bundled_plugin_manifests():
    """Regression test for #28149.

    ``MANIFEST.in`` controls the sdist (downstream packagers such as
    Homebrew build from sdist). It must list the ``plugin.yaml`` files so
    they survive the sdist trip, mirroring the wheel-side package-data
    entry. ``recursive-include plugins plugin.yaml plugin.yml`` is the
    minimal pattern that satisfies both ``.yaml`` and the legacy ``.yml``
    suffix accepted by ``_scan_directory_level``.
    """
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "plugins" in manifest, (
        "MANIFEST.in must reference the plugins tree so plugin manifests "
        "survive the sdist trip (#28149)."
    )
    assert "plugin.yaml" in manifest, (
        "MANIFEST.in must include 'plugin.yaml' so the sdist (and "
        "downstream Homebrew bottles) ship plugin manifests (#28149)."
    )
    assert "plugin.yml" in manifest, (
        "MANIFEST.in must also include 'plugin.yml'. _scan_directory_level "
        "accepts both suffixes, and dropping the legacy one would silently "
        "lose any plugin that uses it from the sdist."
    )


def test_bundled_web_search_plugin_manifests_resolve_via_importlib():
    """Regression test for #28149.

    Each bundled web-search provider must expose its ``plugin.yaml`` via
    the ``importlib.resources`` API so ``PluginManager._scan_directory()``
    can find it. This probe resolves through the ``plugins.web`` package
    only (avoiding per-provider package imports), which keeps the test
    cheap while still asserting the manifests are discoverable through
    the resources API the loader uses.
    """
    from importlib.resources import files

    expected_providers = (
        "firecrawl",
        "tavily",
        "exa",
        "parallel",
        "ddgs",
        "searxng",
        "brave_free",
    )
    web_root = files("plugins.web")
    for provider in expected_providers:
        manifest = web_root.joinpath(provider, "plugin.yaml")
        assert manifest.is_file(), (
            f"plugins/web/{provider}/plugin.yaml is not accessible via "
            "importlib.resources; rebuild the wheel after restoring the "
            "'plugins' entry in [tool.setuptools.package-data] (#28149)."
        )
