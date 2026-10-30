from __future__ import annotations

from pathlib import Path


def test_doctor_uses_registration_to_reject_bad_hook_and_callback_signature(
    tmp_path: Path,
) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "bad-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        "\n".join(
            [
                "name: bad-plugin",
                "version: 0.1.0",
                "description: broken contract",
                "provides_hooks:",
                "  - typo_hook",
                "  - pre_tool_call",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def callback(tool_name):\n"
        "    return None\n\n"
        "def register(ctx):\n"
        "    ctx.register_hook('typo_hook', callback)\n"
        "    ctx.register_hook('pre_tool_call', callback)\n",
        encoding="utf-8",
    )

    report = doctor_plugin(plugin)
    messages = "\n".join(f.message for f in report.findings)
    assert report.ok is False
    assert "unknown hook 'typo_hook'" in messages
    assert "must accept **kwargs" in messages


def test_doctor_accepts_manifest_defaults_from_runtime_parser(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "minimal"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: minimal\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n    pass\n", encoding="utf-8"
    )

    report = doctor_plugin(plugin)
    assert report.ok, report.format_text()
    assert report.manifest is not None
    assert report.manifest.kind == "standalone"


def test_doctor_restores_global_tool_policy_and_module_state(tmp_path: Path) -> None:
    import sys

    from hermes_cli.plugin_dev import doctor_plugin
    from tools.registry import registry

    target = tmp_path / "cleanup-plugin"
    target.mkdir()
    (target / "plugin.yaml").write_text(
        "name: cleanup-plugin\nprovides_tools: [cleanup_plugin_ping]\n",
        encoding="utf-8",
    )
    (target / "__init__.py").write_text(
        "import json\n\n"
        "def ping(args, **kwargs):\n    return json.dumps({'ok': True})\n\n"
        "def register(ctx):\n"
        "    ctx.register_tool(name='cleanup_plugin_ping', toolset='cleanup', "
        "schema={'name': 'cleanup_plugin_ping', 'description': 'test', "
        "'parameters': {'type': 'object'}}, handler=ping)\n",
        encoding="utf-8",
    )
    before_policy = dict(registry._plugin_override_policy)
    before_modules = {
        name
        for name in sys.modules
        if name == "hermes_plugins" or name.startswith("hermes_plugins.")
    }

    report = doctor_plugin(target)

    assert report.ok, report.format_text()
    assert report.registered_tools == ("cleanup_plugin_ping",)
    assert registry.get_entry("cleanup_plugin_ping") is None
    assert registry._plugin_override_policy == before_policy
    after_modules = {
        name
        for name in sys.modules
        if name == "hermes_plugins" or name.startswith("hermes_plugins.")
    }
    assert after_modules == before_modules


def test_doctor_blocks_live_network(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "network-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: network-plugin\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "import socket\n\n"
        "def register(ctx):\n"
        "    socket.create_connection(('example.com', 443))\n",
        encoding="utf-8",
    )

    report = doctor_plugin(plugin)
    assert report.ok is False
    assert "network access is disabled while Plugin Doctor runs" in report.format_text()


def test_doctor_default_target_does_not_copy_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """``hermes plugins doctor`` with no argument defaults to ``.``.

    Before the manifest guard, that copied the whole working directory into
    a temporary HERMES_HOME — running it from ``$HOME`` copied the home
    directory, cloud-storage placeholders included.
    """
    import os
    import shutil

    from hermes_cli import plugin_dev

    workdir = tmp_path / "workdir"
    (workdir / "big").mkdir(parents=True)
    (workdir / "big" / "payload.bin").write_text("x" * 1024, encoding="utf-8")
    monkeypatch.chdir(workdir)

    copied: list[tuple[str, str]] = []
    real_copytree = shutil.copytree

    def _tracking_copytree(src, dst, *args, **kwargs):
        copied.append((os.fspath(src), os.fspath(dst)))
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(plugin_dev.shutil, "copytree", _tracking_copytree)

    report = plugin_dev.doctor_plugin()

    assert report.ok is False
    assert copied == []


def test_doctor_does_not_copy_home_with_a_plugin_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    """A home with one plugin checkout next to other directories is not a category.

    ``_holds_plugin`` used to accept any immediate child manifest, so
    ``hermes plugins doctor`` from ``$HOME`` resolved to the home and
    ``copytree`` copied it. A category directory's children are plugins.
    """
    import os

    from hermes_cli import plugin_dev

    home = tmp_path / "fake-home"
    (home / ".hermes" / "cache" / "scratch").mkdir(parents=True)
    plugin = home / "plugin-checkout"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: checkout\n", encoding="utf-8")
    monkeypatch.chdir(home)

    copied: list[str] = []

    def _stub(src, dst, *args, **kwargs):
        copied.append(os.fspath(src))
        raise RuntimeError("copytree must not run")

    monkeypatch.setattr(plugin_dev.shutil, "copytree", _stub)

    assert plugin_dev.resolve_plugin_path(plugin) == plugin.resolve()
    report = plugin_dev.doctor_plugin(".")
    assert copied == []
    assert report.ok is False
    assert "holds no plugin manifest" in report.format_text()


def test_resolve_accepts_category_layout(tmp_path: Path) -> None:
    """A category directory holds no manifest itself but discovery finds one."""
    from hermes_cli.plugin_dev import resolve_plugin_path

    category = tmp_path / "image_gen"
    plugin = category / "openai"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: openai\n", encoding="utf-8")

    assert resolve_plugin_path(category) == category.resolve()


def test_doctor_stages_only_the_plugin_of_a_category(tmp_path: Path, monkeypatch) -> None:
    """A category may hold loose files and dirs the staging copy drops or discovery never scans;
    none of them is copied."""
    import os
    import shutil

    from hermes_cli import plugin_dev

    category = tmp_path / "tools"
    plugin = category / "minimal"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: minimal\n", encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (category / "disk-image.iso").write_bytes(b"x" * 1024)
    for skipped in ("__data__", ".git", ".claude-plugin"):
        (category / skipped).mkdir()

    copied: list[str] = []
    real_copytree = shutil.copytree

    def _tracking_copytree(src, dst, *args, **kwargs):
        copied.append(os.fspath(src))
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(plugin_dev.shutil, "copytree", _tracking_copytree)

    report = plugin_dev.doctor_plugin(category)
    assert report.ok, report.format_text()
    assert copied == [os.fspath(plugin.resolve())]
    assert report.manifest is not None
    assert report.manifest.key == "tools/minimal"  # staged under the category, as discovery keys it


def test_doctor_rejects_category_of_several_plugins_before_copying(
    tmp_path: Path, monkeypatch
) -> None:
    """Doctor validates one plugin, so copying a whole category first only wastes disk."""
    from hermes_cli import plugin_dev

    category = tmp_path / "web"
    for name in ("a", "b"):
        (category / name).mkdir(parents=True)
        (category / name / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")

    copied: list[object] = []
    monkeypatch.setattr(plugin_dev.shutil, "copytree", lambda src, *a, **k: copied.append(src))

    report = plugin_dev.doctor_plugin(category)
    assert copied == []
    assert report.ok is False
    assert "holds 2 plugins (a, b)" in report.format_text()


def test_doctor_names_plugin_dirs_of_a_target_that_is_not_a_category(
    tmp_path: Path, monkeypatch
) -> None:
    """The refusal names the plugin inside, so the user knows what to pass instead."""
    import os

    from hermes_cli import plugin_dev

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    for name, manifest in (("myplugin", "plugin.yaml"), (".claude-plugin", "plugin.json")):
        (repo / name).mkdir()
        (repo / name / manifest).write_text("name: x\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    report = plugin_dev.doctor_plugin(".")
    assert report.ok is False
    # Discovery skips .claude-plugin, so suggesting it would send the user to a dead end.
    assert f"plugin directory inside it: {os.path.join('.', 'myplugin')}." in report.format_text()


def test_resolve_prefers_installed_id_over_unrelated_local_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """A same-named local directory must not shadow the installed plugin."""
    from hermes_cli import plugin_dev

    hermes_home = tmp_path / "hermes-home"
    installed = hermes_home / "plugins" / "sample"
    installed.mkdir(parents=True)
    (installed / "plugin.yaml").write_text("name: sample\n", encoding="utf-8")

    workdir = tmp_path / "workdir"
    (workdir / "sample").mkdir(parents=True)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(plugin_dev, "get_hermes_home", lambda: hermes_home)

    assert plugin_dev.resolve_plugin_path("sample") == installed.resolve()


def test_doctor_removes_temp_home_when_staging_copy_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """A copy failure (e.g. ENOSPC) must not strand a hermes-plugin-doctor-* dir."""
    import errno
    import shutil
    import tempfile


    from hermes_cli import plugin_dev

    plugin = tmp_path / "sample"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: sample\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n    pass\n", encoding="utf-8"
    )

    scratch = tmp_path / "scratch-tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def _enospc(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copytree", _enospc)

    caught: OSError | None = None
    try:
        with plugin_dev._doctor_runtime(plugin):
            pass  # pragma: no cover - staging fails before yield
    except OSError as exc:
        caught = exc

    assert caught is not None and caught.errno == errno.ENOSPC
    # Check for leftovers WHILE the exception (and its traceback frames) is
    # still referenced: the old code relied on TemporaryDirectory's GC
    # finalizer, which cannot run while the traceback pins the frame — the
    # exact window where a stranded hermes-plugin-doctor-* dir was observed.
    leftovers = list(scratch.glob("hermes-plugin-doctor-*"))
    assert leftovers == [], f"stranded doctor temp dirs: {leftovers}"


def test_doctor_loads_model_provider_plugins_through_provider_discovery(tmp_path: Path) -> None:
    """`kind: model-provider` registers a ProviderProfile at import and has no register(ctx);
    doctor must judge it by that contract and leave the live registry untouched."""
    import providers
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "acme-provider"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: acme-provider\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\nfrom providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='acme-doctor-probe', auth_type='external_process',\n"
        "                                  process_command='acme', aliases=('acme-alias',)))\n",
        encoding="utf-8")
    registry_before = dict(providers._REGISTRY)

    report = doctor_plugin(plugin)
    assert report.ok, report.format_text()
    assert report.registered_providers == ("acme-doctor-probe",)
    assert "provider(s): acme-doctor-probe" in report.format_text()
    assert providers._REGISTRY == registry_before
    assert "acme-alias" not in providers._ALIASES

    (plugin / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    report = doctor_plugin(plugin)
    assert not report.ok
    assert any("registered no ProviderProfile" in f.message for f in report.findings)


def test_doctor_loads_a_model_provider_plugin_from_its_category(tmp_path: Path) -> None:
    """``plugins/model-providers/<name>`` is the runtime layout; the provider loader must get the
    staged plugin directory, not the category around it."""
    import providers
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "model-providers" / "acme-provider"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: acme-provider\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\nfrom providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='acme-category-probe', auth_type='external_process',\n"
        "                                  process_command='acme'))\n",
        encoding="utf-8")

    registry_before = dict(providers._REGISTRY)

    report = doctor_plugin(plugin.parent)
    assert report.ok, report.format_text()
    assert report.registered_providers == ("acme-category-probe",)
    assert providers._REGISTRY == registry_before
