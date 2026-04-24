from types import SimpleNamespace

from hermes_cli.memory_setup import _import_name_for_dependency, _install_dependencies


def test_import_name_for_dependency_handles_versioned_requirements():
    assert _import_name_for_dependency("hindsight-client>=0.4.22") == "hindsight_client"
    assert _import_name_for_dependency("hermes-membase>=0.1.5") == "membase_hermes"


def test_import_name_for_dependency_handles_extras_and_plain_names():
    assert _import_name_for_dependency("some-package[extra]>=1.2") == "some_package"
    assert _import_name_for_dependency("supermemory") == "supermemory"
    assert _import_name_for_dependency("marker-package; python_version >= '3.11'") == "marker_package"


def test_install_dependencies_ignores_repo_uv_config(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "membase"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        "pip_dependencies:\n"
        "  - example-missing-package>=1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("plugins.memory.find_provider_dir", lambda name: plugin_dir)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("subprocess.run", fake_run)

    _install_dependencies("membase")

    cmd, kwargs = calls[0]
    assert cmd[:4] == ["/usr/bin/uv", "pip", "install", "--no-config"]
    assert "--python" in cmd
    assert "example-missing-package>=1" in cmd
    assert kwargs["check"] is True
