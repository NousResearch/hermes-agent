"""Retained provider packages outlive temp cleanup, updates, and other processes."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_OLD = 946684800
_CUTOFF = _OLD + 86400 * 30
_CHILD = textwrap.dedent("""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, sys.argv[1])
    import providers
    from hermes_constants import get_process_hermes_home
    from providers import get_provider_profile
    from providers import plugin_reload

    before = get_provider_profile("snapshot-demo")
    assert before is not None

    def location(profile):
        module = sys.modules[type(profile).__module__]
        return {
            "package": str(Path(module.__file__).parent),
            "lazy_loaded": module.__name__ + ".lazy" in sys.modules,
        }

    def observe(profile, access):
        try:
            return {"value": profile.create_client(access=access), "error": None}
        except (ImportError, OSError, subprocess.CalledProcessError) as exc:
            return {"value": None, "error": type(exc).__name__ + ": " + str(exc)}

    print(json.dumps({
        "root": str(plugin_reload._snapshot_root),
        "home": str(get_process_hermes_home()),
        "temp": tempfile.gettempdir(),
        "loader": plugin_reload.__file__,
        "isolated": sys.flags.isolated,
        "no_bytecode": sys.dont_write_bytecode,
        **location(before),
    }), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        if command["op"] == "update":
            from hermes_cli.plugins import PluginManager
            PluginManager().discover_and_load(force=True)
            after = get_provider_profile("snapshot-demo")
            result = location(after)
        else:
            profile = before if command["version"] == "old" else after
            result = observe(profile, command["access"])
        print(json.dumps(result), flush=True)
""")


def _write_release(plugin: Path, version: int) -> None:
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(
        f"name: snapshot-demo\nkind: model-provider\nversion: {version}.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(textwrap.dedent("""
        import subprocess
        import sys
        from pathlib import Path
        from providers import register_provider
        from providers.base import ProviderProfile

        class Profile(ProviderProfile):
            def create_client(self, *, access):
                if access == "lazy":
                    from .lazy import value
                    return value
                return subprocess.check_output(
                    [sys.executable, "-I", "-B", str(Path(__file__).with_name("helper.py"))],
                    text=True, timeout=10).strip()

        register_provider(Profile(name="snapshot-demo", base_url="process://snapshot-demo",
                                  auth_type="external_process"))
    """), encoding="utf-8")
    (plugin / "lazy.py").write_text(f"value = 'v{version}'\n", encoding="utf-8")
    (plugin / "helper.py").write_text(f"print('v{version}')\n", encoding="utf-8")
    for path in plugin.iterdir():
        os.utime(path, (_OLD + version, _OLD + version))


@pytest.fixture
def installation(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  isolation: in-process\n", encoding="utf-8")
    plugin = home / "plugins" / "snapshot-demo"
    _write_release(plugin, 1)
    os_temp = tmp_path / "os-temp"
    os_temp.mkdir()
    return home, plugin, os_temp


@contextmanager
def _provider_process(home: Path, os_temp: Path, stderr_path: Path):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "HERMES_HOME": str(home),
        "HERMES_RUNTIME_DIR": str(home / "tools"),
        "TMPDIR": str(os_temp), "TMP": str(os_temp), "TEMP": str(os_temp),
        "LANG": "C.UTF-8", "TZ": "UTC",
    }
    for key in ("SYSTEMROOT", "USERPROFILE", "LOCALAPPDATA"):
        if key in os.environ:
            env[key] = str(home) if key != "SYSTEMROOT" else os.environ[key]
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", _CHILD, str(_REPO)],
            cwd=_REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True,
        )
        lines = queue.Queue()

        def read_lines():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()

        def receive(command=None):
            if command is not None:
                process.stdin.write(json.dumps(command) + "\n")
                process.stdin.flush()
            line = lines.get(timeout=30)
            assert line is not None, stderr_path.read_text(encoding="utf-8")
            return json.loads(line)

        try:
            ready = receive()
            assert Path(ready["home"]).resolve() == home.resolve()
            assert Path(ready["temp"]).resolve() == os_temp.resolve()
            assert Path(ready["loader"]).resolve() == _REPO / "providers" / "plugin_reload.py"
            assert ready["isolated"] == 1 and ready["no_bytecode"]
            yield ready, receive
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                raise
            finally:
                reader.join(timeout=10)
                process.stdout.close()
            assert process.returncode == 0, stderr_path.read_text(encoding="utf-8")


def _age_cleanup(os_temp: Path) -> list[str]:
    removed = []
    for path in os_temp.rglob("*"):
        if path.is_file() and path.stat().st_mtime < _CUTOFF:
            removed.append(str(path.relative_to(os_temp)))
            path.unlink()
    return sorted(removed)


@pytest.mark.parametrize("access", ["lazy", "helper"])
@pytest.mark.parametrize("cleanup", [False, True], ids=["healthy", "aged-temp"])
def test_retained_release_survives_temp_cleanup_and_update(installation, tmp_path, record_property, access, cleanup):
    home, plugin, os_temp = installation
    stale = os_temp / "stale-control"
    stale.write_text("old", encoding="utf-8")
    os.utime(stale, (_OLD, _OLD))
    fresh = os_temp / "fresh-control"
    fresh.write_text("new", encoding="utf-8")
    with _provider_process(home, os_temp, tmp_path / "child.stderr") as (ready, send):
        assert not ready["lazy_loaded"]
        assert (Path(ready["package"]) / "lazy.py").stat().st_mtime < _CUTOFF
        _write_release(plugin, 2)
        updated = send({"op": "update"})
        assert updated["package"] != ready["package"] and not updated["lazy_loaded"]
        removed = _age_cleanup(os_temp) if cleanup else []
        old = send({"op": "read", "version": "old", "access": access})
        new = send({"op": "read", "version": "new", "access": access})
        observation = {"ready": ready, "updated": updated, "removed": removed, "old": old, "new": new}
        record_property("observation", json.dumps(observation, sort_keys=True))
        assert fresh.is_file()
        assert stale.exists() is not cleanup
        assert old == {"value": "v1", "error": None}, observation
        assert new == {"value": "v2", "error": None}, observation
    assert not Path(ready["root"]).exists()


def test_process_exit_removes_only_its_own_snapshot(installation, tmp_path, record_property):
    home, _plugin, os_temp = installation
    with _provider_process(home, os_temp, tmp_path / "second.stderr") as (second, read_second):
        with _provider_process(home, os_temp, tmp_path / "first.stderr") as (first, read_first):
            assert first["root"] != second["root"]
            assert first["package"] != second["package"]
            assert read_first({"op": "read", "version": "old", "access": "helper"}) == {
                "value": "v1", "error": None}
        assert not Path(first["root"]).exists()
        assert Path(second["package"]).is_dir()
        lazy = read_second({"op": "read", "version": "old", "access": "lazy"})
        helper = read_second({"op": "read", "version": "old", "access": "helper"})
        observation = {"first": first, "second": second, "lazy": lazy, "helper": helper}
        record_property("observation", json.dumps(observation, sort_keys=True))
        assert lazy == helper == {"value": "v1", "error": None}, observation
    assert not Path(second["root"]).exists()
