"""Config-selected plugin names (``memory.provider``, ``cron.provider``, ``context.engine``) stay
inside their plugin roots, and a directory outside the bundled root is never classified bundled
(which would import it in-process even under ``plugins.isolation: host``) (#141328)."""

import os
from pathlib import Path

import pytest

import plugins.context_engine as context_engine
import plugins.cron_providers as cron_providers
import plugins.memory as memory

_KINDS = [
    (memory, "find_provider_dir", "_MEMORY_PLUGINS_DIR", "MemoryProvider\ndef register(ctx): pass\n"),
    (cron_providers, "find_provider_dir", "_CRON_PLUGINS_DIR", "register_cron_scheduler\n"),
    (context_engine, "find_engine_dir", "_CONTEXT_ENGINE_PLUGINS_DIR", "register_context_engine\n"),
]


def _outside_plugin(tmp_path: Path, marker_body: str) -> Path:
    target = tmp_path / "outside" / "evil"
    target.mkdir(parents=True)
    ran = tmp_path / "ran"
    (target / "__init__.py").write_text(f"# {marker_body}\nopen({str(ran)!r}, 'w').close()\n")
    return target


def _lexical_escape(root: Path, target: Path) -> str:
    try:
        return os.path.relpath(target, root)
    except ValueError:  # different drives on Windows: no relative spelling exists
        pytest.skip("no relative path between the plugin root and tmp_path")


@pytest.mark.parametrize("module,finder,root_attr,body", _KINDS, ids=["memory", "cron", "context"])
def test_names_that_are_paths_are_not_found(tmp_path, monkeypatch, module, finder, root_attr, body):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "plugins").mkdir(parents=True)
    target = _outside_plugin(tmp_path, body)
    root = getattr(module, root_attr)
    find = getattr(module, finder)
    for name in (_lexical_escape(root, target), str(target),
                 _lexical_escape(tmp_path / "home" / "plugins", target)):
        assert find(name) is None, name


class _RecordingHost:
    def __init__(self):
        self.calls = []

    def __getattr__(self, attr):
        def _call(*args, **kwargs):
            self.calls.append(attr)
            raise RuntimeError("routed to plugin host")
        return _call


@pytest.mark.parametrize("module,loader,root_attr,body", [
    (memory, "_load_provider_from_dir", "_MEMORY_PLUGINS_DIR", "MemoryProvider\n"),
    (cron_providers, "_load_provider_from_dir", "_CRON_PLUGINS_DIR", "register_cron_scheduler\n"),
    (context_engine, "_load_engine_from_dir", "_CONTEXT_ENGINE_PLUGINS_DIR", "register_context_engine\n"),
], ids=["memory", "cron", "context"])
def test_escaped_dir_is_not_treated_as_bundled(tmp_path, monkeypatch, module, loader, root_attr, body):
    """A ``<root>/../..`` spelling lists the bundled root among its lexical parents; it must still
    go to the plugin host, never be imported into this process."""
    target = _outside_plugin(tmp_path, body)
    root = getattr(module, root_attr)
    escaped = root / _lexical_escape(root, target)
    host = _RecordingHost()
    monkeypatch.setattr("hermes_cli.plugin_isolation.user_plugin_host", lambda: host)

    with pytest.raises(RuntimeError, match="routed to plugin host"):
        getattr(module, loader)(escaped)

    assert host.calls
    assert not (tmp_path / "ran").exists()


@pytest.mark.platforms("posix")
def test_user_dir_symlink_into_the_bundled_root_is_not_bundled(tmp_path):
    """A ``$HERMES_HOME/plugins/<x>`` symlink into a bundled dir resolves under the bundled root, but
    is not bundled: it would load in-process past isolation, and the link can be swapped later."""
    from plugins import plugin_loader
    root = tmp_path / "bundled"
    bundled = root / "real"
    bundled.mkdir(parents=True)
    link = tmp_path / "plugins" / "alias"
    link.parent.mkdir()
    link.symlink_to(bundled, target_is_directory=True)
    assert plugin_loader.is_direct_child(bundled, root)
    assert not plugin_loader.is_direct_child(link, root)
