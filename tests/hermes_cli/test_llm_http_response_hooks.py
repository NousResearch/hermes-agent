"""Early HTTP observations use normal plugin discovery, profile and unload ownership."""

from types import SimpleNamespace

import hermes_yaml as yaml
import httpx
import pytest

from agent.served_model import install_llm_response_hook
from hermes_cli import plugins


_PLUGIN = '''
SEEN = []

def fail_on_created(status_code):
    if status_code == 201:
        raise RuntimeError("observer failed")

def observe(status_code, get_header):
    SEEN.append((status_code, [get_header(name) for name in (
        "x-request-id", "missing", "duplicate", "oversized", "x" * 129, "non-ascii-\\u2603")]))
    return {"action": "block"}

def register(ctx):
    ctx.register_hook("llm_http_response", fail_on_created)
    ctx.register_hook("llm_http_response", observe)
'''


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    directory = home / "plugins" / "observer"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("name: observer\nversion: '1.0'\n", encoding="utf-8")
    (directory / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["observer"]}}), encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(plugins, "_plugin_manager", None)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    manager = plugins.get_plugin_manager()
    yield home, manager
    manager.unload()


def _client(status=200):
    return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status, headers=[
        ("x-request-id", "receipt"), ("duplicate", "a"), ("duplicate", "b"),
        ("oversized", "v" * 4097),
    ], stream=httpx.ByteStream(b"unchanged"))))


def test_response_observer_is_early_readonly_and_fail_open(plugin_home):
    _, manager = plugin_home
    with _client(201) as client:
        sdk = SimpleNamespace(_client=client)
        assert install_llm_response_hook(sdk) is sdk
        install_llm_response_hook(sdk)
        with client.stream("GET", "https://example.test") as response:
            assert not response.is_stream_consumed
            module = manager._plugins["observer"].module
            assert module.SEEN == [(201, ["receipt", None, None, None, None, None])]
            assert response.status_code == 201  # callback's block directive is not policy
            assert response.read() == b"unchanged"


def test_cached_client_uses_current_profile_and_registration(plugin_home, tmp_path):
    _, manager = plugin_home
    with _client() as client:
        install_llm_response_hook(SimpleNamespace(_client=client))
        client.get("https://example.test")
        previous = manager._plugins["observer"].module.SEEN
        assert len(previous) == 1
        other = tmp_path / "other"
        other.mkdir()
        with plugins._plugin_home_scope(other):
            client.get("https://example.test")
        assert len(previous) == 1
        manager.unload("observer")
        client.get("https://example.test")
        assert len(previous) == 1
        manager.discover_and_load(force=True)
        client.get("https://example.test")
        assert manager._plugins["observer"].module.SEEN == [(200, ["receipt", None, None, None, None, None])]


@pytest.mark.platforms("any")
@pytest.mark.parametrize("keyword_hook", [False, True])
def test_plugin_host_refuses_response_header_access(plugin_home, keyword_hook):
    from hermes_cli.plugin_isolation_audit import audit_plugin_dir

    home, manager = plugin_home
    if keyword_hook:
        source = home / "plugins" / "observer" / "__init__.py"
        source.write_text(_PLUGIN.replace('register_hook("llm_http_response", ',
                                          'register_hook(hook_name="llm_http_response", callback='), encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["observer"], "isolation": "host"},
    }), encoding="utf-8")
    assert audit_plugin_dir(home / "plugins" / "observer").verdict == "in_process"
    try:
        manager.discover_and_load()
        assert "llm_http_response" in manager._plugins["observer"].error
        assert not manager.has_hook("llm_http_response")
    finally:
        manager._plugin_host().shutdown()
