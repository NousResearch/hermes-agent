"""#145452 on its own config: a ``providers.<id>`` route resolves to ``provider: custom``, and the
``custom``-keyed catalog paths (titling's fast model, the ``custom`` catalog, the picker's bare
custom row) must probe it with the entry's own key, never the ambient ``OPENAI_API_KEY``.

Real config.yaml under a temp ``HERMES_HOME``, real runtime resolution, a loopback ``/v1/models``
server recording ``Authorization``.
"""

from __future__ import annotations

import http.server
import json
import threading
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from hermes_cli import models as models_mod
from hermes_cli import models_pricing

ROUTE_KEY = "route-key-secret"
OPENAI_KEY = "sk-proj-ambient-openai"


@pytest.fixture
def catalog_server():
    seen: list[str | None] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            body = json.dumps({"data": [{"id": "proxy-small-mini", "pricing": {
                "prompt": "0.000001", "completion": "0.000002"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()
    server.server_close()


@pytest.fixture
def issue_config(tmp_path, monkeypatch, catalog_server):
    """The issue's config: ``model.provider: litellm`` on the entry's own URL, key in ``key_env``."""
    url, seen = catalog_server
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "my-combo", "provider": "litellm", "base_url": url, "key_env": "HERMES_LITELLM_KEY"},
        "providers": {"litellm": {"name": "my litellm", "base_url": url, "key_env": "HERMES_LITELLM_KEY",
                                  "models": {"my-combo": {"context_length": 500000}}}},
        "auxiliary": {"title_generation": {"prefer_fast_model": True}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_LITELLM_KEY", ROUTE_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    for var in ("OPENAI_BASE_URL", "CUSTOM_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    yield url, seen
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()


def test_titling_fast_model_reads_the_declared_entrys_catalog(issue_config):
    """The runtime is ``custom`` with ``requested_provider: litellm``; titling must read the
    owning entry's catalog with its key instead of a ``custom`` lookup that never probes."""
    from agent.auxiliary_client import _main_route_target
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _url, seen = issue_config
    runtime = resolve_runtime_provider()
    assert (runtime["provider"], runtime.get("requested_provider")) == ("custom", "litellm")

    _provider, model, *_ = _main_route_target(runtime, "title_generation")

    assert model == "proxy-small-mini"
    assert seen == [f"Bearer {ROUTE_KEY}"]


@pytest.mark.parametrize(("name", "provider", "expected", "probes"), [
    ("litellm", "custom:litellm", "proxy-small-mini", [f"Bearer {ROUTE_KEY}"]),  # prefixed spelling of the entry
    ("deepseek", "deepseek", "my-combo", []),  # entry named like a built-in: never the built-in's catalog
])
def test_titling_reads_the_owning_entry_by_its_own_name(
        tmp_path, monkeypatch, catalog_server, name, provider, expected, probes):
    from agent.auxiliary_client import _main_route_target
    from hermes_cli.runtime_provider import resolve_runtime_provider

    url, seen = catalog_server
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "my-combo", "provider": provider, "base_url": url},
        "providers": {name: {"name": f"my {name}", "base_url": url, "key_env": "HERMES_ROUTE_KEY"}},
        "auxiliary": {"title_generation": {"prefer_fast_model": True}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ROUTE_KEY", ROUTE_KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-registry-key")
    runtime = resolve_runtime_provider()
    assert runtime["provider"] == "custom"

    _provider, model, *_ = _main_route_target(runtime, "title_generation")

    assert model == expected
    assert seen == probes


def test_titling_skips_the_named_entry_when_the_runtime_runs_on_another_url(issue_config):
    """A ``--base-url`` override keeps ``requested_provider: litellm`` on a URL the entry does not
    own; the entry's fast model would be sent to that URL, so titling stays on the main model."""
    from agent.auxiliary_client import _main_route_target
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _url, seen = issue_config
    runtime = resolve_runtime_provider(requested="litellm", explicit_base_url="http://127.0.0.1:9/v1")
    assert (runtime["provider"], runtime.get("requested_provider")) == ("custom", "litellm")

    _provider, model, *_ = _main_route_target(runtime, "title_generation")

    assert model == "my-combo"
    assert seen == []


def test_custom_catalog_probe_never_sends_the_ambient_openai_key(issue_config):
    """``provider_model_ids("custom")`` probes ``model.base_url``; the entry owning it supplies the
    key, and an OpenAI key is never a fallback for a third-party host (#91308)."""
    _url, seen = issue_config

    assert models_mod.provider_model_ids("custom", force_refresh=True) == ["proxy-small-mini"]
    assert seen == [f"Bearer {ROUTE_KEY}"]


def test_custom_catalog_probe_with_no_credential_is_keyless(issue_config, monkeypatch):
    """With no route credential at all the probe goes keyless, not with the OpenAI key."""
    _url, seen = issue_config
    monkeypatch.delenv("HERMES_LITELLM_KEY")

    models_mod.provider_model_ids("custom", force_refresh=True)

    assert OPENAI_KEY not in " ".join(h or "" for h in seen)


def test_picker_bare_custom_row_probes_with_the_entrys_key(issue_config):
    """The ``/model`` picker's bare custom row probes the current custom endpoint with the
    owning entry's key, not keyless (the issue's ``hermes-cli`` no-auth fingerprint)."""
    from hermes_cli.model_switch_providers import list_authenticated_providers

    url, seen = issue_config
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        list_authenticated_providers(
            current_provider="custom", current_base_url=url, current_model="my-combo",
            user_providers={}, custom_providers=[], refresh=True, probe_current_custom_provider=True)

    assert seen and set(seen) == {f"Bearer {ROUTE_KEY}"}


@pytest.mark.parametrize(("base_url", "expected"), [
    ("http://127.0.0.1:9/v1", ""),                      # third-party host: no OpenAI key (#91308)
    ("https://api.openai.com/v1", OPENAI_KEY),          # the key's own host still gets it
])
def test_custom_probe_with_no_declared_entry_gates_the_openai_key_on_its_host(
        tmp_path, monkeypatch, base_url, expected):
    """A bare ``custom`` endpoint no ``providers:`` entry declares falls back to the host-gated env
    keys only, so the ambient OpenAI key reaches api.openai.com and nothing else."""
    from hermes_cli.auth_declared_providers import custom_catalog_probe_api_key

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "m", "provider": "custom", "base_url": base_url}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    for var in ("OPENAI_BASE_URL", "CUSTOM_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert custom_catalog_probe_api_key({"provider": "custom", "base_url": base_url}, base_url) == expected


def test_picker_bare_row_runs_a_key_cmd_helper_only_when_it_probes(tmp_path, monkeypatch, catalog_server):
    """Cache-only picker builds (status, non-blocking GUI reads) must not run the entry's key_cmd
    helper; a probing build mints it and sends the token."""
    import sys

    from hermes_cli.model_switch_providers import list_authenticated_providers

    url, seen = catalog_server
    runs = tmp_path / "runs.txt"
    helper = tmp_path / "mint.py"
    helper.write_text(f"open({str(runs)!r}, 'a').write('x')\nprint('cmd-token')\n", encoding="utf-8")
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "my-combo", "provider": "litellm", "base_url": url},
        "providers": {"litellm": {"name": "my litellm", "base_url": url, "key_cmd": f'"{sys.executable}" "{helper}"'}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    def build(**kw):
        list_authenticated_providers(current_provider="custom", current_base_url=url, for_picker=True,
                                     current_model="my-combo", **kw)

    for _ in range(2):
        build(refresh=False, probe_current_custom_provider=False, non_blocking_catalogs=True)
    assert not runs.exists() and seen == []
    build(refresh=True, probe_current_custom_provider=True)
    assert runs.read_text() == "x"
    assert seen == ["Bearer cmd-token"]


def test_custom_catalog_mints_a_key_cmd_entrys_token_before_probing(tmp_path, monkeypatch, catalog_server):
    """``provider_model_ids("custom")`` hands the key to a plain-string probe: the owning entry's
    ``key_cmd`` source must be minted, never sent as its object repr."""
    import sys

    url, seen = catalog_server
    helper = tmp_path / "mint.py"
    helper.write_text("print('cmd-token')\n", encoding="utf-8")
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "my-combo", "provider": "custom", "base_url": url},
        "providers": {"litellm": {"name": "my litellm", "base_url": url, "key_cmd": f'"{sys.executable}" "{helper}"'}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert models_mod.provider_model_ids("custom", force_refresh=True) == ["proxy-small-mini"]
    assert seen == ["Bearer cmd-token"]
