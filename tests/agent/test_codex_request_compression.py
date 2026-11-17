"""Official Codex Responses bodies go out zstd-compressed (official-client parity), with a plain-JSON resend."""

import asyncio
import json

import httpx
import pytest

from agent.codex_headers import codex_request_compression_hooks, codex_zstd_replay_client_cls
from agent.process_bootstrap import build_keepalive_http_client

OFFICIAL = "https://chatgpt.com/backend-api/codex"
BARE_400 = {"detail": "Bad Request"}


@pytest.fixture(autouse=True)
def profile_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    return tmp_path


def _disable_compression(home) -> None:
    (home / "config.yaml").write_text("agent:\n  codex_request_compression: false\n", encoding="utf-8")


def _send_through_hooks(client, url: str, payload: dict):
    request = client.build_request("POST", url, json=payload)
    for hook in client.event_hooks["request"]:
        hook(request)
    return request


def _post_against(answers, payload, *, async_mode=False):
    """POST through the official-endpoint hooks and replay client; the wire answers from ``answers``."""
    seen = []

    def handle(request):
        seen.append((request.headers.get("content-encoding"), request.content))
        status, body = answers[len(seen) - 1]
        return httpx.Response(status, json=body)

    hooks = codex_request_compression_hooks(OFFICIAL, async_mode=async_mode)
    assert hooks is not None
    client = codex_zstd_replay_client_cls(async_mode)(event_hooks=hooks, transport=httpx.MockTransport(handle))
    if not async_mode:
        with client:
            return client.post(f"{OFFICIAL}/responses", json=payload), seen

    async def go():
        async with client:
            return await client.post(f"{OFFICIAL}/responses", json=payload)

    return asyncio.run(go()), seen


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_official_codex_body_is_zstd_and_lossless_by_default(async_mode):
    zstd = pytest.importorskip("compression.zstd")
    payload = {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi " * 500}]}
    response, seen = _post_against([(200, {"ok": True})], payload, async_mode=async_mode)
    assert response.status_code == 200
    (encoding, body), = seen
    assert encoding == "zstd"
    assert json.loads(zstd.decompress(body)) == payload


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_bare_400_on_zstd_body_is_resent_once_as_plain_json(async_mode):
    """The backend rejects some valid zstd frames with a bare 400 and accepts the same JSON (#148898)."""
    pytest.importorskip("compression.zstd")
    payload = {"model": "gpt-6.1-sol", "input": "x" * 131073}
    response, seen = _post_against([(400, BARE_400), (200, {"ok": True})], payload, async_mode=async_mode)
    assert response.status_code == 200
    assert [encoding for encoding, _ in seen] == ["zstd", None]
    assert json.loads(seen[1][1]) == payload


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_plain_json_rejection_is_returned_without_another_resend(async_mode):
    pytest.importorskip("compression.zstd")
    response, seen = _post_against([(400, BARE_400), (400, BARE_400)], {"model": "x", "input": "hi"},
                                   async_mode=async_mode)
    assert (response.status_code, response.json()) == (400, BARE_400)
    assert [encoding for encoding, _ in seen] == ["zstd", None]


@pytest.mark.parametrize("answer", [
    (400, {"detail": "Bad Request", "param": "tools"}),
    (400, {"error": {"message": "Invalid value for 'input'"}}),
    (500, BARE_400),
], ids=["extra-field", "other-400", "not-400"])
def test_other_errors_are_not_resent(answer):
    pytest.importorskip("compression.zstd")
    response, seen = _post_against([answer], {"model": "x", "input": "hi"})
    assert (response.status_code, response.json()) == answer
    assert len(seen) == 1


def test_keepalive_client_uses_the_replay_client_for_official_codex():
    pytest.importorskip("compression.zstd")
    client = build_keepalive_http_client(OFFICIAL)
    try:
        assert isinstance(client, codex_zstd_replay_client_cls(False))
    finally:
        client.close()


def test_opt_out_sends_plain_json(profile_home):
    _disable_compression(profile_home)
    client = build_keepalive_http_client(OFFICIAL)
    try:
        request = _send_through_hooks(client, f"{OFFICIAL}/responses", {"model": "x", "input": "hello"})
    finally:
        client.close()
    assert "Content-Encoding" not in request.headers
    assert type(client) is httpx.Client


def test_ignore_user_config_keeps_the_default(profile_home, monkeypatch):
    pytest.importorskip("compression.zstd")
    _disable_compression(profile_home)
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    client = build_keepalive_http_client(OFFICIAL)
    try:
        request = _send_through_hooks(client, f"{OFFICIAL}/responses", {"model": "x", "input": "hello"})
    finally:
        client.close()
    assert request.headers["Content-Encoding"] == "zstd"


@pytest.mark.parametrize("base_url, path", [
    ("https://my-proxy.example.com/backend-api/codex", "/responses"),  # custom proxy: never compressed
    ("https://api.openai.com/v1", "/responses"),
    (OFFICIAL, "/models"),  # official host, non-Responses endpoint
])
def test_other_destinations_stay_raw_json(base_url, path):
    payload = {"model": "x", "input": "hello"}
    client = build_keepalive_http_client(base_url)
    try:
        request = _send_through_hooks(client, f"{base_url}{path}", payload)
    finally:
        client.close()
    assert "Content-Encoding" not in request.headers
    assert json.loads(request.content) == payload


def test_config_read_failure_keeps_the_http_client(monkeypatch):
    import hermes_cli.config

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", boom)
    client = build_keepalive_http_client(OFFICIAL)
    assert client is not None
    try:
        assert not client.event_hooks["request"]
    finally:
        client.close()


def test_unavailable_config_backend_keeps_plain_json(monkeypatch):
    import hermes_cli.config
    from hermes_cli.config_backend import ConfigBackendUnavailable

    def unavailable():
        raise ConfigBackendUnavailable("backend down")

    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", unavailable)
    assert codex_request_compression_hooks(OFFICIAL) is None
