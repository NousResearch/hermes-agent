"""Public CLI parser/handler contracts; default writes never open resources."""
import asyncio
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from plugins.platforms.qqbot import manage
from plugins.platforms.qqbot.management import QQBotManagementClient


@pytest.mark.parametrize("operation", ["menu-put", "panel-create", "panel-update", "panel-targets", "panel-delete"])
def test_mutations_preview_without_credentials_or_http(operation, tmp_path, monkeypatch, capsys):
    def forbidden():
        pytest.fail("offline preview opened management resources")
    monkeypatch.setattr(manage, "open_management_client", forbidden)
    payload = {"future": {"empty": []}}
    path = tmp_path / "body.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    argv = [operation]
    if operation in {"panel-update", "panel-targets", "panel-delete"}:
        argv.append("a/b?#")
    if operation != "panel-delete":
        argv += ["--file", str(path)]
    assert manage.main(argv) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] is True
    assert preview["payload"] == (None if operation == "panel-delete" else payload)
    if operation.startswith("panel-") and operation != "panel-create":
        assert "a%2Fb%3F%23" in preview["path"]


def test_read_list_and_apply(tmp_path, capsys):
    client = type("Client", (), {"request": AsyncMock(return_value={"next_cursor": "next", "is_end": False})})()
    assert manage.main(["panel-list", "--scope", "future", "--cursor", "a&b", "--limit", "99"], client=client) == 0
    client.request.assert_awaited_with("GET", "/v2/panels?scope=future&cursor=a%26b&limit=99", None)
    assert json.loads(capsys.readouterr().out)["next_cursor"] == "next"
    path = tmp_path / "body.json"
    path.write_text('{"future":true}', encoding="utf-8")
    assert manage.main(["menu-put", "--file", str(path), "--apply"], client=client) == 0
    client.request.assert_awaited_with("PUT", "/v2/menu", {"future": True})


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_unicode_payload_with_or_without_bom_previews_offline(encoding, tmp_path, monkeypatch, capsys):
    def forbidden():
        pytest.fail("offline payload parsing opened management resources")

    monkeypatch.setattr(manage, "open_management_client", forbidden)
    payload = {"label": "菜单选项", "future": {"empty": []}}
    path = tmp_path / "body.json"
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode(encoding))
    assert manage.main(["menu-put", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "dry_run": True, "method": "PUT", "path": "/v2/menu", "payload": payload,
    }


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("body", ["[]", "not json"])
def test_bad_input_is_local_error(body, encoding, tmp_path, capsys):
    path = tmp_path / "body.json"
    path.write_bytes(body.encode(encoding))
    assert manage.main(["menu-put", "--file", str(path)]) == 2
    assert "JSON object" in capsys.readouterr().err


def test_help_and_remote_failure_are_safe(capsys):
    with pytest.raises(SystemExit) as exc:
        manage.main(["--help"])
    assert exc.value.code == 0
    assert "panel-targets" in capsys.readouterr().out
    client = type("Client", (), {"request": AsyncMock(side_effect=RuntimeError("secret-value"))})()
    assert manage.main(["menu-get"], client=client) == 1
    assert "secret-value" not in capsys.readouterr().err


@pytest.mark.parametrize("error_type", [OSError, ValueError, RuntimeError, httpx.ConnectError, httpx.ReadTimeout])
def test_expected_request_errors_are_secret_safe(error_type, capsys, caplog):
    client = type("Client", (), {"request": AsyncMock(side_effect=error_type("secret-value"))})()
    assert manage.main(["menu-get"], client=client) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qqbot: request failed; check configuration, permissions and connectivity\n"
    assert caplog.records == []


@pytest.mark.parametrize("error_type", [TypeError, AssertionError, asyncio.CancelledError, KeyboardInterrupt])
def test_unexpected_errors_and_cancellation_propagate(error_type, capsys):
    error = error_type("unexpected failure")
    client = type("Client", (), {"request": AsyncMock(side_effect=error)})()
    with pytest.raises(error_type) as exc:
        manage.main(["menu-get"], client=client)
    assert exc.value is error
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("authentication", [True, False])
@pytest.mark.parametrize("failure", ["status", "json", "business", "non-object"])
def test_real_http_failures_are_secret_safe(failure, authentication, monkeypatch, capsys, caplog):
    calls = []

    def respond(request):
        calls.append(request)
        if not authentication and request.url.host == "bots.qq.com":
            return httpx.Response(200, json={"access_token": "secret-token"})
        if failure == "status":
            return httpx.Response(403, text="secret-value")
        if failure == "json":
            return httpx.Response(200, text="secret-value")
        if failure == "business":
            return httpx.Response(200, json={"code": 123, "message": "secret-value"})
        return httpx.Response(200, json=["secret-value"])

    @asynccontextmanager
    async def open_client():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            yield QQBotManagementClient(http, "app", "secret-value")

    monkeypatch.setattr(manage, "open_management_client", open_client)
    assert manage.main(["menu-get"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qqbot: request failed; check configuration, permissions and connectivity\n"
    assert len(calls) == (1 if authentication else 2)
    assert caplog.records == []
