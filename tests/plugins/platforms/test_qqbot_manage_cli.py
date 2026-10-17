"""Public CLI parser/handler contracts; default writes never open resources."""
import json
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.qqbot import manage


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


@pytest.mark.parametrize("body", ["[]", "not json"])
def test_bad_input_is_local_error(body, tmp_path, capsys):
    path = tmp_path / "body.json"
    path.write_text(body, encoding="utf-8")
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
