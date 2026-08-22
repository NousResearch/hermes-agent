"""Tests for the true gateway final-egress plugin boundary."""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import _build_auto_reset_notice, _finalize_gateway_output


def test_gateway_final_output_transforms_then_validates(monkeypatch):
    calls = []

    def _transform(name, *, response_text, **kwargs):
        calls.append(("transform", name, response_text, kwargs))
        return "結果：無法安全顯示\n變更：已移除內部錯誤。\n下一步：請重試。", True

    def _validate(name, *, response_text, **kwargs):
        calls.append(("validate", name, response_text, kwargs))
        assert "private/project" not in response_text

    monkeypatch.setattr("hermes_cli.plugins.invoke_text_hook", _transform)
    monkeypatch.setattr("hermes_cli.plugins.invoke_validation_hook", _validate)
    source = SimpleNamespace(platform=Platform.TELEGRAM, user_id="u1")

    gateway = object()
    result = _finalize_gateway_output(
        source,
        "The request failed: internal stack at C:/private/project\n"
        "MEDIA:C:/private/project/opaque.txt\nmodel · C:/private/project",
        user_message="run it",
        gateway=gateway,
        force_brief=True,
    )

    assert result.startswith("結果：")
    assert [call[0] for call in calls] == ["transform", "validate"]
    assert calls[0][1] == "finalize_gateway_output"
    assert calls[1][1] == "validate_gateway_output"
    assert calls[0][3]["gateway"] is gateway
    assert calls[0][3]["force_brief"] is True


def test_telegram_auto_reset_notice_omits_session_metadata():
    notice = _build_auto_reset_notice(
        Platform.TELEGRAM,
        "inactive for 2h",
        "Model: provider/model\nEndpoint: http://local.private:9000\nCWD: C:/private",
    )

    assert notice.startswith("結果：工作階段已自動重設")
    assert "provider/model" not in notice
    assert "local.private" not in notice
    assert "C:/private" not in notice


def test_gateway_final_output_hook_failure_is_safe(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("internal terminal failure")

    monkeypatch.setattr("hermes_cli.plugins.invoke_text_hook", _raise)
    source = SimpleNamespace(platform=Platform.TELEGRAM, user_id="u1")

    result = _finalize_gateway_output(
        source, "raw output that must not be delivered", user_message="run it"
    )

    assert result.startswith("結果：無法安全顯示")
    assert "raw output" not in result
    assert "internal terminal failure" not in result
