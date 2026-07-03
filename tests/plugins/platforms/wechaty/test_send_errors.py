"""Wechaty sidecar send-error mapping."""

from plugins.platforms.wechaty.adapter import _send_result_from_sidecar_error


def test_session_expired_ret_is_not_retryable():
    exc = RuntimeError(
        "Wechaty sidecar /send error: wechat_session_expired "
        "(http=503, wechat_ret=1101, retryable=False)"
    )
    result = _send_result_from_sidecar_error(exc)
    assert not result.success
    assert not result.retryable
    assert "1101" in (result.error or "")
    assert "Re-scan" in (result.error or "")


def test_rate_limited_ret_is_retryable():
    exc = RuntimeError(
        "Wechaty sidecar /send error: wechat_rate_limited "
        "(http=503, wechat_ret=1205, retryable=True)"
    )
    result = _send_result_from_sidecar_error(exc)
    assert not result.success
    assert result.retryable
    assert "1205" in (result.error or "")


def test_unknown_error_not_retryable():
    exc = RuntimeError("Wechaty sidecar /send error: room not found (http=500)")
    result = _send_result_from_sidecar_error(exc)
    assert not result.success
    assert not result.retryable
