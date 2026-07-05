"""Seam-test + behavior test for apex_overlay.cn_im_messages (hc-401 SEAM D).

Pins the four upstream choke points the seam monkey-patches so an upstream
rename turns a silently-disarmed guard (English leaking to CN-IM users) into a
loud CI failure:

* ``gateway.run._prepare_gateway_status_message(platform, event_type, message)``
* ``gateway.run.GatewayRunner._deliver_platform_notice(self, source, content)``
* ``gateway.run.GatewayRunner._handle_message(self, event)``  (mid-run rejections)
* ``gateway.platforms.base.BasePlatformAdapter._send_with_retry(self, chat_id, content, ...)``

Behavior proven (the localizer is the load-bearing data):
* ``_hc228_localize("dingtalk", <english busy-ack fragment>)`` → Chinese;
* ``_hc228_localize("feishu", <same>)`` → unchanged;
* a real answer (no control fragment) passes through byte-for-byte even on a
  CN-IM platform (the safety property that makes the _send_with_retry wrap ok);
* the wrapped choke points route their user-facing text through the localizer.

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter).
"""

from __future__ import annotations

import asyncio
import inspect

from apex_overlay import cn_im_messages


_BUSY_ACK_EN = "⚡ Interrupting current task. I'll respond to your message shortly."
_BUSY_ACK_ZH_FRAGMENT = "正在中断当前任务"


# ---------------------------------------------------------------------------
# Seam assertions — pin the four patched symbols' existence + signature
# ---------------------------------------------------------------------------

def test_seam_target_prepare_status_signature():
    from gateway import run as run_mod

    fn = getattr(run_mod, cn_im_messages._TARGET_PREPARE_FN, None)
    assert fn is not None, (
        "gateway.run._prepare_gateway_status_message is gone — CN-IM status "
        "localization loses its central choke point. Update "
        "apex_overlay.cn_im_messages._TARGET_PREPARE_FN."
    )
    params = list(inspect.signature(fn).parameters)
    assert params[:3] == ["platform", "event_type", "message"], (
        f"_prepare_gateway_status_message params changed to {params!r}; the "
        f"overlay wrapper forwards (platform, event_type, message)."
    )


def test_seam_target_deliver_notice_signature():
    from gateway.run import GatewayRunner

    method = getattr(GatewayRunner, cn_im_messages._TARGET_DELIVER_METHOD, None)
    assert method is not None, (
        "GatewayRunner._deliver_platform_notice is gone — the no-home-channel "
        "notice can no longer be localized. Update "
        "apex_overlay.cn_im_messages._TARGET_DELIVER_METHOD."
    )
    assert inspect.iscoroutinefunction(method)
    params = list(inspect.signature(method).parameters)
    assert params[:3] == ["self", "source", "content"], (
        f"_deliver_platform_notice params changed to {params!r}; the overlay "
        f"wrapper forwards (self, source, content)."
    )


def test_seam_target_handle_message_signature():
    from gateway.run import GatewayRunner

    method = getattr(GatewayRunner, cn_im_messages._TARGET_HANDLE_METHOD, None)
    assert method is not None, (
        "GatewayRunner._handle_message is gone — mid-run command rejections can "
        "no longer be localized. Update apex_overlay.cn_im_messages._TARGET_HANDLE_METHOD."
    )
    assert inspect.iscoroutinefunction(method)
    params = list(inspect.signature(method).parameters)
    assert params[:2] == ["self", "event"], (
        f"_handle_message params changed to {params!r}; the overlay wrapper "
        f"forwards (self, event) and localizes the str return via event.source.platform."
    )


def test_seam_target_send_with_retry_signature():
    from gateway.platforms.base import BasePlatformAdapter

    method = getattr(BasePlatformAdapter, cn_im_messages._TARGET_SEND_METHOD, None)
    assert method is not None, (
        "BasePlatformAdapter._send_with_retry is gone — the busy-ack / "
        "gateway-busy messages can no longer be localized. Update "
        "apex_overlay.cn_im_messages._TARGET_SEND_METHOD."
    )
    assert inspect.iscoroutinefunction(method)
    params = list(inspect.signature(method).parameters)
    assert params[:3] == ["self", "chat_id", "content"], (
        f"_send_with_retry params changed to {params!r}; the overlay wrapper "
        f"forwards (self, chat_id, content) and gates on self.platform."
    )


def test_apply_wraps_all_four_and_is_idempotent():
    from gateway import run as run_mod
    from gateway.run import GatewayRunner
    from gateway.platforms.base import BasePlatformAdapter

    cn_im_messages._APPLIED = False
    assert cn_im_messages.apply() is True

    assert getattr(run_mod._prepare_gateway_status_message, cn_im_messages._MARK, False)
    assert getattr(GatewayRunner._deliver_platform_notice, cn_im_messages._MARK, False)
    assert getattr(GatewayRunner._handle_message, cn_im_messages._MARK, False)
    assert getattr(BasePlatformAdapter._send_with_retry, cn_im_messages._MARK, False)

    # Idempotent: no double-wrap.
    refs = (
        run_mod._prepare_gateway_status_message,
        GatewayRunner._deliver_platform_notice,
        GatewayRunner._handle_message,
        BasePlatformAdapter._send_with_retry,
    )
    assert cn_im_messages.apply() is True
    assert run_mod._prepare_gateway_status_message is refs[0]
    assert GatewayRunner._deliver_platform_notice is refs[1]
    assert GatewayRunner._handle_message is refs[2]
    assert BasePlatformAdapter._send_with_retry is refs[3]


# ---------------------------------------------------------------------------
# Behavior — the localizer (load-bearing data)
# ---------------------------------------------------------------------------

def test_localize_busy_ack_for_dingtalk():
    out = cn_im_messages._hc228_localize("dingtalk", _BUSY_ACK_EN)
    assert _BUSY_ACK_ZH_FRAGMENT in out, f"expected Chinese busy-ack, got {out!r}"
    assert "Interrupting current task" not in out


def test_localize_leaves_feishu_unchanged():
    out = cn_im_messages._hc228_localize("feishu", _BUSY_ACK_EN)
    assert out == _BUSY_ACK_EN, "feishu must be returned unchanged (不动飞书)"


def test_localize_leaves_international_unchanged():
    for plat in ("telegram", "slack", "discord", "whatsapp"):
        out = cn_im_messages._hc228_localize(plat, _BUSY_ACK_EN)
        assert out == _BUSY_ACK_EN, f"{plat} must keep English"


def test_localize_status_detail_regex():
    en = "⏳ Queued for the next turn (iteration 6/90, 3 min elapsed, running: terminal). I'll respond once the current task finishes."
    out = cn_im_messages._hc228_localize("wecom", en)
    assert "第 6/90 轮" in out
    assert "已运行 3 分钟" in out
    assert "正在执行: terminal" in out
    assert "iteration" not in out and "elapsed" not in out


def test_localize_mid_run_rejections_including_moa():
    # the /moa rejection was NOT in the original build-time literal list; the
    # method-return wrap covers it via _HC228_ZH_LITERALS_EXTRA.
    moa = "Agent is running — wait or /stop first, then run /moa."
    out = cn_im_messages._hc228_localize("dingtalk", moa)
    assert "AI 助手正在运行" in out
    assert "/moa" in out  # command token stays literal
    assert "Agent is running" not in out


def test_localize_no_op_on_real_answer_even_on_cn_platform():
    """A real answer with no control fragment passes through byte-for-byte.

    This is the safety property that makes wrapping _send_with_retry (which
    carries ALL content) acceptable.
    """
    answer = "这是一个正常的回答，包含 running: 关键字之外无控制片段。The weather is nice."
    # note: deliberately avoid any exact control fragment
    plain = "Here is your analysis of Q4 revenue. Everything looks good."
    assert cn_im_messages._hc228_localize("dingtalk", plain) == plain


def test_localize_is_fail_safe():
    # None / empty pass through; an exploding platform object returns original.
    assert cn_im_messages._hc228_localize("dingtalk", None) is None
    assert cn_im_messages._hc228_localize("dingtalk", "") == ""

    class _Boom:
        @property
        def value(self):
            raise RuntimeError("boom")

    # platform key resolution fails → treated as non-CN → unchanged
    assert cn_im_messages._hc228_localize(_Boom(), _BUSY_ACK_EN) == _BUSY_ACK_EN


# ---------------------------------------------------------------------------
# Behavior — the wrappers route text through the localizer
# ---------------------------------------------------------------------------

class _Plat:
    def __init__(self, value):
        self.value = value


def test_prepare_status_wrapper_localizes():
    def _orig(platform, event_type, message):
        return message  # pretend it passed the raw text through

    wrapped = cn_im_messages._wrap_prepare_status(_orig)
    out = wrapped(_Plat("dingtalk"), "busy", _BUSY_ACK_EN)
    assert _BUSY_ACK_ZH_FRAGMENT in out
    # feishu unchanged
    assert wrapped(_Plat("feishu"), "busy", _BUSY_ACK_EN) == _BUSY_ACK_EN
    # None return passes through
    assert wrapped(_Plat("dingtalk"), "busy", None) is None


def test_deliver_notice_wrapper_localizes_content():
    captured = {}

    async def _orig(self, source, content):
        captured["content"] = content

    wrapped = cn_im_messages._wrap_deliver_notice(_orig)
    notice_en = "📬 No home channel is set for WeCom. A home channel is where Hermes delivers cron job results and cross-platform messages.\n\nType /sethome to make this chat your home channel, or ignore to skip."
    src = type("S", (), {"platform": _Plat("wecom")})()

    asyncio.new_event_loop().run_until_complete(wrapped(object(), src, notice_en))
    assert "尚未设置主频道" in captured["content"]
    assert "No home channel is set for" not in captured["content"]


def test_handle_message_wrapper_localizes_str_return():
    async def _orig(self, event):
        return "Agent is running — wait or /stop first, then switch models."

    wrapped = cn_im_messages._wrap_handle_message(_orig)
    ev = type("E", (), {"source": type("S", (), {"platform": _Plat("dingtalk")})()})()
    out = asyncio.new_event_loop().run_until_complete(wrapped(object(), ev))
    assert "AI 助手正在运行" in out

    # None return passes through
    async def _orig_none(self, event):
        return None

    wrapped_none = cn_im_messages._wrap_handle_message(_orig_none)
    assert asyncio.new_event_loop().run_until_complete(wrapped_none(object(), ev)) is None


def test_send_with_retry_wrapper_localizes_for_cn_only():
    captured = {}

    async def _orig(self, chat_id, content, *a, **k):
        captured["content"] = content
        return "sent"

    wrapped = cn_im_messages._wrap_send_with_retry(_orig)

    # CN-IM adapter → localized
    cn_self = type("A", (), {"platform": _Plat("dingtalk")})()
    asyncio.new_event_loop().run_until_complete(wrapped(cn_self, "c1", _BUSY_ACK_EN))
    assert _BUSY_ACK_ZH_FRAGMENT in captured["content"]

    # feishu adapter → unchanged
    fs_self = type("A", (), {"platform": _Plat("feishu")})()
    asyncio.new_event_loop().run_until_complete(wrapped(fs_self, "c1", _BUSY_ACK_EN))
    assert captured["content"] == _BUSY_ACK_EN

    # real answer on CN adapter → byte-for-byte
    plain = "Here is your Q4 revenue report."
    asyncio.new_event_loop().run_until_complete(wrapped(cn_self, "c1", plain))
    assert captured["content"] == plain


def test_plugin_register_applies_seam():
    from tests.apex_overlay.conftest import run_plugin_register_with_stubbed_seams

    called = run_plugin_register_with_stubbed_seams("_cn_im_plugin_under_test")
    assert "cn_im_messages" in called, (
        "plugin.register() must call cn_im_messages.apply()"
    )
