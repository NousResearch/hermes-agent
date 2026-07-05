"""hc-401 (was patch_native_agent_runtime_messages_zh.py / hc-228 + hc-340) —
Chinese-ize proactively-pushed gateway control/status messages on native CN-IM
entries, as a zero-in-place overlay seam.

What this replaces
==================
QQ/WeCom/WeChat/DingTalk real-machine acceptance leaked English gateway control
messages to end users, e.g. the busy-ack "⚡ Interrupting current task
(iteration 6/90). I'll respond to your message shortly.", the first-time
"/busy …" onboarding tip, "📬 No home channel is set for …", and the
"Agent is running — … /stop first" mid-run command rejections. Terminal users on
these platforms can't read them.

The original cloud fix was a build-time in-place patch on ``gateway/run.py`` that
injected a localizer + translation tables at module scope and then spliced
``_hc228_localize(...)`` into a fixed set of private send/return sites (two
busy-ack ``content=message`` sends, the no-home-channel notice, and each specific
``Agent is running`` return string). This module re-expresses that as a
zero-in-place overlay: the translation tables (the load-bearing data) are copied
verbatim below, and ``apply()`` monkey-patches a small set of CENTRAL choke
functions so their user-facing text flows through the localizer — rather than
editing brittle inline string sites.

Only the five native Chinese-IM entries are localized
(``wecom / weixin / dingtalk / wecom_callback / qqbot``). Every other platform is
returned unchanged:
  * Feishu — per ticket, 不动飞书 (no regression on the validated Feishu path);
  * international platforms (telegram/slack/discord/whatsapp/…) keep English,
    which is correct for their users.

The choke points patched (all zero-in-place monkey-patches)
===========================================================
1. ``gateway.run._prepare_gateway_status_message(platform, event_type, message)``
   — a module function every filtered agent status callback flows through. Its
   return value is localized (gated on ``platform``).
2. ``GatewayRunner._deliver_platform_notice(self, source, content)`` — the notice
   delivery method. ``content`` is localized (gated on ``source.platform``)
   before delivery. Covers the "No home channel is set for …" notice.
3. ``GatewayRunner._handle_message(self, event)`` — its return value (the
   slash-command reply, incl. every "Agent is running — …" mid-run rejection) is
   localized (gated on ``event.source.platform``). Wrapping the METHOD RETURN
   (rather than each specific ``return`` string as the original patch did) covers
   ALL such rejections — including ones the original string-list missed (e.g. the
   ``/moa`` "wait or /stop first, then run /moa." rejection present in this
   runtime) and any future rejection strings — with zero in-place edits.
4. ``BasePlatformAdapter._send_with_retry(self, chat_id, content, ...)`` — the
   busy-ack / gateway-busy ``content=message`` send sites in
   ``_handle_active_session_busy_message`` build ``message`` inline and call
   ``adapter._send_with_retry(content=message)`` directly, WITHOUT flowing through
   ``_prepare_gateway_status_message`` (verified by reading run.py). There is no
   narrower central function for that busy-ack path, so we localize ``content`` at
   ``_send_with_retry`` — gated strictly on the CN-IM platform.

   SAFETY of patching _send_with_retry (it carries ALL content, incl. real
   answers): ``_hc228_localize`` is a pure find-and-replace over a fixed table of
   English CONTROL-message fragments and returns the text UNCHANGED when no
   fragment matches. A real model answer contains none of those exact control
   fragments, so it passes through byte-for-byte. Combined with the CN-IM
   platform gate, the blast radius is: "on wecom/weixin/dingtalk/wecom_callback/
   qqbot, any outgoing text that happens to contain a known English gateway
   control fragment is Chinese-ized." That is precisely the intended set. We do
   NOT blanket-translate.

COVERAGE BOUNDARY (mirrors the original patch's honesty)
========================================================
Covered: the busy-ack family (interrupt/queue/steer/subagent/gateway-busy incl.
the "(iteration i/m, N min elapsed, running: tool)" detail), the first-time
busy-input onboarding tip (appended into the busy-ack ``message``), the
no-home-channel notice, and the "Agent is running — …" mid-run command
rejections.
NOT covered (follow-up, lower blast radius): other slash-command usage replies
(e.g. /help, /usage, /insights output that are not routed through _handle_message
as a plain string return), and the long-running "⏳ Working — N min" heartbeat.

Idempotent (``_MARK`` sentinels + module ``_APPLIED``) and fail-safe: the
localizer's own try/except returns the original text on any error, and every
wrapper falls through to the unmodified host path on error. ``apply()`` returns
False only if a target symbol is missing (the seam-test turns that into a loud CI
failure).
"""

from __future__ import annotations

import functools
import logging
import re

logger = logging.getLogger(__name__)

# Upstream targets we monkey-patch — centralized so the seam-test pins them.
_TARGET_RUN_MODULE = "gateway.run"
_TARGET_PREPARE_FN = "_prepare_gateway_status_message"
_TARGET_RUNNER_CLASS = "GatewayRunner"
_TARGET_DELIVER_METHOD = "_deliver_platform_notice"
_TARGET_HANDLE_METHOD = "_handle_message"
_TARGET_BASE_MODULE = "gateway.platforms.base"
_TARGET_ADAPTER_CLASS = "BasePlatformAdapter"
_TARGET_SEND_METHOD = "_send_with_retry"

_APPLIED = False
_MARK = "_apex_overlay_cn_im_localize"


# ===========================================================================
# Translation tables — COPIED VERBATIM from
# scripts/patch_native_agent_runtime_messages_zh.py (the load-bearing data).
# Do not paraphrase: these strings must byte-match the English the gateway
# emits, or a fragment leaks in English.
# ===========================================================================

_HC228_ZH_PLATFORMS = frozenset(
    {"wecom", "weixin", "dingtalk", "wecom_callback", "qqbot"}
)

# Ordered longest/most-specific first so partial fragments never clobber a
# fuller match. Every English fragment of the covered message family is listed,
# so a gated platform never sees a partial-English message.
_HC228_ZH_LITERALS = [
    # --- first-time busy-input onboarding tips (full sentences) ---
    (
        "\U0001f4a1 First-time tip — I queued your message instead of "
        "interrupting. Send `/busy interrupt` to make new messages stop the "
        "current task immediately, or `/busy status` to check. This notice "
        "won't appear again.",
        "\U0001f4a1 首次提示——我把你的"
        "消息排队了，没有打断当前"
        "任务。发送 `/busy interrupt` 可让新消"
        "息立即停止当前任务，或发"
        "送 `/busy status` 查看状态。此提示"
        "只显示一次。",
    ),
    (
        "\U0001f4a1 First-time tip — I steered your message into the "
        "current run; it will arrive after the next tool call instead of "
        "interrupting. Send `/busy interrupt` or `/busy queue` to change this, "
        "or `/busy status` to check. This notice won't appear again.",
        "\U0001f4a1 首次提示——我把你的"
        "消息插入了当前运行，它会"
        "在下一次工具调用后到达，"
        "而不是打断当前任务。发送 "
        "`/busy interrupt` 或 `/busy queue` 可更改，或"
        "发送 `/busy status` 查看状态。此提"
        "示只显示一次。",
    ),
    (
        "\U0001f4a1 First-time tip — I just interrupted my current task to "
        "answer you. Send `/busy queue` to queue follow-ups for after the "
        "current task instead, `/busy steer` to inject them mid-run without "
        "interrupting, or `/busy status` to check. This notice won't appear "
        "again.",
        "\U0001f4a1 首次提示——我刚刚中"
        "断当前任务来回答你。发送 "
        "`/busy queue` 可改为把后续消息排到"
        "当前任务之后，`/busy steer` 可在运"
        "行中插入而不打断，或发送 "
        "`/busy status` 查看状态。此提示只"
        "显示一次。",
    ),
    # --- gateway-busy (full strings, both gerunds x both phrasings) ---
    (
        "⏳ Gateway is restarting and is not accepting another turn right now.",
        "⏳ 网关正在重启，暂时无法"
        "处理新一轮消息。",
    ),
    (
        "⏳ Gateway is shutting down and is not accepting another turn right now.",
        "⏳ 网关正在关闭，暂时无法"
        "处理新一轮消息。",
    ),
    (
        "⏳ Gateway is restarting and is not accepting new work right now.",
        "⏳ 网关正在重启，暂时无法"
        "接受新任务。",
    ),
    (
        "⏳ Gateway is shutting down and is not accepting new work right now.",
        "⏳ 网关正在关闭，暂时无法"
        "接受新任务。",
    ),
    (
        "⏳ Gateway restarting — queued for the next turn after it comes back.",
        "⏳ 网关重启中——已排队，"
        "恢复后处理下一轮。",
    ),
    (
        "⏳ Gateway shutting down — queued for the next turn after it comes back.",
        "⏳ 网关关闭中——已排队，"
        "恢复后处理下一轮。",
    ),
    # --- busy-ack core (prefix + suffix fragments around {status_detail}) ---
    ("⏩ Steered into current run", "⏩ 已插入当前运行"),
    (
        ". Your message arrives after the next tool call.",
        "。你的消息将在下一次工具"
        "调用后送达。",
    ),
    ("⏳ Subagent working", "⏳ 子任务运行中"),
    (
        " — your message is queued for when it finishes (use /stop to cancel everything).",
        "——你的消息已排队，等它"
        "完成后处理（发送 /stop 取消全"
        "部）。",
    ),
    ("⏳ Queued for the next turn", "⏳ 已排队到下一轮"),
    (
        ". I'll respond once the current task finishes.",
        "。当前任务完成后我会回复"
        "你。",
    ),
    ("⚡ Interrupting current task", "⚡ 正在中断当前任务"),
    (
        ". I'll respond to your message shortly.",
        "。稍后回复你。",
    ),
    # --- command-while-running rejections (hc-340): _handle_message returns
    # these when a slash command is typed mid-run. Full sentences for the
    # fixed-text rejections; the catch-all interpolates the command name
    # `/{name}`, so it is split into a prefix + suffix fragment that keep the
    # command token literal. "AI 助手" per the zh user-surface guideline. ---
    (
        "Agent is running — wait or /stop first, then switch models.",
        "AI 助手正在运行——请先等待或发送 /stop，然后再切换模型。",
    ),
    (
        "Agent is running — wait or /stop first, then change runtime.",
        "AI 助手正在运行——请先等待或发送 /stop，然后再切换运行时。",
    ),
    (
        "Agent is running — use /goal status / pause / clear mid-run, or "
        "/stop before setting a new goal.",
        "AI 助手正在运行——运行中可用 /goal status / pause / clear，"
        "或先发送 /stop 再设置新目标。",
    ),
    ("⏳ Agent is running — ", "⏳ AI 助手正在运行——"),
    (
        " can't run mid-turn. Wait for the current response or `/stop` first.",
        " 不能在任务进行中执行，请等当前回复完成，"
        "或先发送 `/stop`。",
    ),
    # --- No home channel notice (fragments around {platform} / {sethome_cmd}) ---
    ("\U0001f4ec No home channel is set for ", "\U0001f4ec 尚未设置主频道（"),
    (
        ". A home channel is where Hermes delivers cron job results and "
        "cross-platform messages.\n\nType ",
        "）。主频道用于接收 Hermes 的定"
        "时任务结果和跨平台消息。"
        "\n\n发送 ",
    ),
    (
        " to make this chat your home channel, or ignore to skip.",
        " 即可把当前会话设为主频道"
        "，或忽略本提示。",
    ),
]

# status_detail numeric/dynamic bits.
_HC228_ZH_REGEX = [
    (re.compile(r"(\d+) min elapsed"), "已运行 \\1 分钟"),
    (re.compile(r"iteration (\d+)/(\d+)"), "第 \\1/\\2 轮"),
    (re.compile(r"running: "), "正在执行: "),
]

# v2026.7.1-era extra rejection strings NOT in the original build-time table's
# literal list (the original patched only the specific returns it enumerated and
# would have missed these). Because this seam wraps the whole ``_handle_message``
# return, these are localized too. Listed here so the coverage is explicit and
# the seam-test can assert them.
_HC228_ZH_LITERALS_EXTRA = [
    (
        "Agent is running — use /goal status / pause / clear / wait mid-run, "
        "or /stop before setting a new goal.",
        "AI 助手正在运行——运行中可用 /goal status / pause / clear / wait，"
        "或先发送 /stop 再设置新目标。",
    ),
    (
        "Agent is running — wait or /stop first, then run /moa.",
        "AI 助手正在运行——请先等待或发送 /stop，然后再运行 /moa。",
    ),
]


def _platform_config_key(platform) -> str:
    """Resolve a Platform enum (or raw string) to its config.yaml key.

    Self-contained (does not depend on gateway.run._platform_config_key being in
    scope). For the five CN-IM platforms the enum ``.value`` IS the config key
    (``Platform.WECOM.value == "wecom"``), so this matches upstream for every
    platform this seam cares about. ``LOCAL`` never reaches these send paths.
    """
    try:
        val = getattr(platform, "value", platform)
        return str(val or "").lower()
    except Exception:
        return ""


def _hc228_localize(platform, text):
    """Chinese-ize gateway control/status text for native CN-IM entries only
    (wecom/weixin/dingtalk/wecom_callback/qqbot). Feishu and international
    platforms are returned unchanged. Fully defensive: any error returns the
    original text so a turn is never broken by localization.

    A no-op on any non-matching text (real answers pass through byte-for-byte),
    which is what makes it safe to run over broad send surfaces gated by platform.
    """
    try:
        if not text:
            return text
        key = _platform_config_key(platform) if platform is not None else ""
        if key not in _HC228_ZH_PLATFORMS:
            return text
        out = text
        for needle, repl in _HC228_ZH_LITERALS:
            out = out.replace(needle, repl)
        for needle, repl in _HC228_ZH_LITERALS_EXTRA:
            out = out.replace(needle, repl)
        for pat, repl in _HC228_ZH_REGEX:
            out = pat.sub(repl, out)
        return out
    except Exception:
        return text


# ===========================================================================
# Monkey-patch wrappers (all fail-safe: fall through to host on error)
# ===========================================================================

def _wrap_prepare_status(orig):
    """Localize the return of ``_prepare_gateway_status_message``."""

    @functools.wraps(orig)
    def wrapper(platform, event_type, message, *args, **kwargs):
        result = orig(platform, event_type, message, *args, **kwargs)
        try:
            if result is None:
                return result
            return _hc228_localize(platform, result)
        except Exception:
            return result

    setattr(wrapper, _MARK, True)
    return wrapper


def _wrap_deliver_notice(orig):
    """Localize ``content`` in ``_deliver_platform_notice`` (no-home-channel etc.)."""

    @functools.wraps(orig)
    async def wrapper(self, source, content, *args, **kwargs):
        try:
            content = _hc228_localize(getattr(source, "platform", None), content)
        except Exception:
            pass
        return await orig(self, source, content, *args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper


def _wrap_handle_message(orig):
    """Localize the string return of ``_handle_message`` (mid-run rejections).

    ``_handle_message`` returns ``Optional[str]``; the string returns are the
    slash-command replies incl. every "Agent is running — …" rejection. Non-str
    returns (None) pass through untouched.
    """

    @functools.wraps(orig)
    async def wrapper(self, event, *args, **kwargs):
        result = await orig(self, event, *args, **kwargs)
        try:
            if isinstance(result, str) and result:
                platform = getattr(getattr(event, "source", None), "platform", None)
                return _hc228_localize(platform, result)
        except Exception:
            return result
        return result

    setattr(wrapper, _MARK, True)
    return wrapper


def _wrap_send_with_retry(orig):
    """Localize ``content`` in ``_send_with_retry`` for CN-IM platforms only.

    Covers the busy-ack / gateway-busy ``content=message`` send sites that don't
    flow through any narrower central function. Safe because ``_hc228_localize``
    is a no-op on non-control text and the platform gate is strict — see module
    docstring's SAFETY note.
    """

    @functools.wraps(orig)
    async def wrapper(self, chat_id, content, *args, **kwargs):
        try:
            content = _hc228_localize(getattr(self, "platform", None), content)
        except Exception:
            pass
        return await orig(self, chat_id, content, *args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper


# ===========================================================================
# Public entry point
# ===========================================================================

def apply() -> bool:
    """Install the CN-IM localization seam onto upstream. Idempotent, fail-safe.

    Returns True when all four choke points are wrapped (or already present),
    False if any target symbol is missing (the seam-test turns that into a hard
    CI failure). Never raises into plugin discovery.
    """
    global _APPLIED
    if _APPLIED:
        return True

    import importlib

    ok = True

    # 1. _prepare_gateway_status_message (module function).
    try:
        run_mod = importlib.import_module(_TARGET_RUN_MODULE)
        orig_prepare = getattr(run_mod, _TARGET_PREPARE_FN)
        if not getattr(orig_prepare, _MARK, False):
            setattr(run_mod, _TARGET_PREPARE_FN, _wrap_prepare_status(orig_prepare))
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not patch %s.%s — CN-IM status localization is "
            "NOT active for gateway status callbacks. Upstream may have renamed "
            "it. (%s)",
            _TARGET_RUN_MODULE, _TARGET_PREPARE_FN, exc,
        )
        ok = False

    # 2 + 3. GatewayRunner methods.
    try:
        run_mod = importlib.import_module(_TARGET_RUN_MODULE)
        runner_cls = getattr(run_mod, _TARGET_RUNNER_CLASS)

        orig_deliver = getattr(runner_cls, _TARGET_DELIVER_METHOD)
        if not getattr(orig_deliver, _MARK, False):
            setattr(runner_cls, _TARGET_DELIVER_METHOD, _wrap_deliver_notice(orig_deliver))

        orig_handle = getattr(runner_cls, _TARGET_HANDLE_METHOD)
        if not getattr(orig_handle, _MARK, False):
            setattr(runner_cls, _TARGET_HANDLE_METHOD, _wrap_handle_message(orig_handle))
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not patch %s.%s.{%s,%s} — CN-IM localization of "
            "the no-home-channel notice / mid-run rejections is NOT active. "
            "Upstream may have renamed them. (%s)",
            _TARGET_RUN_MODULE, _TARGET_RUNNER_CLASS, _TARGET_DELIVER_METHOD,
            _TARGET_HANDLE_METHOD, exc,
        )
        ok = False

    # 4. BasePlatformAdapter._send_with_retry (busy-ack / gateway-busy).
    try:
        base_mod = importlib.import_module(_TARGET_BASE_MODULE)
        adapter_cls = getattr(base_mod, _TARGET_ADAPTER_CLASS)
        orig_send = getattr(adapter_cls, _TARGET_SEND_METHOD)
        if not getattr(orig_send, _MARK, False):
            setattr(adapter_cls, _TARGET_SEND_METHOD, _wrap_send_with_retry(orig_send))
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not patch %s.%s.%s — CN-IM localization of the "
            "busy-ack / gateway-busy messages is NOT active. Upstream may have "
            "renamed it. (%s)",
            _TARGET_BASE_MODULE, _TARGET_ADAPTER_CLASS, _TARGET_SEND_METHOD, exc,
        )
        ok = False

    _APPLIED = ok
    if ok:
        logger.debug("apex_overlay: CN-IM message localization seam applied")
    return ok
