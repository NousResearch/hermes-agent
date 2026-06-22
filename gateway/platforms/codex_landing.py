"""codex_landing — 把审查结果发送回 Feishu

在业务层我们没有自己的网络发送实现，只是把最终文本交给
外部提供的 async callable ``send_reply``（在 feishu.py 中已经实现）。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


async def post_result(
    chat_id: str,
    status: str,
    result_msg: str,
    md_path: Path | None = None,
    patch_path: Path | None = None,
    send_reply: Callable[[str], asyncio.Future] | None = None,
) -> None:
    """组织内容并通过 ``send_reply`` 回写到飞书会话。

    参数
    ----
    * ``chat_id`` – 飞书会话 ID（在外层已决定）。
    * ``status`` – ``success`` / ``failure``（仅日志使用）。
    * ``result_msg`` – 人类可读的简要说明。
    * ``md_path``/``patch_path`` – 生成的文件路径，若存在会嵌入正文。
    * ``send_reply`` – async callable，接受字符串并负责发送。
    """
    parts = [result_msg]

    # 只展示文件路径，不贴原始 diff（避免飞书消息刷屏）
    if md_path and md_path.is_file():
        parts.append(f"\n📄 审查报告: {md_path}")
    if patch_path and patch_path.is_file():
        parts.append(f"📄 完整 diff: {patch_path}")

    final_text = "\n".join(parts)

    # 若外部未提供发送函数，用 lark-cli 直接发送
    if send_reply is None:
        print(f"[Feishu Reply] chat_id={chat_id}:\n{final_text}")
        return

    try:
        await send_reply(final_text)
    except Exception as e:
        logger.warning("send_reply failed (%s), falling back to lark-cli", e)
        await _send_via_lark_cli(chat_id, final_text)


async def _send_via_lark_cli(chat_id: str, text: str) -> None:
    """通过 lark-cli 独立发送消息，不依赖网关闭包"""
    import subprocess
    await asyncio.to_thread(
        subprocess.run,
        ["lark-cli", "im", "send", "--chat-id", chat_id, "--text", text],
        capture_output=True, timeout=15,
    )