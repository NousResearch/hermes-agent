"""feishu_codex — 业务适配层

在 hermes-source/gateway/platforms/feishu.py 中会以如下方式调用：
    asyncio.create_task(handle_codex_message(text, chat_id, _send_reply))

参数说明
---------
* text – 完整的用户消息文本，例如 "codex 修复 XXX"
* chat_id – 飞书会话标识，用于后续发送回复
* send_reply – async callable，接受单个字符串并负责把内容发回飞书
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)


async def handle_codex_message(
    text: str,
    chat_id: str,
    send_reply: Callable[[str], asyncio.Future],
) -> None:
    """
    1. 立即给用户一个"审核中"反馈（通过 send_reply）。
    2. 调用 codex_review.run_review 执行 Codex + 提取 diff + 评分。
    3. 使用 codex_landing.post_result 将结果回写到飞书。
    """
    # 1️⃣ 立即回复
    await send_reply("🔎 Codex 正在执行，请稍候…")

    # 2️⃣ 调用审查子模块
    from .codex_review import run_review
    from .codex_landing import post_result

    project_name = "hermes-source"
    description = text

    # 剥离 /codex 或 codex 前缀，保留用户实际指令
    user_instruction = description
    if user_instruction.startswith("/codex"):
        user_instruction = user_instruction[len("/codex"):].strip()
    elif user_instruction.lower().startswith("codex"):
        user_instruction = user_instruction[len("codex"):].strip()

    md_path, patch_path, status, result_msg, _score = await run_review(
        project_name, user_instruction
    )

    # 3️⃣ 回写飞书
    await post_result(
        chat_id=chat_id,
        status=status,
        result_msg=result_msg,
        md_path=md_path,
        patch_path=patch_path,
        send_reply=send_reply,
    )