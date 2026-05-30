"""阶段一 LLM 快速分类 — 约 500ms，temperature=0，结构化输出。

参考系统 tools/approval._smart_approve 的 prompt 设计：
  - 明确告知"多数命中是误报"（降低 LLM 的过度谨慎倾向）
  - 注入危险信号上下文（从 stage1_rules 提取）
  - 只输出 ALLOW / ESCALATE，不输出 DENY

DENY 的决定权留给阶段二 ACP（有完整上下文和工具）或第二层系统审批。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _build_classification_prompt(
    tool_name: str,
    args: Dict[str, Any],
    context: Dict[str, Any],
    task_id: str,
    session_id: str,
) -> str:
    """构造 LLM 分类 prompt，注入从 stage1_rules 提取的危险信号上下文。

    参考系统 _smart_approve 的关键设计：
      1. 明确告知 LLM"多数命中是误报"（防止过度拒绝）
      2. 给出具体的误报例子帮助 LLM 校准
      3. 注入匹配到的规则描述作为上下文
    """
    args_str = json.dumps(args, ensure_ascii=False, default=str)
    if len(args_str) > 2000:
        args_str = args_str[:2000] + "...(truncated)"

    signals = context.get("signals", [])
    signals_text = "\n".join(f"  - {s}" for s in signals) if signals else "  - 无危险信号"

    return f"""你是 AI Agent 的安全审查员。评估以下工具调用是否安全。

工具: {tool_name}
参数: {args_str}

【系统检测到的风险信号】
{signals_text}

重要提示 — 多数命中是误报:
  许多被规则引擎标记的操作实际上是完全无害的。例如：
  - `python -c "print('hello')"` 被标记为"通过 -c 标志执行脚本"，但实际上只是打印文本
  - `rm -rf ./node_modules` 被标记为"递归删除"，但实际上只是清理项目依赖

  请根据操作的 ACTUAL 风险来判断，而不仅仅是看规则标签。

判断标准:
  - ALLOW — 操作安全，不会造成系统损坏或数据丢失
    例: 项目目录内的文件修改、构建操作、包管理、git 操作、
         常规文件读写、在用户项目路径下的操作
  - ESCALATE — 不确定，需要更深入的审查（交给阶段二 ACP 处理）

路径判断:
  - /home/、/mnt/f/、~/projects/ 等用户项目路径 → 大概率安全
  - /etc/、/boot/、~/.ssh/ 等系统/安全路径 → 需要谨慎

仅回答一个词: ALLOW 或 ESCALATE"""


def llm_classify(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
    context: Dict[str, Any],
    task_id: str,
    session_id: str,
) -> str:
    """调用 LLM 进行快速安全分类。

    Args:
        tool_name: 工具名
        args: 工具参数
        cfg: 插件配置
        context: 从 stage1_rules.extract_context() 获取的风险信号
        task_id: 任务 ID
        session_id: 会话 ID

    Returns:
        "ALLOW" | "ESCALATE"（不输出 DENY）
    """
    fail_open = cfg.get("fail_open", True)
    timeout = cfg.get("stage1", {}).get("timeout", 5)

    try:
        from agent.auxiliary_client import call_llm

        prompt = _build_classification_prompt(
            tool_name, args, context, task_id, session_id
        )
        response = call_llm(
            task="approval",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=32,
            timeout=timeout,
        )
        answer = (response.choices[0].message.content or "").strip().upper()

        if "ALLOW" in answer:
            return "ALLOW"
        else:
            # 包括 "ESCALATE" 和任何无法识别的回答
            return "ESCALATE"

    except Exception as exc:
        logger.warning(
            "Stage1 LLM classify failed: %s (fail_open=%s)", exc, fail_open
        )
        if fail_open:
            return "ALLOW"
        return "ESCALATE"
