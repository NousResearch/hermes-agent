"""Stage 1: LLM fast classification — ~500ms, temperature=0, structured output.

Follows system tools/approval._smart_approve prompt design:
  - Explicitly tells LLM "most matches are false positives" (reduce over-caution)
  - Injects risk signal context (extracted by stage1_rules)
  - Only outputs ALLOW / ESCALATE, never DENY

DENY authority is reserved for Stage 2 ACP (full context) or system layer.
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
    """Build LLM classification prompt with risk signal context from stage1_rules.

    Follows system _smart_approve key design:
      1. Tell LLM "most matches are false positives" (prevent over-denial)
      2. Give concrete false-positive examples to calibrate
      3. Inject matched rule descriptions as context
    """
    args_str = json.dumps(args, ensure_ascii=False, default=str)
    if len(args_str) > 2000:
        args_str = args_str[:2000] + "...(truncated)"

    signals = context.get("signals", [])
    signals_text = "\n".join(f"  - {s}" for s in signals) if signals else "  - No risk signals"

    return f"""You are an AI Agent safety reviewer. Evaluate whether the following tool call is safe.

Tool: {tool_name}
Args: {args_str}

[System-detected risk signals]
{signals_text}

Important — most matches are false positives:
  Many operations flagged by the rule engine are actually harmless. For example:
  - `python -c "print('hello')"` is flagged as "executing script via -c flag", but it just prints text
  - `rm -rf ./node_modules` is flagged as "recursive delete", but it is just cleaning project dependencies

  Judge by the ACTUAL risk of the operation, not just the rule label.

Criteria:
  - ALLOW — operation is safe, no risk of system damage or data loss
    Examples: file modifications within project directories, build operations,
             package management, git operations, normal file I/O,
             operations under user project paths
  - ESCALATE — unsure, needs deeper review (handled by Stage 2 ACP)

Path judgment:
  - /home/, /mnt/f/, ~/projects/ etc. user project paths → likely safe
  - /etc/, /boot/, ~/.ssh/ etc. system/security paths → exercise caution

Answer with exactly one word: ALLOW or ESCALATE"""


def llm_classify(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
    context: Dict[str, Any],
    task_id: str,
    session_id: str,
) -> str:
    """Call LLM for fast safety classification.

    Args:
        tool_name: Tool name
        args: Tool arguments
        cfg: Plugin config
        context: Risk signals from stage1_rules.extract_context()
        task_id: Task ID
        session_id: Session ID

    Returns:
        "ALLOW" | "ESCALATE" (never DENY)
    """
    fail_open = cfg.get("fail_open", True)
    timeout = cfg.get("stage1", {}).get("timeout", 5)
    # Read provider/model from plugin_guard config so the user's
    # plugin_guard.provider / plugin_guard.model actually takes effect.
    # call_llm resolution path: explicit args → auxiliary.{task} → "auto"
    # Without explicit args, call_llm looks in auxiliary.approval, which is
    # a different config namespace — plugin_guard config would be ignored.
    explicit_provider = cfg.get("provider")
    explicit_model = cfg.get("model")

    try:
        from agent.auxiliary_client import call_llm

        prompt = _build_classification_prompt(
            tool_name, args, context, task_id, session_id
        )
        response = call_llm(
            task="approval",
            provider=explicit_provider,
            model=explicit_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=32,
            timeout=timeout,
        )
        answer = (response.choices[0].message.content or "").strip().upper()

        if "ALLOW" in answer:
            return "ALLOW"
        else:
            # Including "ESCALATE" and any unrecognized answer
            return "ESCALATE"

    except Exception as exc:
        logger.warning(
            "Stage1 LLM classify failed: %s (fail_open=%s)", exc, fail_open
        )
        if fail_open:
            return "ALLOW"
        return "ESCALATE"
