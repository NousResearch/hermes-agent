"""Stage 2 ACP Agent deep review — independent Hermes instance for semantic reasoning.

Uses hermes chat -q --profile approval to launch an independent approval agent.
Has full conversation context, Hindsight approval memory, read-only tool access.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def acp_agent_review(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
    task_id: str = "",
    session_id: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Call ACP Agent for deep review."""
    stage2_cfg = cfg.get("stage2", {})
    profile = stage2_cfg.get("profile", "approval")
    timeout = stage2_cfg.get("timeout", 15)

    args_str = json.dumps(args, ensure_ascii=False, default=str)
    if len(args_str) > 3000:
        args_str = args_str[:3000] + "..."

    review_prompt = f"""You are a security approval agent. Review the following tool call and determine safety.

Tool: {tool_name}
Args: {args_str}

Judgment criteria (answer JSON only, no other text):
- ALLOW: operation is safe (file modifications in project dirs, git operations, etc.)
- DENY:  operation is dangerous (system dir writes, recursive deletion, privilege escalation, etc.)
- MODIFY: allow but suggest parameter changes

Output format: {{"verdict": "ALLOW|DENY|MODIFY", "reason": "explain reasoning in English", "suggestion": "modification suggestion"}}"""

    try:
        # Launch approval agent via hermes chat -q (-t is toolsets shorthand)
        result = subprocess.run(
            [
                "hermes", "chat", "-q",
                "-p", profile,
                "-t", "terminal,file,web",
                review_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HERMES_ACP_MODE": "approval"},
        )

        if result.returncode != 0:
            raise RuntimeError(f"ACP agent exited {result.returncode}: {result.stderr[:200]}")

        # Parse JSON output
        output = result.stdout.strip()
        try:
            data = json.loads(output)
            verdict = str(data.get("verdict", "DENY")).upper()
            detail = {
                "reason": str(data.get("reason", "ACP agent review")),
                "suggestion": str(data.get("suggestion", "No suggestion")),
            }
            return verdict, detail
        except json.JSONDecodeError:
            # Fallback: text analysis
            if "ALLOW" in output.upper():
                return "ALLOW", {"reason": output[:200], "suggestion": ""}
            else:
                return "DENY", {"reason": output[:200], "suggestion": ""}

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ACP agent timed out after {timeout}s")
