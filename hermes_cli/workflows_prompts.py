"""Prompt helpers for Hermes workflow agent-task cells.

The generic workflow template renderer intentionally treats mixed strings like
"hello ${ input.name }" as literals for pass-node output compatibility. Agent
prompts need friendlier text behavior, so prompt rendering lives here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hermes_cli.workflows_engine import render_template

_INLINE_TEMPLATE_RE = re.compile(r"\$\{\s*([^}]+?)\s*\}")


def _stringify_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def render_prompt_text(text: str, context: dict[str, Any]) -> str:
    """Render `${ ... }` placeholders anywhere inside an agent prompt string."""

    def replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        rendered = render_template("${ " + expression + " }", context)
        return _stringify_prompt_value(rendered)

    return _INLINE_TEMPLATE_RE.sub(replace, text)


def _render_prompt_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        rendered = render_template(value, context)
        if not isinstance(rendered, str) or rendered != value:
            return rendered
        return render_prompt_text(value, context)
    if isinstance(value, list):
        return [_render_prompt_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_prompt_value(item, context) for key, item in value.items()}
    return value


def render_agent_prompt(prompt: Any, context: dict[str, Any]) -> str:
    """Render a workflow agent_task prompt into a human-readable task body.

    Backward compatibility:
    - String prompts become natural text with inline template interpolation.
    - Mapping/list prompts are still supported and rendered recursively, then
      pretty-printed as JSON so existing workflow definitions keep working.
    """

    if isinstance(prompt, str):
        return render_prompt_text(prompt, context).strip()

    rendered = _render_prompt_value(prompt, context)
    if isinstance(rendered, str):
        return rendered.strip()
    return json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True)
