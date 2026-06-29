from __future__ import annotations

from pathlib import Path

from .artifact_paths import implementation_artifacts_relpath
from .utils import read_text


def render_step_prompt(
    contract: dict[str, object],
    *,
    project_root: str | Path,
    story_id: str,
    story_prefix: str,
    extra_instruction: str,
) -> str:
    prompt_cfg = _dict_value(contract.get("prompt"))
    assets_cfg = _dict_value(contract.get("assets"))
    assets = _dict_value(assets_cfg.get("files"))
    template = read_text(str(prompt_cfg.get("templatePath") or ""))
    replacements = {
        "{{story_id}}": story_id,
        "{{story_prefix}}": story_prefix,
        "{{label}}": str(contract.get("label") or ""),
        "{{implementation_artifacts}}": implementation_artifacts_relpath(project_root),
        "{{skill_line}}": _prompt_line("READ this skill first", str(assets.get("skill") or "")),
        "{{workflow_line}}": _prompt_line("READ this workflow file next", str(assets.get("workflow") or "")),
        "{{instructions_line}}": _prompt_line("Then read", str(assets.get("instructions") or "")),
        "{{checklist_line}}": _prompt_line("Validate with", str(assets.get("checklist") or "")),
        "{{template_line}}": _prompt_line("Use template", str(assets.get("template") or "")),
        "{{extra_instruction}}": extra_instruction.strip() or str(prompt_cfg.get("defaultExtraInstruction") or ""),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _prompt_line(label: str, path: str) -> str:
    return f"{label}: {path}\n" if path else ""
