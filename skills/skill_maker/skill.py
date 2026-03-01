import os
import json
from pathlib import Path

# Projenin ana skills dizinini bulur
SKILLS_ROOT = Path(__file__).parent.parent

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "create_skill",
                "description": "Hermes için yeni bir yetenek (skill) klasörü ve dosyaları oluşturur. Kullanıcı yeni bir özellik istediğinde bunu kullan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string", "description": "Klasör adı (örn: 'hesap_makinesi')"},
                        "description": {"type": "string", "description": "Yetenek ne işe yarar?"},
                        "trigger_phrases": {"type": "array", "items": {"type": "string"}, "description": "Tetikleyici cümleler"},
                        "logic_code": {"type": "string", "description": "logic.py dosyasının Python kodu. run(**kwargs) içermeli."}
                    },
                    "required": ["skill_name", "description", "trigger_phrases", "logic_code"]
                }
            }
        }
    ]

def create_skill(skill_name: str, description: str, trigger_phrases: list, logic_code: str) -> dict:
    skill_dir = SKILLS_ROOT / skill_name
    try:
        skill_dir.mkdir(parents=True, exist_ok=False)
        triggers = "\n".join(f"- {p}" for p in trigger_phrases)
        skill_md = f"# {skill_name}\n\n## Description\n{description}\n\n## Triggers\n{triggers}"
        (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
        (skill_dir / "logic.py").write_text(logic_code, encoding="utf-8")
        return {"success": True, "path": str(skill_dir)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def dispatch(tool_name: str, tool_input: dict) -> str:
    if tool_name == "create_skill":
        return json.dumps(create_skill(**tool_input))
    return json.dumps({"error": "Unknown tool"})
