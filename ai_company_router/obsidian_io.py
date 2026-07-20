# Obsidian I/O Utilities
import json
import os
VAULT_PATH = os.environ.get(
    "OBSIDIAN_VAULT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "obsidian_vault"),
)
DASHBOARD_FILE = os.path.join(VAULT_PATH, "output", "dashboard.md")
INPUT_FILE = os.path.join(VAULT_PATH, "input.md")

def read_task():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return f.read()

def write_status(status):
    os.makedirs(os.path.dirname(DASHBOARD_FILE), exist_ok=True)
    if isinstance(status, dict):
        content = f"# System Status\n\n```json\n{json.dumps(status, indent=2, default=str)}\n```\n"
    else:
        content = f"# System Status\n{status}"
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(content)
