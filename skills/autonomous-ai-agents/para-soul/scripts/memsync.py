#!/usr/bin/env python3
"""Para-Soul MemSync — Sync agent memory + distill long-term memory.

Reads:
  1. Hermes memory (MEMORY.md, USER.md) → memory.md
  2. Instruction files from known directories → memory.md
  3. Skills inventory from ~/.hermes/skills/ → skills.json
  4. growth-log entries >14 days old → LLM distillation → long-term-memory.md

No pip dependencies beyond Python stdlib (+ requests for LLM distillation).
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

PARA_HOME = Path(os.environ.get("PARA_HOME", Path.home() / ".para"))
HERMES_MEMORIES = Path.home() / ".hermes" / "memories"

# ── Source 1: Hermes memory files ───────────────────

def read_hermes_memories():
    """Read MEMORY.md and USER.md from ~/.hermes/memories/"""
    entries = []
    for fname in ["MEMORY.md", "USER.md"]:
        fp = HERMES_MEMORIES / fname
        if fp.exists():
            content = fp.read_text(encoding='utf-8')
            items = [s.strip() for s in content.split("§") if s.strip()]
            entries.append((fname, items))
    return entries


# ── Source 2: Instruction files ──────────────────────

INSTRUCTION_FILES = [
    "CLAUDE.md", "AGENTS.md", ".cursorrules", ".windsurfrules",
    ".clinerules", ".roorules", "CODEBUDDY.md",
    ".github/copilot-instructions.md", "COPILOT.md", "CONVENTIONS.md",
]

def scan_instruction_files():
    """Scan known directories for agent instruction files."""
    found = {}
    search_dirs = [
        Path.cwd(), Path.home(),
        Path("/mnt/d/边飞"), Path("/mnt/d/边飞/Paragate"),
        Path("/mnt/d/边飞/HERMES 芳疗"),
    ]
    for base in search_dirs:
        if not base.exists(): continue
        d = base
        while d != d.parent:
            for fname in INSTRUCTION_FILES:
                fp = d / fname
                if fp.exists() and str(fp) not in found:
                    try:
                        content = fp.read_text(encoding='utf-8')[:5000]
                        if len(content.strip()) > 50:
                            found[str(fp)] = content[:3000]
                    except Exception: pass
            d = d.parent
    return found


# ── Source 3: Skills inventory ───────────────────────

def scan_skills():
    """Scan ~/.hermes/skills/ for installed skills."""
    skills_dir = Path.home() / ".hermes" / "skills"
    if not skills_dir.exists():
        return {"installed": [], "favorites": [], "wishlist": [], "deprecated": []}

    installed = []
    for cat_dir in skills_dir.iterdir():
        if cat_dir.is_dir() and not cat_dir.name.startswith("."):
            for skill_dir in cat_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    installed.append(f"{cat_dir.name}/{skill_dir.name}")

    return {
        "installed": sorted(installed),
        "favorites": [],
        "wishlist": [],
        "deprecated": [],
    }


# ── Merge and write memory.md ────────────────────────

def build_memory_md(hermes, instructions):
    lines = ["# Memory", "", f"Auto-synced: {datetime.now().isoformat()[:19]}", ""]
    
    for fname, entries in hermes:
        lines.append(f"## {fname}")
        for entry in entries:
            lines.append(f"{entry}")
            lines.append("")
        lines.append("")
    
    if instructions:
        lines.append("## Agent Instructions")
        for path, content in instructions.items():
            fname = Path(path).name
            lines.append(f"### {fname} ({path})")
            for line in content.split("\n"):
                line = line.strip()
                if line and len(line) > 10:
                    lines.append(f"- {line[:200]}")
            lines.append("")
    
    return "\n".join(lines)


# ── Source 4: growth-log → long-term-memory distillation ──

def read_growth_log_entries(older_than_days=14):
    """Read growth-log entries older than N days."""
    log_dir = PARA_HOME / "growth-log"
    if not log_dir.is_dir():
        return []

    cutoff = datetime.now() - timedelta(days=older_than_days)
    old_entries = []

    for lf in sorted(log_dir.glob("*.md")):
        content = lf.read_text()
        sections = content.split("\n## ")
        for section in sections[1:]:  # skip header
            lines = section.strip().split("\n")
            if lines:
                # Parse date from first line (## YYYY-MM-DD)
                date_str = lines[0].strip()
                try:
                    entry_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if entry_date < cutoff:
                        entry_text = "\n".join(lines[1:]).strip()
                        if entry_text:
                            old_entries.append({
                                "date": date_str,
                                "text": entry_text[:300]
                            })
                except ValueError:
                    pass

    return old_entries


def distill_to_long_term(old_entries):
    """Use LLM to distill old growth-log entries into milestones."""
    if not old_entries:
        return None

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("AUXILIARY_VISION_API_KEY")
    if not api_key:
        print("  ⚠️  No API key for LLM distillation. Skipping long-term-memory update.")
        return None

    # Build prompt
    entry_text = ""
    for e in old_entries:
        entry_text += f"## {e['date']}\n{e['text']}\n\n"

    current_ltm = ""
    ltm_path = PARA_HOME / "long-term-memory.md"
    if ltm_path.exists():
        current_ltm = ltm_path.read_text()[-2000:]

    try:
        import requests
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "qwen-plus",
                "input": {"messages": [
                    {"role": "system", "content": "你是记忆蒸馏分析师。从growth-log中提取里程碑事件，追加到long-term-memory。每条里程碑格式：'### YYYY-MM-DD 范围' + 一句话摘要。只输出新增的里程碑，不要重复已有的。保持简洁，每条约30-50字。中文输出。"},
                    {"role": "user", "content": f"已有长期记忆:\n{current_ltm}\n\n待蒸馏的growth-log（超过14天未处理的条目）:\n{entry_text}"}
                ]},
                "parameters": {"result_format": "message"}
            }, timeout=60
        )
        result = resp.json()["output"]["choices"][0]["message"]["content"]
        return result
    except Exception as e:
        print(f"  ⚠️  LLM distillation failed: {e}")
        return None


def append_long_term_memory(milestones):
    """Append distilled milestones to long-term-memory.md."""
    if not milestones:
        return

    ltm_path = PARA_HOME / "long-term-memory.md"
    existing = ""
    if ltm_path.exists():
        existing = ltm_path.read_text()

    # Only append if milestones contain actual content (not just headers)
    if len(milestones.strip()) > 20:
        new_content = existing.rstrip() + "\n\n" + milestones + "\n"
        ltm_path.write_text(new_content)
        print(f"  ✅ long-term-memory.md updated with distilled milestones")
    else:
        print(f"  ℹ️  No new milestones to distill")


# ── Main ─────────────────────────────────────────────

def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] MemSync")

    # Step 1: Sync memory.md
    hermes = read_hermes_memories()
    instructions = scan_instruction_files()
    memory_content = build_memory_md(hermes, instructions)
    (PARA_HOME / "memory.md").write_text(memory_content, encoding='utf-8')

    hermes_count = sum(len(e) for _, e in hermes)
    print(f"  Hermes files: {len(hermes)} ({hermes_count} entries)")
    print(f"  Instruction files: {len(instructions)}")
    print(f"  ✅ memory.md ({len(memory_content)} chars)")

    # Step 2: Sync skills.json
    skills = scan_skills()
    (PARA_HOME / "skills.json").write_text(json.dumps(skills, indent=2, ensure_ascii=False))
    print(f"  ✅ skills.json ({len(skills['installed'])} skills)")

    # Step 3: Distill growth-log → long-term-memory (entries >14 days)
    old_entries = read_growth_log_entries(older_than_days=14)
    if old_entries:
        print(f"  Found {len(old_entries)} growth-log entries older than 14 days")
        milestones = distill_to_long_term(old_entries)
        if milestones:
            append_long_term_memory(milestones)
    else:
        print(f"  No growth-log entries older than 14 days to distill")

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] MemSync done")


if __name__ == "__main__":
    main()
