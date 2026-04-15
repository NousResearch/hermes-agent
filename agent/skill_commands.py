"""Shared slash command helpers for skills and built-in prompt-style modes.

Shared between CLI (cli.py) and gateway (gateway/run.py) so both surfaces
can invoke skills via /skill-name commands and prompt-only built-ins like
/plan.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_skill_commands: Dict[str, Dict[str, Any]] = {}
_PLAN_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Patterns for sanitizing skill names into clean hyphen-separated slugs.
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")

_CATEGORY_ZH = {
    "apple": "Apple／macOS",
    "autonomous-ai-agents": "自主代理",
    "creative": "創意生成",
    "data-science": "資料科學",
    "devops": "DevOps",
    "email": "電子郵件",
    "gaming": "遊戲",
    "github": "GitHub",
    "leisure": "生活搜尋",
    "mcp": "MCP",
    "media": "媒體內容",
    "mlops": "MLOps",
    "note-taking": "知識整理",
    "openclaw-transfer": "OpenClaw 移轉",
    "productivity": "生產力",
    "red-teaming": "紅隊測試",
    "research": "研究",
    "smart-home": "智慧家庭",
    "social-media": "社群媒體",
    "software-development": "軟體開發",
}

_CATEGORY_DEFAULTS = {
    "apple": ("處理 Apple／macOS 裝置、訊息、提醒事項與系統操作", "你要我操作 Apple 生態服務、裝置或原生 App 時"),
    "autonomous-ai-agents": ("協調或啟動其他 AI 代理一起工作", "任務需要多代理分工、長時間自治或代理比較時"),
    "creative": ("生成設計、圖像、影片、互動視覺或創意內容", "你要我做視覺概念、創意發想或內容生成時"),
    "data-science": ("做資料探索、分析與 notebook 類工作", "你要我做資料分析、實驗或互動式探索時"),
    "devops": ("處理部署、webhook、通道切換與系統運維", "你要我做部署、切換、監控或基礎設施操作時"),
    "email": ("處理收信、寄信、搜尋與郵件整理", "任務主體是電子郵件往來時"),
    "gaming": ("處理遊戲、自動遊玩或遊戲伺服器相關工作", "你要我碰遊戲流程或遊戲環境時"),
    "github": ("處理 PR、review、issue、repo 與 CI 流程", "任務跟 GitHub 協作或 PR 推進直接相關時"),
    "leisure": ("找附近地點、店家或生活資訊", "你要我查附近餐廳、店面或生活目的地時"),
    "mcp": ("連接、設定或使用 MCP 伺服器與工具", "你要我接外部工具協定或整合 MCP 能力時"),
    "media": ("處理影音內容、轉錄、GIF、音樂或 YouTube 類素材", "你要我碰影音素材或內容轉換時"),
    "mlops": ("處理模型訓練、推論、量化、評測與模型工具鏈", "任務是模型工程、推論部署或訓練實驗時"),
    "note-taking": ("整理知識庫、Obsidian、wiki 與跨機器脈絡", "你要我整理知識、建立 wiki 或做長期脈絡管理時"),
    "openclaw-transfer": ("借用 OpenClaw donor 技能來補強規劃、設計、QA、瀏覽器與自治流程", "核心技能不夠用，或我要借 donor workflow 提高品質時"),
    "productivity": ("處理文件、試算表、Notion、Google Workspace 與營運資料", "你要我整理營運資訊、表單、文件或協作資料時"),
    "red-teaming": ("做模型越獄、攻防測試或安全對抗演練", "你明確要做紅隊或安全測試時"),
    "research": ("查論文、研究資料、追蹤來源與整理研究結論", "你要我做研究、找論文或追最新資料時"),
    "smart-home": ("控制智慧家庭設備", "你要我操作家中智慧裝置時"),
    "social-media": ("操作社群平台帳號與內容", "你要我發文、查貼文或管理社群互動時"),
    "software-development": ("處理規劃、實作、除錯、測試與交付的核心開發流程", "大多數寫程式、修 bug、重構、規劃任務都應優先想到它"),
}

_KEYWORD_HINTS = [
    ("slack", ("處理 Slack 溝通、thread、機器人或整合流程", "任務跟 Slack 主頻道、thread、通知或整合有關時")),
    ("telegram", ("處理 Telegram 機器人、webhook 或對話流程", "任務跟 Telegram bot、訊息流或 webhook 有關時")),
    ("github", ("處理 GitHub PR、review、issue 或 CI 工作", "你要我推進 repo 協作與 PR 時")),
    ("browser", ("操作瀏覽器、網站流程與頁面互動", "你要我實際打開網站、點擊或驗證介面時")),
    ("playwright", ("用自動化瀏覽器驗證網站或操作流程", "你要我做 web 自動化測試與回歸驗證時")),
    ("design", ("處理設計系統、視覺稿或 UI 品質", "你要我做設計探索、視覺評估或 UI 打磨時")),
    ("figma", ("讀取或實作 Figma 設計稿", "任務直接提到 Figma、節點或設計還原時")),
    ("qa", ("做 QA、找 bug、驗證體驗與回歸問題", "功能做好後要全面驗證時")),
    ("debug", ("定位錯誤、找 root cause 並收斂修復路徑", "測試失敗、功能異常或行為不符合預期時")),
    ("review", ("做設計審查、程式審查或計畫審查", "你要我先看方案品質、風險與缺口時")),
    ("plan", ("先把需求拆成可執行的規格與步驟", "任務很大、很多步或需要先想清楚再做時")),
    ("skill", ("整理、安裝或管理技能系統", "你要我處理技能本身、安裝包或技能策略時")),
    ("wiki", ("整理 wiki、知識圖譜與長期知識頁面", "你要我整理知識庫、頁面關聯與操作手冊時")),
    ("obsidian", ("操作 Obsidian vault 與知識筆記", "你要我直接整理或查找 Obsidian 內容時")),
    ("notion", ("操作 Notion 頁面、資料庫與知識整理", "任務主體在 Notion 時")),
    ("google", ("操作 Google Workspace 服務", "你要我碰 Gmail、Calendar、Drive、Sheets、Docs 時")),
    ("calendar", ("處理行事曆、排程與時程資料", "你要我查行程、重建業務節奏或整理日程時")),
    ("sheet", ("操作試算表、表格與欄位資料", "你要我整理 spreadsheet 或收入表時")),
    ("pdf", ("處理 PDF 讀取、編修、轉換與 OCR", "任務涉及 PDF 文件時")),
    ("docx", ("處理 Word 文件讀寫與排版", "任務涉及 Word 或正式文件交付時")),
    ("ppt", ("處理簡報與投影片內容", "你要我做簡報、讀簡報或改簡報時")),
    ("image", ("生成、編輯或分析圖片", "你要我做圖片生成、修圖或視覺理解時")),
    ("video", ("處理影片生成、轉換、視覺化或影片素材", "任務直接碰影片或動畫時")),
    ("audio", ("處理音訊、語音、音樂或轉錄素材", "你要我碰音檔、配音、轉錄或音樂時")),
    ("youtube", ("處理 YouTube 影片、逐字稿與內容整理", "你要我從 YouTube 拿內容或整理影片重點時")),
    ("research", ("做研究、蒐集資料與整理來源", "你要我查資料、比對觀點或做研究彙整時")),
    ("arxiv", ("搜尋與整理學術論文", "你要我找論文或追研究進展時")),
    ("security", ("做安全檢查、威脅建模或風險審查", "你要我做安全檢視與防護盤點時")),
    ("deploy", ("部署服務並處理上線流程", "你要我把東西 deploy 到正式或預覽環境時")),
    ("workflow", ("建立工作流、流程規格與自動化路徑", "你要我設計 SOP、agent flow 或營運流程時")),
]

_CURATED_SKILL_ZH_HINTS = {
    "gstack": "做什麼：Garry Tan 的 AI software factory 方法包，整合 office-hours、plan-review、review、qa、ship、browse 等完整 sprint 角色；適合：你要我用一整套高強度產品/工程流程來推大任務時",
    "thinking-hound-mode": "做什麼：來自 asterwei416/thinking-hound-mode 的工程紀律模式，會在 planner 與 execution 之間切節奏，先研究再下手；適合：你要我先想清楚、查最新官方資料、再動手做大功能或高風險改動時",
    "web-access": "做什麼：來自 eze-is/web-access 的完整上網技能，會在搜尋、抓取、CDP 瀏覽器操作之間選最有效通道，包含登入態與動態頁；適合：你要我真正上網查、讀、點、登入、抓內容或操作網站時",
    "playwright": "做什麼：基於 Playwright 的真實瀏覽器自動化與驗證，適合穩定重跑 flow、表單、截圖與回歸測試；適合：你要我把網站流程測到可重現、可驗證、可 script 化時",
    "dogfood": "做什麼：把產品當成真使用者去跑一遍，系統化找 bug、截證據、整理 QA 報告；適合：功能做完後要做體驗驗證、找缺陷、確認能不能交付時",
    "github-pr-workflow": "做什麼：把 PR 從 branch、commit、push、開 PR、追 CI 到 merge 整段管完；適合：你要我直接推 GitHub PR 流程而不是只做局部 git 操作時",
    "github-code-review": "做什麼：針對 diff / PR 做結構化 code review，找 correctness、安全性、測試缺口與可維護性問題；適合：你要我審 PR、審 diff、合併前把風險挑出來時",
    "systematic-debugging": "做什麼：用 root-cause-first 的方式 debug，先重現、讀錯誤、追資料流，再動手修；適合：bug、測試失敗、行為異常，而且不能靠猜的時候",
    "subagent-driven-development": "做什麼：把實作拆成多個子代理 lane 去並行完成，並帶 spec review / quality review；適合：任務大、可以切 lane，而且你要我用多代理同步推進時",
    "requesting-code-review": "做什麼：在 commit 或 push 前做驗證管線，包含安全掃描、測試、獨立 reviewer 與修正迴圈；適合：改完一批東西後要先驗證品質，再送出去時",
    "phase-based-autonomous-delivery": "做什麼：把長任務變成 phase ladder，一段做完就自動推下一段，不因小中斷停住；適合：你要我 one-shot 持續推進，不要每個 phase 都回頭問時",
    "parallel-architecture-review": "做什麼：把大型架構題拆成多角度平行審查，再收斂成一份 canonical blueprint；適合：重建系統、重做 workflow、做 operating model 時",
    "self-improving-agent": "做什麼：來自 peterskoett/self-improving-agent 的自我進化方法，重點是從每次任務提煉 learnings、錯誤、模式與修正，再回寫成可持續改進的系統；適合：你要我建立 Hermes 自我進化迴圈、方法、記錄與長期演進排程時",
    "boss-mode-mobile-decision-ui": "做什麼：把充滿內部術語的工程後台翻成老闆看得懂、手機上可決策的 UI；適合：你要我把 agent / approval / project 介面改成 boss-mode 時",
    "hermes-dashboard-project-control-plane": "做什麼：把 Hermes 9119 dashboard 從狀態頁升級成真正的專案控制台，能看到 project、approval、autopilot 與推進狀態；適合：你要我把 Hermes 前台變成可控的 project OS 時",
    "slack-mcp-boss-mode-control-layer": "做什麼：把 Slack 變成 Hermes 的 boss-mode 控制層，用主頻道交辦、thread 執行、摘要回主頻道；適合：你要把 Slack 當成主要管理介面時",
    "hermes-agent-web-dashboard": "做什麼：啟動、驗證與延伸 Hermes 內建 web dashboard，處理 build、health check、臨時曝光與介面驗證；適合：你要我直接把 9119 dashboard 跑起來、測起來、改起來時",
}


def _category_zh(category: str) -> str:
    return _CATEGORY_ZH.get(category, category or "未分類")


def _extract_skill_category(skill_md: Path, scan_dir: Path) -> str:
    try:
        rel = skill_md.parent.relative_to(scan_dir)
    except Exception:
        return ""
    return rel.parts[0] if len(rel.parts) >= 2 else ""


def _clean_description_for_hint(description: str) -> str:
    text = re.sub(r"\s+", " ", description).strip()
    text = re.sub(r"^[|>\-\s]+", "", text)
    text = re.sub(r"^use when\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^use this skill\s+", "", text, flags=re.IGNORECASE)
    return text


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _keyword_hint(description: str) -> tuple[str, str] | None:
    lower = description.lower()
    for keyword, hint in _KEYWORD_HINTS:
        if keyword in lower:
            return hint
    return None


def _build_generated_skill_zh_description(name: str, category: str, description: str) -> str:
    cleaned = _clean_description_for_hint(description)
    if _contains_cjk(cleaned):
        short = cleaned[:46] + ("..." if len(cleaned) > 46 else "")
        return f"做什麼：{short}；適合：任務明確對到這個技能名稱或描述時"

    hint = _keyword_hint(cleaned)
    if hint:
        does_what, use_when = hint
    else:
        does_what, use_when = _CATEGORY_DEFAULTS.get(
            category,
            (f"處理{_category_zh(category)}相關工作", "任務跟這個分類直接匹配時"),
        )
    return f"做什麼：{does_what}；適合：{use_when}"


def _extract_skill_zh_description(frontmatter: dict[str, Any], name: str, category: str, description: str) -> str:
    curated = _CURATED_SKILL_ZH_HINTS.get(name)
    if curated:
        return curated

    zh_description = frontmatter.get("zh_description")
    if isinstance(zh_description, str) and zh_description.strip():
        return zh_description.strip()

    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        hermes_meta = metadata.get("hermes")
        if isinstance(hermes_meta, dict):
            zh_description = hermes_meta.get("zh_description")
            if isinstance(zh_description, str) and zh_description.strip():
                return zh_description.strip()

    return _build_generated_skill_zh_description(name, category, description)


def build_plan_path(
    user_instruction: str = "",
    *,
    now: datetime | None = None,
) -> Path:
    """Return the default workspace-relative markdown path for a /plan invocation.

    Relative paths are intentional: file tools are task/backend-aware and resolve
    them against the active working directory for local, docker, ssh, modal,
    daytona, and similar terminal backends. That keeps the plan with the active
    workspace instead of the Hermes host's global home directory.
    """
    slug_source = (user_instruction or "").strip().splitlines()[0] if user_instruction else ""
    slug = _PLAN_SLUG_RE.sub("-", slug_source.lower()).strip("-")
    if slug:
        slug = "-".join(part for part in slug.split("-")[:8] if part)[:48].strip("-")
    slug = slug or "conversation-plan"
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return Path(".hermes") / "plans" / f"{timestamp}-{slug}.md"


def _load_skill_payload(skill_identifier: str, task_id: str | None = None) -> tuple[dict[str, Any], Path | None, str] | None:
    """Load a skill by name/path and return (loaded_payload, skill_dir, display_name)."""
    raw_identifier = (skill_identifier or "").strip()
    if not raw_identifier:
        return None

    try:
        from tools.skills_tool import SKILLS_DIR, skill_view

        identifier_path = Path(raw_identifier).expanduser()
        if identifier_path.is_absolute():
            try:
                normalized = str(identifier_path.resolve().relative_to(SKILLS_DIR.resolve()))
            except Exception:
                normalized = raw_identifier
        else:
            normalized = raw_identifier.lstrip("/")

        loaded_skill = json.loads(skill_view(normalized, task_id=task_id))
    except Exception:
        return None

    if not loaded_skill.get("success"):
        return None

    skill_name = str(loaded_skill.get("name") or normalized)
    skill_path = str(loaded_skill.get("path") or "")
    skill_dir = None
    if skill_path:
        try:
            skill_dir = SKILLS_DIR / Path(skill_path).parent
        except Exception:
            skill_dir = None

    return loaded_skill, skill_dir, skill_name


def _inject_skill_config(loaded_skill: dict[str, Any], parts: list[str]) -> None:
    """Resolve and inject skill-declared config values into the message parts.

    If the loaded skill's frontmatter declares ``metadata.hermes.config``
    entries, their current values (from config.yaml or defaults) are appended
    as a ``[Skill config: ...]`` block so the agent knows the configured values
    without needing to read config.yaml itself.
    """
    try:
        from agent.skill_utils import (
            extract_skill_config_vars,
            parse_frontmatter,
            resolve_skill_config_values,
        )

        # The loaded_skill dict contains the raw content which includes frontmatter
        raw_content = str(loaded_skill.get("raw_content") or loaded_skill.get("content") or "")
        if not raw_content:
            return

        frontmatter, _ = parse_frontmatter(raw_content)
        config_vars = extract_skill_config_vars(frontmatter)
        if not config_vars:
            return

        resolved = resolve_skill_config_values(config_vars)
        if not resolved:
            return

        lines = ["", "[Skill config (from ~/.hermes/config.yaml):"]
        for key, value in resolved.items():
            display_val = str(value) if value else "(not set)"
            lines.append(f"  {key} = {display_val}")
        lines.append("]")
        parts.extend(lines)
    except Exception:
        pass  # Non-critical — skill still loads without config injection


def _build_skill_message(
    loaded_skill: dict[str, Any],
    skill_dir: Path | None,
    activation_note: str,
    user_instruction: str = "",
    runtime_note: str = "",
) -> str:
    """Format a loaded skill into a user/system message payload."""
    from tools.skills_tool import SKILLS_DIR

    content = str(loaded_skill.get("content") or "")

    parts = [activation_note, "", content.strip()]

    # ── Inject resolved skill config values ──
    _inject_skill_config(loaded_skill, parts)

    if loaded_skill.get("setup_skipped"):
        parts.extend(
            [
                "",
                "[Skill setup note: Required environment setup was skipped. Continue loading the skill and explain any reduced functionality if it matters.]",
            ]
        )
    elif loaded_skill.get("gateway_setup_hint"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['gateway_setup_hint']}]",
            ]
        )
    elif loaded_skill.get("setup_needed") and loaded_skill.get("setup_note"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['setup_note']}]",
            ]
        )

    supporting = []
    linked_files = loaded_skill.get("linked_files") or {}
    for entries in linked_files.values():
        if isinstance(entries, list):
            supporting.extend(entries)

    if not supporting and skill_dir:
        for subdir in ("references", "templates", "scripts", "assets"):
            subdir_path = skill_dir / subdir
            if subdir_path.exists():
                for f in sorted(subdir_path.rglob("*")):
                    if f.is_file() and not f.is_symlink():
                        rel = str(f.relative_to(skill_dir))
                        supporting.append(rel)

    if supporting and skill_dir:
        try:
            skill_view_target = str(skill_dir.relative_to(SKILLS_DIR))
        except ValueError:
            # Skill is from an external dir — use the skill name instead
            skill_view_target = skill_dir.name
        parts.append("")
        parts.append("[This skill has supporting files you can load with the skill_view tool:]")
        for sf in supporting:
            parts.append(f"- {sf}")
        parts.append(
            f'\nTo view any of these, use: skill_view(name="{skill_view_target}", file_path="<path>")'
        )

    if user_instruction:
        parts.append("")
        parts.append(f"The user has provided the following instruction alongside the skill invocation: {user_instruction}")

    if runtime_note:
        parts.append("")
        parts.append(f"[Runtime note: {runtime_note}]")

    return "\n".join(parts)


def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan ~/.hermes/skills/ and return a mapping of /command -> skill info.

    Returns:
        Dict mapping "/skill-name" to {name, description, skill_md_path, skill_dir}.
    """
    global _skill_commands
    _skill_commands = {}
    try:
        from tools.skills_tool import SKILLS_DIR, _parse_frontmatter, skill_matches_platform, _get_disabled_skill_names
        from agent.skill_utils import get_external_skills_dirs
        disabled = _get_disabled_skill_names()
        seen_names: set = set()

        # Scan local dir first, then external dirs
        dirs_to_scan = []
        if SKILLS_DIR.exists():
            dirs_to_scan.append(SKILLS_DIR)
        dirs_to_scan.extend(get_external_skills_dirs())

        for scan_dir in dirs_to_scan:
            for skill_md in scan_dir.rglob("SKILL.md"):
                if any(part in ('.git', '.github', '.hub') for part in skill_md.parts):
                    continue
                try:
                    content = skill_md.read_text(encoding='utf-8')
                    frontmatter, body = _parse_frontmatter(content)
                    # Skip skills incompatible with the current OS platform
                    if not skill_matches_platform(frontmatter):
                        continue
                    name = frontmatter.get('name', skill_md.parent.name)
                    if name in seen_names:
                        continue
                    # Respect user's disabled skills config
                    if name in disabled:
                        continue
                    description = frontmatter.get('description', '')
                    if not description:
                        for line in body.strip().split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                description = line[:80]
                                break
                    category = _extract_skill_category(skill_md, scan_dir)
                    zh_description = _extract_skill_zh_description(frontmatter, name, category, description)
                    seen_names.add(name)
                    # Normalize to hyphen-separated slug, stripping
                    # non-alnum chars (e.g. +, /) to avoid invalid
                    # Telegram command names downstream.
                    cmd_name = name.lower().replace(' ', '-').replace('_', '-')
                    cmd_name = _SKILL_INVALID_CHARS.sub('', cmd_name)
                    cmd_name = _SKILL_MULTI_HYPHEN.sub('-', cmd_name).strip('-')
                    if not cmd_name:
                        continue
                    _skill_commands[f"/{cmd_name}"] = {
                        "name": name,
                        "description": description or f"Invoke the {name} skill",
                        "zh_description": zh_description,
                        "category": category,
                        "skill_md_path": str(skill_md),
                        "skill_dir": str(skill_md.parent),
                    }
                except Exception:
                    continue
    except Exception:
        pass
    return _skill_commands


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Return the current skill commands mapping (scan first if empty)."""
    if not _skill_commands:
        scan_skill_commands()
    return _skill_commands


def resolve_skill_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed /command to its canonical skill_cmds key.

    Skills are always stored with hyphens — ``scan_skill_commands`` normalizes
    spaces and underscores to hyphens when building the key. Hyphens and
    underscores are treated interchangeably in user input: this matches
    ``_check_unavailable_skill`` and accommodates Telegram bot-command names
    (which disallow hyphens, so ``/claude-code`` is registered as
    ``/claude_code`` and comes back in the underscored form).

    Returns the matching ``/slug`` key from ``get_skill_commands()`` or
    ``None`` if no match.
    """
    if not command:
        return None
    cmd_key = f"/{command.replace('_', '-')}"
    return cmd_key if cmd_key in get_skill_commands() else None


def build_skill_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
    runtime_note: str = "",
) -> Optional[str]:
    """Build the user message content for a skill slash command invocation.

    Args:
        cmd_key: The command key including leading slash (e.g., "/gif-search").
        user_instruction: Optional text the user typed after the command.

    Returns:
        The formatted message string, or None if the skill wasn't found.
    """
    commands = get_skill_commands()
    skill_info = commands.get(cmd_key)
    if not skill_info:
        return None

    loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id)
    if not loaded:
        return f"[Failed to load skill: {skill_info['name']}]"

    loaded_skill, skill_dir, skill_name = loaded
    activation_note = (
        f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want '
        "you to follow its instructions. The full skill content is loaded below.]"
    )
    return _build_skill_message(
        loaded_skill,
        skill_dir,
        activation_note,
        user_instruction=user_instruction,
        runtime_note=runtime_note,
    )


def build_preloaded_skills_prompt(
    skill_identifiers: list[str],
    task_id: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Load one or more skills for session-wide CLI preloading.

    Returns (prompt_text, loaded_skill_names, missing_identifiers).
    """
    prompt_parts: list[str] = []
    loaded_names: list[str] = []
    missing: list[str] = []

    seen: set[str] = set()
    for raw_identifier in skill_identifiers:
        identifier = (raw_identifier or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)

        loaded = _load_skill_payload(identifier, task_id=task_id)
        if not loaded:
            missing.append(identifier)
            continue

        loaded_skill, skill_dir, skill_name = loaded
        activation_note = (
            f'[SYSTEM: The user launched this CLI session with the "{skill_name}" skill '
            "preloaded. Treat its instructions as active guidance for the duration of this "
            "session unless the user overrides them.]"
        )
        prompt_parts.append(
            _build_skill_message(
                loaded_skill,
                skill_dir,
                activation_note,
            )
        )
        loaded_names.append(skill_name)

    return "\n\n".join(prompt_parts), loaded_names, missing
