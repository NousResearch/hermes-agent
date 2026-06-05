"""Agent Evolution Plugin — 理性(数据驱动) + 感性(涌现认知) 融合架构

让任何用户一句话就能让agent获得自我进化能力。

功能：
1. 理性层：成功率追踪、反模式库、策略自动调整
2. 感性层：情绪状态追踪、融合决策、进化报告
3. 分层记忆：core/active/archive 三层智能记忆管理
4. 一键部署：自动初始化数据库和身份文件

使用方式：
    用户只需一句话：
    "下载这个包并部署进化系统：https://vbit.top/vermes/downloads/agent-evolution-package.tar.gz"

    Vermes 会自己：
    1. 下载包
    2. 解压
    3. 运行 init.sh
    4. 根据自己的领域添加反模式
    5. 开始记录执行结果和情绪信号
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------

PLUGIN_NAME = "agent-evolution"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Agent 进化系统 — 理性(数据驱动) + 感性(涌现认知) 融合架构"


# ---------------------------------------------------------------------------
# Plugin initialization
# ---------------------------------------------------------------------------

def get_plugin_dir() -> Path:
    """Get the plugin directory."""
    return Path(__file__).parent


def get_evolution_dir() -> Path:
    """Get the evolution data directory."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(hermes_home) / "evolution"


def is_initialized() -> bool:
    """Check if evolution system is initialized."""
    evolution_dir = get_evolution_dir()
    return (
        (evolution_dir / "self-model.db").exists()
        and (evolution_dir / "fusion-state.db").exists()
    )


def initialize(role_name: str = "通用助手") -> Dict[str, Any]:
    """Initialize the evolution system.
    
    Args:
        role_name: The role name for the agent (e.g., "超级交易员", "工程师")
        
    Returns:
        dict with initialization result
    """
    plugin_dir = get_plugin_dir()
    init_script = plugin_dir / "scripts" / "init.sh"
    
    if not init_script.exists():
        return {
            "success": False,
            "error": f"Init script not found: {init_script}"
        }
    
    try:
        # Run init script
        result = subprocess.run(
            ["bash", str(init_script), role_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "message": f"Evolution system initialized for role: {role_name}",
                "output": result.stdout
            }
        else:
            return {
                "success": False,
                "error": f"Init script failed: {result.stderr}",
                "output": result.stdout
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Init script timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to run init script: {e}"
        }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return tool schemas for this plugin."""
    return [
        {
            "name": "evolution_status",
            "description": "查看进化系统状态",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "evolution_report",
            "description": "生成进化报告",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "报告天数（默认7天）",
                        "default": 7
                    }
                },
                "required": []
            }
        },
        {
            "name": "evolution_record",
            "description": "记录执行结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "任务类型"
                    },
                    "action": {
                        "type": "string",
                        "description": "执行动作"
                    },
                    "tool": {
                        "type": "string",
                        "description": "使用的工具"
                    },
                    "success": {
                        "type": "integer",
                        "description": "是否成功（0或1）",
                        "enum": [0, 1]
                    }
                },
                "required": ["task", "action", "tool", "success"]
            }
        }
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Handle tool calls for this plugin."""
    import json
    
    if tool_name == "evolution_status":
        return _handle_evolution_status()
    elif tool_name == "evolution_report":
        days = tool_args.get("days", 7)
        return _handle_evolution_report(days)
    elif tool_name == "evolution_record":
        return _handle_evolution_record(tool_args)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _handle_evolution_status() -> str:
    """Handle evolution_status tool call."""
    import json
    
    evolution_dir = get_evolution_dir()
    plugin_dir = get_plugin_dir()
    
    status = {
        "initialized": is_initialized(),
        "evolution_dir": str(evolution_dir),
        "plugin_dir": str(plugin_dir),
    }
    
    if is_initialized():
        # Check self-model.db
        self_model_db = evolution_dir / "self-model.db"
        if self_model_db.exists():
            status["self_model_db"] = {
                "exists": True,
                "size": self_model_db.stat().st_size
            }
        
        # Check fusion-state.db
        fusion_state_db = evolution_dir / "fusion-state.db"
        if fusion_state_db.exists():
            status["fusion_state_db"] = {
                "exists": True,
                "size": fusion_state_db.stat().st_size
            }
        
        # Check SOUL.md
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        soul_md = Path(hermes_home) / "SOUL.md"
        if soul_md.exists():
            status["soul_md"] = {
                "exists": True,
                "size": soul_md.stat().st_size
            }
    
    return json.dumps(status, indent=2, ensure_ascii=False)


def _handle_evolution_report(days: int) -> str:
    """Handle evolution_report tool call."""
    import json
    
    plugin_dir = get_plugin_dir()
    fusion_engine = plugin_dir / "scripts" / "fusion_engine.py"
    
    if not fusion_engine.exists():
        return json.dumps({"error": "fusion_engine.py not found"})
    
    try:
        result = subprocess.run(
            [sys.executable, str(fusion_engine), "report", "--days", str(days)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return json.dumps({"error": result.stderr})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_evolution_record(tool_args: Dict[str, Any]) -> str:
    """Handle evolution_record tool call."""
    import json
    
    plugin_dir = get_plugin_dir()
    self_model = plugin_dir / "scripts" / "self_model.py"
    
    if not self_model.exists():
        return json.dumps({"error": "self_model.py not found"})
    
    task = tool_args.get("task", "")
    action = tool_args.get("action", "")
    tool = tool_args.get("tool", "")
    success = tool_args.get("success", 0)
    
    try:
        result = subprocess.run(
            [
                sys.executable, str(self_model), "record",
                "--task", task,
                "--action", action,
                "--tool", tool,
                "--success", str(success)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return json.dumps({"error": result.stderr})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# System prompt block
# ---------------------------------------------------------------------------

def system_prompt_block() -> str:
    """Return system prompt block for this plugin."""
    if not is_initialized():
        return ""
    
    return """## 进化系统

你的进化系统已激活，包含：
- **理性层**: self-model.db — 追踪成功率、反模式、策略
- **感性层**: fusion-state.db — 情绪状态、融合决策
- **分层记忆**: core/active/archive — 智能记忆管理

使用方法：
- 记录执行结果: `evolution_record`
- 查看自我认知: `evolution_status`
- 生成进化报告: `evolution_report`

分层记忆：
- `memory add "## [core] 永久核心事实"`
- `memory add "## [active] 当前活跃记忆"`
- `memory add "## [archive] 低频归档记忆"`
"""


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx):
    """Register this plugin with Hermes."""
    ctx.register_tools(
        tool_schemas=get_tool_schemas(),
        handler=handle_tool_call,
    )
    
    logger.info("Agent Evolution plugin registered")
