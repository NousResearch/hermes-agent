"""Evolution Manager — 自动进化系统

每次工具调用后自动：
1. 记录执行结果到 self-model.db
2. 分析失败根因，学习反模式
3. 识别领域，自动适配
4. 更新情绪状态

使用方式：
    from agent.evolution_manager import record_tool_outcome
    
    # 在 tool_executor.py 中调用
    record_tool_outcome(agent, function_name, function_args, result, is_error, duration)
"""

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_evolution_dir() -> Path:
    """Get the evolution data directory."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(hermes_home) / "evolution"


def get_self_model_db() -> Path:
    """Get the self-model database path."""
    return get_evolution_dir() / "self-model.db"


def is_evolution_active() -> bool:
    """Check if evolution system is active."""
    return get_self_model_db().exists()


def classify_task(tool_name: str, args: Dict[str, Any]) -> str:
    """Classify task type from tool name and arguments."""
    if tool_name == "terminal":
        cmd = args.get("command", "").lower()
        if "git" in cmd:
            return "版本控制"
        elif "npm" in cmd or "yarn" in cmd:
            return "前端包管理"
        elif "pip" in cmd or "uv" in cmd:
            return "Python包管理"
        elif "docker" in cmd:
            return "容器化"
        elif "ssh" in cmd or "scp" in cmd:
            return "远程部署"
        elif "pytest" in cmd or "test" in cmd:
            return "测试"
        elif "build" in cmd or "compile" in cmd:
            return "构建"
        elif "curl" in cmd or "wget" in cmd:
            return "网络请求"
        else:
            return "终端命令"
    
    elif tool_name == "read_file":
        return "文件读取"
    
    elif tool_name == "write_file":
        return "文件写入"
    
    elif tool_name == "patch":
        return "代码修改"
    
    elif tool_name == "search_files":
        return "代码搜索"
    
    elif tool_name == "web_search":
        return "网络搜索"
    
    elif tool_name == "browser_navigate":
        return "浏览器操作"
    
    elif tool_name == "memory":
        action = args.get("action", "")
        return f"记忆管理:{action}"
    
    elif tool_name == "skill_manage":
        action = args.get("action", "")
        return f"技能管理:{action}"
    
    elif tool_name == "delegate_task":
        return "任务委派"
    
    else:
        return f"其他:{tool_name}"


def detect_domain(tool_name: str, args: Dict[str, Any]) -> str:
    """Detect domain from tool usage patterns."""
    if tool_name == "terminal":
        cmd = args.get("command", "").lower()
        if "git" in cmd:
            return "版本控制"
        elif any(x in cmd for x in ["npm", "yarn", "pnpm", "bun"]):
            return "前端开发"
        elif any(x in cmd for x in ["pip", "uv", "poetry", "conda"]):
            return "Python开发"
        elif any(x in cmd for x in ["docker", "podman", "k8s"]):
            return "容器化"
        elif any(x in cmd for x in ["ssh", "scp", "rsync"]):
            return "远程部署"
        elif any(x in cmd for x in ["pytest", "unittest", "test"]):
            return "测试"
        elif any(x in cmd for x in ["make", "cmake", "gradle", "mvn"]):
            return "构建系统"
        else:
            return "系统管理"
    
    elif tool_name in ["read_file", "write_file", "patch"]:
        path = args.get("path", "").lower()
        if path.endswith(".py"):
            return "Python开发"
        elif path.endswith((".js", ".ts", ".jsx", ".tsx")):
            return "前端开发"
        elif path.endswith((".go", ".rs", ".c", ".cpp", ".h")):
            return "系统编程"
        elif path.endswith((".md", ".txt", ".rst")):
            return "文档编写"
        elif path.endswith((".yaml", ".yml", ".json", ".toml")):
            return "配置管理"
        elif path.endswith((".sh", ".bash", ".zsh")):
            return "脚本编写"
        else:
            return "文件操作"
    
    elif tool_name == "web_search":
        return "网络研究"
    
    elif tool_name == "browser_navigate":
        return "浏览器操作"
    
    else:
        return "通用"


def extract_error_info(result: str) -> Tuple[str, str]:
    """Extract error information from tool result.
    
    Returns (error_type, error_message).
    """
    if not result:
        return "unknown", "Empty result"
    
    # Try to parse as JSON
    try:
        data = json.loads(result)
        if isinstance(data, dict):
            error = data.get("error", "")
            if error:
                return "api_error", str(error)[:200]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Check for common error patterns
    result_lower = result.lower()
    
    if "permission denied" in result_lower:
        return "permission_denied", result[:200]
    elif "not found" in result_lower or "no such file" in result_lower:
        return "not_found", result[:200]
    elif "timeout" in result_lower:
        return "timeout", result[:200]
    elif "connection refused" in result_lower:
        return "connection_refused", result[:200]
    elif "syntaxerror" in result_lower:
        return "syntax_error", result[:200]
    elif "importerror" in result_lower or "modulenotfounderror" in result_lower:
        return "import_error", result[:200]
    elif "typeerror" in result_lower:
        return "type_error", result[:200]
    elif "valueerror" in result_lower:
        return "value_error", result[:200]
    elif "attributeerror" in result_lower:
        return "attribute_error", result[:200]
    elif "keyerror" in result_lower:
        return "key_error", result[:200]
    elif "indexerror" in result_lower:
        return "index_error", result[:200]
    elif "filenotfounderror" in result_lower:
        return "file_not_found", result[:200]
    elif "isADirectoryError" in result_lower:
        return "is_a_directory", result[:200]
    elif "oserror" in result_lower or "ioerror" in result_lower:
        return "io_error", result[:200]
    elif "error" in result_lower or "failed" in result_lower:
        return "general_error", result[:200]
    else:
        return "unknown", result[:200]


def suggest_correction(tool_name: str, error_type: str, error_msg: str) -> str:
    """Suggest correction based on error type."""
    corrections = {
        "permission_denied": "检查文件权限，可能需要 sudo 或修改文件权限",
        "not_found": "检查路径是否正确，文件是否存在",
        "timeout": "检查网络连接，或增加超时时间",
        "connection_refused": "检查服务是否启动，端口是否正确",
        "syntax_error": "检查代码语法，特别是引号、括号、缩进",
        "import_error": "检查模块是否安装，路径是否正确",
        "type_error": "检查参数类型是否正确",
        "value_error": "检查参数值是否有效",
        "attribute_error": "检查对象是否有该属性",
        "key_error": "检查字典键是否存在",
        "index_error": "检查索引是否越界",
        "file_not_found": "检查文件路径是否正确",
        "is_a_directory": "目标路径是目录，不是文件",
        "io_error": "检查文件是否被占用，磁盘空间是否充足",
    }
    
    return corrections.get(error_type, "检查错误信息，分析根因")


def record_tool_outcome(
    agent,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: str,
    is_error: bool,
    duration: float,
) -> None:
    """Record tool execution outcome to self-model.db.
    
    This is called after each tool execution in tool_executor.py.
    """
    if not is_evolution_active():
        return
    
    try:
        # Classify task and domain
        task = classify_task(tool_name, tool_args)
        domain = detect_domain(tool_name, tool_args)
        
        # Extract error info if failed
        error_type = ""
        error_msg = ""
        correction = ""
        if is_error:
            error_type, error_msg = extract_error_info(result)
            correction = suggest_correction(tool_name, error_type, error_msg)
        
        # Record to database
        db_path = get_self_model_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Record outcome
        cursor.execute('''
            INSERT INTO outcomes (timestamp, task, action, tool, success, duration, domain, error_type, error_msg, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            task,
            str(tool_args)[:200],
            tool_name,
            0 if is_error else 1,
            duration,
            domain,
            error_type,
            error_msg,
            str(result)[:500]
        ))
        
        # If failed, check for anti-pattern
        if is_error and error_type:
            cursor.execute('''
                SELECT id, frequency FROM anti_patterns
                WHERE pattern = ? OR (pattern LIKE ? AND domain = ?)
            ''', (f"{tool_name}:{error_type}", f"%{error_type}%", domain))
            
            existing = cursor.fetchone()
            if existing:
                # Increment frequency
                cursor.execute('''
                    UPDATE anti_patterns
                    SET frequency = frequency + 1, last_seen = ?
                    WHERE id = ?
                ''', (timestamp, existing[0]))
            else:
                # Add new anti-pattern
                cursor.execute('''
                    INSERT INTO anti_patterns (timestamp, pattern, correct, domain, frequency, last_seen)
                    VALUES (?, ?, ?, ?, 1, ?)
                ''', (
                    timestamp,
                    f"{tool_name}:{error_type}",
                    correction,
                    domain,
                    timestamp
                ))
        
        conn.commit()
        conn.close()
        
        logger.debug(
            "Evolution: recorded %s %s (success=%s, duration=%.2fs)",
            tool_name, task, not is_error, duration
        )
        
    except Exception as e:
        logger.debug("Evolution recording failed: %s", e)


def get_strategy_advice(tool_name: str, domain: str) -> Optional[str]:
    """Get strategy advice based on historical data."""
    if not is_evolution_active():
        return None
    
    try:
        db_path = get_self_model_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get success rate for this tool+domain
        cursor.execute('''
            SELECT COUNT(*) as total, SUM(success) as successes
            FROM outcomes
            WHERE tool = ? AND domain = ?
        ''', (tool_name, domain))
        
        row = cursor.fetchone()
        if not row or row[0] == 0:
            conn.close()
            return None
        
        total, successes = row
        success_rate = (successes / total * 100) if total > 0 else 0
        
        # Get related anti-patterns
        cursor.execute('''
            SELECT pattern, correct, frequency
            FROM anti_patterns
            WHERE domain = ? OR domain = '通用'
            ORDER BY frequency DESC
            LIMIT 3
        ''', (domain,))
        
        anti_patterns = cursor.fetchall()
        
        conn.close()
        
        # Build advice
        advice_parts = []
        
        if success_rate < 50:
            advice_parts.append(f"⚠️ 历史成功率较低 ({success_rate:.0f}%)，建议谨慎操作")
        elif success_rate < 80:
            advice_parts.append(f"📊 历史成功率中等 ({success_rate:.0f}%)，建议验证结果")
        
        if anti_patterns:
            advice_parts.append("⚠️ 相关反模式:")
            for pattern, correct, freq in anti_patterns[:2]:
                advice_parts.append(f"  - {pattern} → {correct}")
        
        return "\n".join(advice_parts) if advice_parts else None
        
    except Exception as e:
        logger.debug("Evolution advice failed: %s", e)
        return None


def get_evolution_status() -> Dict[str, Any]:
    """Get current evolution system status."""
    if not is_evolution_active():
        return {"active": False}
    
    try:
        db_path = get_self_model_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total outcomes
        cursor.execute("SELECT COUNT(*) FROM outcomes")
        total = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute("SELECT COUNT(*) FROM outcomes WHERE success = 1")
        successes = cursor.fetchone()[0]
        success_rate = (successes / total * 100) if total > 0 else 0
        
        # Anti-patterns count
        cursor.execute("SELECT COUNT(*) FROM anti_patterns")
        anti_patterns_count = cursor.fetchone()[0]
        
        # Top domains
        cursor.execute('''
            SELECT domain, COUNT(*) as count
            FROM outcomes
            GROUP BY domain
            ORDER BY count DESC
            LIMIT 5
        ''')
        top_domains = cursor.fetchall()
        
        # Recent failures
        cursor.execute('''
            SELECT tool, error_type, COUNT(*) as count
            FROM outcomes
            WHERE success = 0
            GROUP BY tool, error_type
            ORDER BY count DESC
            LIMIT 5
        ''')
        recent_failures = cursor.fetchall()
        
        conn.close()
        
        return {
            "active": True,
            "total_outcomes": total,
            "success_rate": round(success_rate, 1),
            "anti_patterns_count": anti_patterns_count,
            "top_domains": top_domains,
            "recent_failures": recent_failures,
        }
        
    except Exception as e:
        logger.debug("Evolution status failed: %s", e)
        return {"active": True, "error": str(e)}
