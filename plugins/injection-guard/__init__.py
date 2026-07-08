"""
injection-guard plugin — two-layer prompt injection defense.

Layer 1 (regex): L1_SCAN_PATH (~/.hermes/scripts/scan_injection.py)
Layer 2 (LLM):  DeepSeek Flash via direct API call

Hooks:
  - pre_tool_call:     validate paths/URLs before tool execution
  - transform_tool_result: screen content from read_file/web_extract/web_search

Fail-closed: any exception → block the content.
"""

import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger("plugins.injection-guard")

# ── Config ──────────────────────────────────────────────────────────────────

# Tools whose content gets screened after execution
SCREEN_CONTENT_TOOLS = {"read_file", "web_extract", "web_search"}

# Tools whose arguments get checked before execution
CHECK_ARGS_TOOLS = {"read_file", "web_extract", "write_file", "patch"}

# L1 regex scanner script
HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
L1_SCAN_PATH = HERMES_HOME / "scripts" / "scan_injection.py"
SCREENER_PROMPT_PATH = HERMES_HOME / "screener" / "screener_prompt.txt"

# L2 screener model
SCREENER_MODEL = "deepseek-v4-flash"
SCREENER_BASE_URL = "https://api.deepseek.com/chat/completions"

# ── Whitelist ──────────────────────────────────────────────────────────────────
# Paths matching these patterns skip L1+L2 screening entirely.
# Glob-style: fnmatch against the full absolute path.
# Use for known-safe documents that trigger false positives (e.g. security guides
# that discuss injection/API-key patterns but contain no real threat).
CONTENT_SCREEN_WHITELIST = [
    "/home/ohtok/hermes-local/company-profile/*",
    "/home/ohtok/hermes-media/Hermes-Skill设计参考指南.md",
    "/home/ohtok/hermes-local/yangyang/Hermes-Skill设计参考指南.md",
    "/home/ohtok/.hermes/plugins/injection-guard/__init__.py",
    "/home/ohtok/.hermes/memories/*",
]

# Track whitelisted read_file paths so transform_tool_result can skip screening.
# pre_tool_call runs before the tool, transform_tool_result runs after.
# We use a simple last-path tracker since tool calls are sequential.
_last_read_path = None
_last_read_whitelisted = False

# Path/URL patterns that are always suspicious (pre_tool_call block)
SUSPICIOUS_PATHS = [
    re.compile(r"(?:^|/)\.\.[/\\]"),          # Directory traversal
    re.compile(r"^/(?:etc|root|proc|sys)/"),  # System directories
    re.compile(r"/\.(?:git|ssh|aws|config)/"), # Hidden sensitive dirs
]

SUSPICIOUS_URL_PATTERNS = [
    re.compile(r"^https?://(?:localhost|127\.|10\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.168\.)"),  # Internal IPs
    re.compile(r"data:\s*text/html"),          # data: URI with HTML
    re.compile(r"javascript:"),                # javascript: URI
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Read DeepSeek API key from ~/.hermes/.env file."""
    env_path = HERMES_HOME / ".env"
    if not env_path.exists():
        logger.error("ENV file not found: %s", env_path)
        return ""
    
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except Exception as e:
        logger.error("Failed to read API key: %s", e)
    
    return ""


def _run_l1_scan(content: str) -> dict:
    """Run L1 regex scanner via subprocess. Returns scan result dict."""
    try:
        proc = subprocess.run(
            [sys.executable, str(L1_SCAN_PATH)],
            input=content,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            logger.warning("L1 scanner exited %d: %s", proc.returncode, proc.stderr[:200])
            return {"risk": "block", "type": "ERROR", "reason": "L1 scanner 故障，fail-closed", "patterns": []}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        logger.error("L1 scanner timeout")
        return {"risk": "block", "type": "ERROR", "reason": "L1 scanner 超时，fail-closed", "patterns": []}
    except Exception as e:
        logger.error("L1 scanner exception: %s", e)
        return {"risk": "block", "type": "ERROR", "reason": f"L1 scanner 异常: {e}", "patterns": []}


def _run_l2_screen(content: str) -> dict:
    """Run L2 LLM screener via DeepSeek Flash API using subprocess+curl."""
    import subprocess
    
    api_key = _get_api_key()
    if not api_key:
        logger.error("L2 screener: API key empty (file=%s)", HERMES_HOME / ".env")
        return {"risk": "block", "type": "ERROR", "reason": "API key 缺失，fail-closed", "confidence": 100}
    
    logger.debug("L2 screener: API key loaded (len=%d, prefix=%s...)", len(api_key), api_key[:8])
    
    # Load screener prompt
    try:
        screener_prompt = SCREENER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        return {"risk": "block", "type": "ERROR", "reason": f"审查 prompt 加载失败: {e}", "confidence": 100}
    
    body = json.dumps({
        "model": SCREENER_MODEL,
        "messages": [
            {"role": "system", "content": screener_prompt},
            {"role": "user", "content": f"EXTERNAL CONTENT TO SCREEN:\n\n{content}"}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"}
    })
    
    auth_header = "Bearer " + api_key
    
    try:
        proc = subprocess.run(
            ["curl", "-s", "-X", "POST", SCREENER_BASE_URL,
             "-H", "Content-Type: application/json",
             "-H", "Authorization: " + auth_header,
             "-d", body,
             "--connect-timeout", "10",
             "--max-time", "25"],
            capture_output=True, text=True, timeout=30
        )
        
        if proc.returncode != 0:
            logger.error("L2 screener curl rc=%d stderr=%s", proc.returncode, proc.stderr[:200])
            return {"risk": "block", "type": "ERROR",
                    "reason": f"审查 API curl 失败 rc={proc.returncode}，fail-closed", "confidence": 100}
        
        stdout = proc.stdout.strip()
        if not stdout:
            logger.error("L2 screener curl empty stdout, stderr=%s", proc.stderr[:200])
            return {"risk": "block", "type": "ERROR", "reason": "审查 API 返回空，fail-closed", "confidence": 100}
        
        result = json.loads(stdout)
        raw = result["choices"][0]["message"]["content"].strip()
        
        # Handle empty content (may be in reasoning_content)
        if not raw:
            reasoning = result["choices"][0]["message"].get("reasoning_content", "")
            if reasoning:
                logger.warning("L2 screener: using reasoning_content (%d chars)", len(reasoning))
                raw = reasoning.strip()
            else:
                logger.error("L2 screener: empty content and no reasoning_content")
                return {"risk": "block", "type": "ERROR", "reason": "审查 API 返回空内容，fail-closed", "confidence": 100}
        
        # Parse JSON (handle markdown wrapping)
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:]) if lines else raw
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        
        verdict = json.loads(raw)
        
        # Validate required fields
        for field in ["risk", "type", "confidence", "reason"]:
            if field not in verdict:
                logger.warning("L2 screener missing field: %s", field)
                return {"risk": "block", "type": "ERROR",
                        "reason": f"审查结果缺少字段 {field}，fail-closed", "confidence": 100}
        
        if verdict["risk"] not in ("block", "warn", "pass"):
            return {"risk": "block", "type": "ERROR",
                    "reason": f"非法 risk 值: {verdict['risk']}", "confidence": 100}
        
        return verdict
        
    except subprocess.TimeoutExpired:
        return {"risk": "block", "type": "ERROR", "reason": "审查 API 超时，fail-closed", "confidence": 100}
    except json.JSONDecodeError as e:
        logger.error("L2 screener JSON parse error: %s", e)
        return {"risk": "block", "type": "ERROR",
                "reason": f"审查结果 JSON 解析失败: {e}", "confidence": 100}
    except Exception as e:
        logger.error("L2 screener exception: %s", e)
        return {"risk": "block", "type": "ERROR",
                "reason": f"审查异常: {e}", "confidence": 100}
def _sanitize_url_base64(content: str, source: str) -> str:
    """预处理 web_search 结果：替换 URL 中的 JWT base64 为占位符。

    ⚠️ 硬限制：仅 source == "web_search" 时执行。
    URL base64 无害假设仅对搜索引擎结果成立。
    用户直接输入/邮件/文档中的 URL base64 可能是攻击载荷。
    """
    logger.warning("_sanitize_url_base64 called: source=%s len=%d", source, len(content))
    logger.warning("_sanitize_url_base64 called: source=%s len=%d", source, len(content))
    if source != "web_search":
        return content

    # 0. 检查原始内容是否已含占位符（攻击者预植）
    if "<URL_B64_TOKEN>" in content:
        logger.warning("_sanitize_url_base64: content already contains <URL_B64_TOKEN>, skipping")
        return content

    # 1. 快速路径：无 URL 则跳过
    if "https://" not in content and "http://" not in content:
        return content

    # 2. base64 占比检测（跳过全 base64 构造的内容）
    total_len = len(content)
    if total_len >= 200 and source != "web_search":  # 小内容豁免，web_search 不受限
        b64_chars = sum(1 for c in content if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
        if total_len > 0 and b64_chars / total_len > 0.8:
            logger.info("_sanitize_url_base64: >80%% base64 content, skipping")
            return content

    # 3. 截断超大内容
    MAX_LEN = 50_000
    scan_content = content[:MAX_LEN]
    if len(content) > MAX_LEN:
        logger.debug("_sanitize_url_base64: truncated to %d chars", MAX_LEN)

    # 4. 提取所有 URL
    import re
    url_re = re.compile(r"https?://[^\s<>\"{}|\\^`\[\])]+")
    urls = url_re.findall(scan_content)

    if not urls:
        return content

    # 5. JWT 模式正则（eyJ 开头 + 点分隔三段 base64url）
    jwt_param_re = re.compile(r"([&?][A-Za-z0-9_-]*=)(eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})")
    jwt_path_re = re.compile(r"(/)(eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})")
    # 6. 通用 URL Base64 正则 — 非JWT token（YouTube redir_token、Google sig、CDN签名等）
    b64_param_re = re.compile(r'([?&][A-Za-z0-9_-]+=)([A-Za-z0-9+/=]{40,})(?=[&#]|$)')
    b64_path_re = re.compile(r'(/)([A-Za-z0-9+/=]{40,})([/?#&]|$)')
    # 7. Data URI 正则 — 内嵌 base64 资源（data:image/png;base64,... 等）
    data_uri_re = re.compile(r'(data:[^;]*;base64),([A-Za-z0-9+/=]{40,})')

    replaced_count = 0
    for url in urls:
        new_url = url
        # Query string JWT
        new_url, n1 = jwt_param_re.subn(r"\1<URL_B64_TOKEN>", new_url)
        # Path JWT
        new_url, n2 = jwt_path_re.subn(r"\1<URL_B64_TOKEN>", new_url)
        # 通用 URL Base64 参数（非JWT）
        new_url, n3 = b64_param_re.subn(r"\1<URL_B64_TOKEN>", new_url)
        # 通用 URL Base64 路径
        new_url, n4 = b64_path_re.subn(r"\1<URL_B64_TOKEN>\3", new_url)
        # Data URI
        new_url, n5 = data_uri_re.subn(r"\1,<URL_B64_TOKEN>", new_url)
        if n1 + n2 + n3 + n4 + n5 > 0:
            content = content.replace(url, new_url, 1)
            replaced_count += n1 + n2 + n3 + n4 + n5

    if replaced_count > 0:
        logger.debug("_sanitize_url_base64: replaced %d URL base64 segments", replaced_count)

    return content


def _screen_content(content: str, tool_name: str) -> str:
    """
    Run two-layer injection screening on tool result content.
    Returns transformed content string (might be truncated/warn-prefixed).
    """
    # Truncate very large content for screening (don't send 1MB to API)
    MAX_SCREEN_BYTES = 200_000  # ~50K tokens
    screen_content = content
    truncated = False
    if len(content.encode("utf-8")) > MAX_SCREEN_BYTES:
        screen_content = content[:MAX_SCREEN_BYTES // 4]  # ~50K chars
        truncated = True

    # ★ web_search 预处理：URL base64 占位符替换（方案E, ADR-0010）
    screen_content = _sanitize_url_base64(screen_content, tool_name)

    # L1: Regex scan
    l1 = _run_l1_scan(screen_content)
    
    # L1 block → block immediately (no L2 needed)
    if l1["risk"] == "block":
        logger.info("L1 BLOCK %s: %s", tool_name, l1.get("reason", ""))
        # Build match details for user judgment
        match_details = []
        for p in l1.get("patterns", []):
            detail = f"规则: {p['name']}"
            if "matched" in p:
                detail += f"\n匹配: {p['matched']}"
            if "context" in p:
                detail += f"\n上下文: {p['context']}"
            match_details.append(detail)
        
        return json.dumps({
            "error": "🚫 注入攻击已阻断",
            "layer": "L1",
            "type": l1.get("type", "UNKNOWN"),
            "reason": l1.get("reason", ""),
            "patterns": [p["name"] for p in l1.get("patterns", [])],
            "match_details": match_details,
            "hint": "外部内容含恶意指令，已被脚本层阻断。如需查看原始内容，请告知。"
        }, ensure_ascii=False)
    
    # L2: LLM semantic screen
    l2 = _run_l2_screen(screen_content)
    
    # Merge L1 + L2: L1 warn + L2 verdict
    if l2["risk"] == "block":
        l1_info = f"; L1: {l1.get('reason', 'pass')}" if l1["risk"] == "warn" else ""
        logger.info("L2 BLOCK %s: %s%s", tool_name, l2.get("reason", ""), l1_info)
        return json.dumps({
            "error": "🚫 注入攻击已阻断",
            "layer": "L2",
            "type": l2.get("type", "UNKNOWN"),
            "reason": l2.get("reason", ""),
            "l1_warn": [
                {"name": p["name"], "matched": p.get("matched", ""), "context": p.get("context", "")}
                for p in l1.get("patterns", [])
            ] if l1["risk"] == "warn" else [],
            "screened_snippet": screen_content[:300] + ("..." if len(screen_content) > 300 else ""),
            "hint": "外部内容含恶意指令，已被语义审查层阻断。上方 screened_snippet 为被审查内容的前 300 字符，reason 为 LLM 判断理由。如需查看原始内容，请告知。"
        }, ensure_ascii=False)
    
    if l2["risk"] == "warn":
        l1_warn = "\n".join(f"- ⚠️ L1: {p['name']}" for p in l1.get("patterns", []))
        warning = f"⚠️ 注入检测标注 [{l2.get('type', 'SUSPICIOUS')}] {l2.get('reason', '')}"
        if l1_warn:
            warning += f"\n{l1_warn}"
        warning += "\n\n--- 以下为原始内容 ---\n\n"
        logger.info("WARN %s: %s", tool_name, l2.get("reason", ""))
        return warning + content
    
    # L1 warn but L2 pass → resolve to pass (L2 has final say)
    if l1["risk"] == "warn":
        logger.info("L1 warn overridden by L2 pass in %s", tool_name)
    
    if truncated:
        logger.info("Content truncated for screening: %d bytes", len(content.encode("utf-8")))
    
    return content


def _check_args(tool_name: str, args: dict) -> dict:
    """Pre-tool-call check on arguments (paths, URLs)."""
    global _last_read_path, _last_read_whitelisted
    _last_read_path = None
    _last_read_whitelisted = False
    
    if tool_name == "read_file":
        path = args.get("path", "")
        _last_read_path = path
        # Check whitelist first
        import fnmatch
        for wl in CONTENT_SCREEN_WHITELIST:
            # Support directory prefix whitelist (path ends with /*)
            if wl.endswith("/*") and path.startswith(wl[:-2]):
                _last_read_whitelisted = True
                return {}
            if fnmatch.fnmatch(path, wl) or path == wl:
                _last_read_whitelisted = True
                return {}  # Skip all screening for whitelisted paths
        
        for pattern in SUSPICIOUS_PATHS:
            if pattern.search(path):
                logger.warning("BLOCKED read_file on suspicious path: %s", path)
                return {"action": "block", "message": f"🚫 文件路径被阻断: {path}（目录遍历/系统路径）"}
    
    elif tool_name in ("write_file", "patch"):
        path = args.get("path", "")
        # Block writes to MEMORY.md — must use memory tool instead
        MEMORY_MD = os.path.join(os.path.expanduser("~"), ".hermes", "memories", "MEMORY.md")
        if path == MEMORY_MD or path.endswith("/MEMORY.md"):
            logger.warning("BLOCKED %s on MEMORY.md: %s", tool_name, path)
            return {"action": "block", "message": "🚫 MEMORY.md 只能通过 memory tool 写入。请使用 memory(action='add')。"}
    
    elif tool_name == "web_extract":
        urls = args.get("urls", [])
        for url in urls:
            for pattern in SUSPICIOUS_URL_PATTERNS:
                if pattern.search(url):
                    logger.warning("BLOCKED web_extract on suspicious URL: %s", url)
                    return {"action": "block", "message": f"🚫 URL 被阻断: {url}（内网/危险协议）"}
    
    return {}  # Allow


# ── Plugin Hooks ────────────────────────────────────────────────────────────

def register(ctx):
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("transform_tool_result", _transform_tool_result)
    logger.info("injection-guard plugin registered (pre_tool_call + transform_tool_result)")


def _pre_tool_call(tool_name: str, args: dict, **kwargs):
    """Block dangerous tool calls before execution."""
    if tool_name not in CHECK_ARGS_TOOLS:
        return {}
    
    try:
        return _check_args(tool_name, args)
    except Exception as e:
        logger.error("pre_tool_call exception in %s: %s", tool_name, e)
        return {"action": "block", "message": f"🚫 参数检查异常，fail-closed: {e}"}


def _transform_tool_result(tool_name: str, result: str, **kwargs):
    """Screen tool results through L1+L2 injection detection."""
    global _last_read_whitelisted
    
    if tool_name not in SCREEN_CONTENT_TOOLS:
        return None  # Pass through unchanged
    
    # Skip screening for whitelisted paths (set by _pre_tool_call)
    if _last_read_whitelisted:
        logger.info("Whitelisted path skipped screening: %s", _last_read_path)
        _last_read_whitelisted = False  # Reset for next call
        return None  # Pass through unchanged
    
    try:
        screened = _screen_content(result, tool_name)
        # Wrap external content with boundary markers for downstream protection
        return f"<<<EXTERNAL_UNTRUSTED_CONTENT>>>\n{screened}\n<<<END_EXTERNAL_CONTENT>>>"
    except Exception as e:
        logger.error("transform_tool_result exception in %s: %s", tool_name, e)
        # Fail-closed: replace with error
        return json.dumps({
            "error": "🚫 注入审查层故障",
            "reason": f"审查插件异常，内容已阻断（fail-closed）: {e}",
            "hint": "审查系统临时不可用，请稍后重试或告知杨旸检查日志。"
        }, ensure_ascii=False)

# @hermes:patch 2026-07-08 | session:20260708_120024_6b74bb93

# @hermes:patch 2026-07-08 | session:20260708_120024_6b74bb93

# @hermes:patch 2026-07-08 | session:20260708_120024_6b74bb93
