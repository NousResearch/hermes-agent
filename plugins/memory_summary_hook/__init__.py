"""pre_llm_call hook — 注入 MEMORY 摘要到 user message
v2: 新对话首轮用 DeepSeek V3 重新生成摘要（Pro 返回 reasoning_content 而非 content，用 V3 替代）。
后续轮次复用，不再依赖每日 cron。
"""
import logging, os, json, time, urllib.request
logger = logging.getLogger(__name__)

SUMMARY_FILE = os.path.expanduser("~/.hermes/memories/MEMORY_SUMMARY.md")
MEMORY_FILE = os.path.expanduser("~/.hermes/memories/MEMORY.md")
DS_KEY_FILE = os.path.expanduser("~/.hermes/.deepseek_key")
STATE_FILE = os.path.expanduser("~/.hermes/state/memory_hook_probe")

def register(ctx):
    ctx.register_hook("pre_llm_call", _inject_summary)
    logger.info("memory-summary-hook: registered (v2 — per-session refresh)")

def _regenerate_summary():
    """读 MEMORY.md → DeepSeek V3 → 写 MEMORY_SUMMARY.md"""
    if not os.path.exists(MEMORY_FILE):
        return False
    
    memory_text = open(MEMORY_FILE).read()
    if len(memory_text) < 100:
        return False
    
    ds_key = open(DS_KEY_FILE).read().strip()
    
    prompt = f"""将以下 MEMORY.md 压缩为 5KB 以内的结构化摘要。
保留：活跃项目状态、关键决策、用户偏好、已知陷阱、环境配置。
丢弃：已完成的记录、过时配置、重复内容。

MEMORY.md 原文：
{memory_text}

按以下结构输出（不要加额外说明）：
# MEMORY.md 结构化摘要 (5K)
> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
> 模型: DeepSeek V3

## 活跃项目状态与关键决策
## 用户偏好与铁律
## 已知陷阱与避坑
（此节只保留结论，删除一切操作细节。判断标准：删完后，一个完全不懂技术的人也能看懂这条陷阱的本质教训。API路径、参数名、端口号、命令、错误码——全删。格式：一行一个陷阱，含结论+教训关键词。）
## 环境与配置要点"""
    
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是记忆压缩器。输出紧凑的结构化摘要，不超过 5KB。只保留活跃信息，丢弃已完成/过时内容。不要加额外说明。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 3000
    }).encode()
    
    try:
        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {ds_key}"}
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        summary = resp["choices"][0]["message"]["content"].strip()
        
        if not summary:
            logger.warning("memory-summary-hook: empty response from API")
            return False
        
        with open(SUMMARY_FILE, "w") as f:
            f.write(summary)
        
        logger.info(f"memory-summary-hook: regenerated ({len(summary)} chars)")
        return True
    except Exception as e:
        logger.warning(f"memory-summary-hook: regeneration failed: {e}")
        return False

def _inject_summary(**kwargs):
    # 探针
    try:
        with open(STATE_FILE, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    except:
        pass
    
    is_first = kwargs.get("is_first_turn", False)
    
    # 新对话首轮 → 重新生成摘要（Pro 模型有 reasoning_content 问题，用 V3）
    if is_first:
        logger.info("memory-summary-hook: first turn, regenerating summary...")
        _regenerate_summary()
    
    if not os.path.exists(SUMMARY_FILE):
        return {}
    summary = open(SUMMARY_FILE).read()
    if not summary.strip():
        return {}
    return {"context": summary}

# @hermes:patch 2026-07-08 | session:20260708_120024_6b74bb93

# @hermes:patch 2026-07-08 | session:20260708_122631_df531cec
