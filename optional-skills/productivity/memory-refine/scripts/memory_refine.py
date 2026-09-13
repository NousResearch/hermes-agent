#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes 记忆自动提炼(memory-refine)

从 Hermes 会话记录(state.db)增量提炼「关于用户的稳定观察」,写入
holographic fact_store。增量锚点只提炼新消息,避免历史重复提炼浪费 token。

设计:
  - 数据源: Hermes state.db 的 messages 表(role=user, 排除 cron_% 会话)
  - 提炼:   调 OpenAI 兼容 /chat/completions,由 LLM 判断稳定观察
  - 去重:   2-gram 中文预检(与已有 facts 比对)+ add_fact 的 content UNIQUE 兜底
  - 写库:   走 holographic MemoryStore 官方入口(自动实体/HRR向量/FTS)

配置(环境变量,均可选,有默认):
  HERMES_HOME            Hermes 数据目录(默认 ~/.hermes)
  HERMES_LLM_BASE_URL    OpenAI 兼容端点(默认 https://api.openai.com/v1)
  HERMES_LLM_API_KEY     API key(必填)
  HERMES_LLM_MODEL       模型名(默认 gpt-4o-mini)

用法:
  python3 memory_refine.py --hours 24 --dry-run   # 冒烟:只打印提炼结果
  python3 memory_refine.py --hours 24 --write     # 正式:提炼并写入 fact_store
  python3 memory_refine.py --write --quiet        # cron:全去重时静默退出
  python3 memory_refine.py --reset-anchor --write # 忽略锚点全量回看(调试)
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置:优先环境变量,默认 ~/.hermes
# ---------------------------------------------------------------------------
HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))

DB = str(HERMES_HOME / "state.db")
STATE_FILE = str(HERMES_HOME / "scripts" / "memory_refine_state.json")
LLM_BASE_URL = os.environ.get("HERMES_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("HERMES_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("HERMES_LLM_MODEL", "gpt-4o-mini")

MAX_SINGLE_MSG = 3000      # 单条 user 消息超过则截断
MAX_TOTAL_CHARS = 60000    # 输入总预算(约 2 万 token,留 reasoning 空间)


def load_token():
    """OpenAI 兼容鉴权:API key 从 HERMES_LLM_API_KEY 读取。"""
    tok = LLM_API_KEY
    if not tok:
        sys.exit("ERROR: 未设置 HERMES_LLM_API_KEY 环境变量(OpenAI 兼容端点的 API key)")
    return tok


def load_state():
    """读增量锚点。返回 {'last_msg_id': int}。"""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"last_msg_id": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def fetch_user_messages(hours, last_msg_id=0):
    """取 user 消息,按会话分组、按时间排序。返回 {session: [(ts, content, msg_id), ...]}

    增量逻辑:
      - 有 last_msg_id → 只取 id > last_msg_id 的新消息
      - 同时用 hours 做上限保护:即使很久没跑,也只回看最近 hours 小时
      - 无 last_msg_id(首次)→ 回看 hours 小时
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    if last_msg_id:
        cur.execute("""
            SELECT id, session_id, timestamp, content FROM messages
            WHERE role='user'
              AND id > ?
              AND timestamp > strftime('%s','now',?)
              AND content IS NOT NULL AND length(content) > 5
              AND session_id NOT LIKE 'cron\\_%' ESCAPE '\\'
            ORDER BY timestamp
        """, (last_msg_id, f"-{hours} hours"))
    else:
        cur.execute("""
            SELECT id, session_id, timestamp, content FROM messages
            WHERE role='user'
              AND timestamp > strftime('%s','now',?)
              AND content IS NOT NULL AND length(content) > 5
              AND session_id NOT LIKE 'cron\\_%' ESCAPE '\\'
            ORDER BY timestamp
        """, (f"-{hours} hours",))
    groups = {}
    max_id = last_msg_id
    for r in cur.fetchall():
        sid = r["session_id"]
        groups.setdefault(sid, []).append((r["timestamp"], r["content"], r["id"]))
        max_id = max(max_id, r["id"])
    con.close()
    return groups, max_id


def format_batch(groups, max_chars):
    """把会话分组转成 prompt 输入文本,按总量预算截断(优先保留最新)。"""
    parts = []
    total = 0
    # 按会话最近活动时间排序(新的在前),保证预算内优先最新会话
    sessions = sorted(groups.items(), key=lambda kv: max(t for t, _, _ in kv[1]), reverse=True)
    for sid, msgs in sessions:
        block = [f"--- 会话 {sid} ---"]
        for ts, content, _mid in msgs:
            text = content if len(content) <= MAX_SINGLE_MSG else content[:MAX_SINGLE_MSG] + "…[截断]"
            block.append(f"[{datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')}] {text}")
        block_txt = "\n".join(block) + "\n"
        if total + len(block_txt) > max_chars:
            continue  # 超预算的整个会话跳过(不截半)
        parts.append(block_txt)
        total += len(block_txt)
    return "\n".join(parts), total


REFINE_PROMPT = """你是一个长期记忆提炼器,阅读用户和 AI 的对话记录,提炼出「关于用户的稳定观察」。

只基于下面给出的对话,禁止编造或推测没有依据的内容。
对每条观察标注确定性:【确定】(对话明确说明)/【推断】(合理但未明说)。
每轮只输出,不要解释。

输出格式:每行一条观察,用「类型|确定性|内容」:
- 类型: 偏好 | 行为模式 | 关注主题 | 个人背景 | 环境事实 | 项目状态
- 例: 偏好|确定|用户喜欢简洁直接的回复,讨厌客服式开场
- 例: 项目状态|确定|用户正在评估 Honcho 记忆提供商,已否决 Mac 自托管方案
- 例: 行为模式|推断|用户倾向于先冒烟验证再决定是否常驻部署

要求:
1. 只提炼「跨会话仍有用」的稳定信息;临时任务、一次性问题不要收
2. 数量 5-15 条,质量优先
3. 中文输出
4. 对话里已经出现在系统记忆中的常识(如用户是交易者)不必重复,重点是新信息

=== 对话记录开始 ===
{conversation}
=== 对话记录结束 ===
"""


def call_llm(prompt, token, max_tokens=4000, retries=2):
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        },
        method="POST",
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            # 空内容:reasoning 模型偶发,重试
            last_err = f"empty content (finish={data['choices'][0].get('finish_reason')}, usage={data.get('usage')})"
            print(f"[重试] {attempt+1}/{retries+1} {last_err}", file=sys.stderr)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            last_err = f"HTTP {e.code}: {detail}"
            print(f"[重试] {attempt+1}/{retries+1} {last_err}", file=sys.stderr)
        except Exception as e:
            last_err = str(e)
            print(f"[重试] {attempt+1}/{retries+1} {last_err}", file=sys.stderr)
        if attempt < retries:
            time.sleep(3)
    sys.exit(f"LLM 调用失败(重试{retries}次仍失败): {last_err}")


def parse_observations(result: str):
    """解析 LLM 提炼输出(每行「类型|确定性|内容」)。返回 [(type, cert, content)]"""
    obs = []
    for line in result.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        obs.append((parts[0], parts[1], parts[2]))
    return obs


def write_to_factstore(observations, dry_run=True):
    """预检去重 + 写入 fact_store(走 MemoryStore 官方入口,自动向量/实体/去重)。

    去重策略:
      1. 精确去重 — MemoryStore.add_fact 自带 content UNIQUE 约束,重复自动返回旧 id
      2. 语义去重 — search_facts 预检,和已有事实高度重合的观察跳过(避免近似重复堆积)
    """
    # 定位 Hermes 安装目录:优先 HERMES_INSTALL_DIR,其次本脚本同仓/常见路径
    hermes_install = os.environ.get("HERMES_INSTALL_DIR", "")
    if not hermes_install:
        # 尝试从当前进程/常见位置推断
        for cand in ["/usr/local/lib/hermes-agent", str(Path.home() / "hermes-agent"),
                     str(Path(__file__).resolve().parent.parent.parent)]:
            if os.path.isdir(os.path.join(cand, "plugins", "memory", "holographic")):
                hermes_install = cand
                break
    if not hermes_install:
        sys.exit("ERROR: 定位不到 Hermes 安装目录,请设置 HERMES_INSTALL_DIR 环境变量")
    sys.path.insert(0, hermes_install)
    from plugins.memory.holographic.store import MemoryStore

    store = MemoryStore()
    CAT_MAP = {"偏好": "user_pref", "项目状态": "project", "环境事实": "general",
               "行为模式": "user_pref", "个人背景": "user_pref", "关注主题": "general"}
    added, skipped = [], []

    # 预检前一次性读全量事实,避免每条约1次查询
    existing = []
    try:
        conn = store._conn
        for r in conn.execute(
            "SELECT fact_id, content FROM facts WHERE trust_score >= 0.3"
        ).fetchall():
            existing.append((r[0], r[1]))
    except Exception as e:
        print(f"[警告] 读取已有事实失败: {e}", file=sys.stderr)
        existing = []

    def _is_dup(content: str) -> str | None:
        """LIKE 语义预检:观察内容里每个长度>=2的连续片段,在已有事实中命中即视为重复。"""
        if len(content) < 2:
            return None
        # 中文:按 2-gram 拆,匹配到 2 个及以上相同片段即判重
        grams = {content[i:i+2] for i in range(0, len(content)-1)}
        for fid, ec in existing:
            match = sum(1 for g in grams if g in ec)
            if len(grams) and match >= max(2, int(len(grams) * 0.15)):
                return f"语义近似已存在(fact_id={fid})"
        return None

    for otype, cert, content in observations:
        if len(content) > 500:
            content = content[:500]
        category = CAT_MAP.get(otype, "general")
        dup_note = _is_dup(content)
        if dup_note:
            skipped.append((content, dup_note))
            continue
        if dry_run:
            added.append((content, "预检通过(待写入)"))
            continue
        try:
            fid = store.add_fact(content, category=category, tags="memory_refine")
            added.append((content, f"新增 fact_id={fid}"))
        except Exception as e:
            skipped.append((content, f"写入失败: {e}"))

    return added, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24, help="回看上限(小时);增量锚点优先,锚点无则回看这么多")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="全被去重挡掉时静默退出(no_agent cron 用)")
    ap.add_argument("--reset-anchor", action="store_true", help="忽略已有锚点,全量回看 hours(调试用)")
    args = ap.parse_args()

    token = load_token()
    state = load_state()
    last_id = 0 if args.reset_anchor else state.get("last_msg_id", 0)

    groups, max_id = fetch_user_messages(args.hours, last_msg_id=last_id)
    if not groups:
        # 无新消息(或全被过滤)→ 无输出,正常退出(不推进锚点)
        if args.quiet:
            return 0
        print("没有新的 user 消息可提炼", file=sys.stderr)
        return 0
    convo, used = format_batch(groups, MAX_TOTAL_CHARS)
    n_sessions = len(groups)
    n_msgs = sum(len(v) for v in groups.values())
    print(f"[输入] 增量(上次id={last_id}, 本次至{max_id}) {n_sessions} 会话 / {n_msgs} 条 user 消息 / 使用 {used} 字符", file=sys.stderr)

    prompt = REFINE_PROMPT.format(conversation=convo)
    t0 = time.time()
    result = call_llm(prompt, token)
    print(f"[LLM] {LLM_MODEL} 耗时 {time.time()-t0:.1f}s\n", file=sys.stderr)

    observations = parse_observations(result)
    if not observations:
        # 提炼失败:不推进锚点,下次重试同一批
        sys.exit("提炼结果解析为空,不推进锚点,下次重试")

    if args.write:
        # --write:预检去重 + 写入 fact_store
        added, skipped = write_to_factstore(observations, dry_run=False)
        # 写库完成(无论新增与否)即推进锚点:
        #   - 全被去重 = 这批观察都已入库/已存在,消息已消化
        #   - 部分新增 = 新观察已入库
        #   - 若这里不推进,重复批次会无限重提,浪费 token
        if max_id > state.get("last_msg_id", 0):
            state["last_msg_id"] = max_id
            save_state(state)
        if args.quiet and not added:
            # 全被去重挡掉 → 静默退出(锚点已推进,不重复)
            return 0
        print(f"[写库] 新增 {len(added)} 条,跳过 {len(skipped)} 条,锚点推进至 {max_id}")
        for c, note in added:
            print(f"  + [{note}] {c[:80]}")
        for c, note in skipped:
            print(f"  - [{note}] {c[:80]}")
    else:
        # dry-run:只打印提炼结果,附每条的类型/确定性(不推进锚点)
        for otype, cert, content in observations:
            print(f"{otype}|{cert}|{content}")


if __name__ == "__main__":
    main()
