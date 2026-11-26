#!/usr/bin/env python3
"""Task Looper — 持续迭代，独立会话，状态外存。"""
import json, os, sys, time, subprocess, signal, shutil
from pathlib import Path

TASK_ID = sys.argv[1]
PROMPT = sys.argv[2]
INTERVAL_MIN = int(sys.argv[3])  # minutes (从 SKILL.md 传来的是分钟)
INTERVAL_SEC = INTERVAL_MIN * 60  # 转为秒

# 🔧 Fix: 使用环境变量 HERMES_SKILL_DIR 或 PATH 定位 hermes
def find_hermes():
    """Fix #2: 便携式 Hermes 路径解析，不硬编码 /home/andore/"""
    # 1) 环境变量
    hermes_home = os.environ.get("HERMES_SKILL_DIR")
    if hermes_home:
        candidate = Path(hermes_home) / ".." / "bin" / "hermes"
        if candidate.exists():
            return str(candidate)
    # 2) PATH
    path_candidate = shutil.which("hermes")
    if path_candidate:
        return path_candidate
    # 3) 兜底
    return "hermes"

HERMES_BIN = find_hermes()

STATE_DIR = Path.home() / ".hermes" / "task_loops" / TASK_ID
STATE_FILE = STATE_DIR / "state.json"
STOP_FILE = STATE_DIR / "stop"

STATE_DIR.mkdir(parents=True, exist_ok=True)

# Init state
state = {"task": PROMPT, "interval_min": INTERVAL_MIN, "round": 0, "last_summary": "", "status": "running"}
if STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text())
    except:
        pass

def save_state(**updates):
    state.update(updates)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

save_state(status="running")

print(f"[Looper] 任务: {PROMPT[:80]}")
print(f"[Looper] 间隔: {INTERVAL_MIN} 分钟")
print(f"[Looper] HEMRES: {HERMES_BIN}")
print(f"[Looper] 停止信号: {STOP_FILE}")

loop_completed = False
while not loop_completed:
    # Check stop signal
    if STOP_FILE.exists():
        save_state(status="stopped")
        STOP_FILE.unlink(missing_ok=True)
        print(f"[Looper] 🛑 收到停止信号，循环结束。共 {state['round']} 轮。")
        break  # Fix #3: break 后不执行最后的 save_state(completed)

    state["round"] += 1
    save_state(status=f"running_round_{state['round']}")

    full_prompt = PROMPT
    if state.get("last_summary"):
        full_prompt = f"[上轮结果] {state['last_summary']}\n\n[本轮任务] {PROMPT}"

    print(f"\n{'='*50}")
    print(f"[Looper] 第 {state['round']} 轮开始...")

    try:
        result = subprocess.run(
            [HERMES_BIN, "chat", "-q", full_prompt],
            capture_output=True, text=True, timeout=max(300, INTERVAL_SEC * 2),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        output = result.stdout.strip() or result.stderr.strip()[:500]
        summary = output[-200:] if len(output) > 200 else output
        save_state(last_summary=summary, last_output=output[:2000])
        print(f"[Looper] ✅ 第 {state['round']} 轮完成")

    except subprocess.TimeoutExpired:
        save_state(last_summary="[超时]", status="timeout_round")
        print("[Looper] ⚠️ 超时")
    except Exception as e:
        save_state(last_summary=f"[错误: {e}]", status="error_round")
        print(f"[Looper] ❌ 错误: {e}")

    # Wait for next interval
    print(f"[Looper] 等待 {INTERVAL_MIN} 分钟...")
    for _ in range(INTERVAL_SEC):
        if STOP_FILE.exists():
            loop_completed = True
            break
        time.sleep(1)

# Fix #3: 只有正常完成才写 completed，不覆盖 stop 状态
if not STOP_FILE.exists():
    save_state(status="completed")
    print("[Looper] ✅ 正常完成")
