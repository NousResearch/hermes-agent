#!/usr/bin/env python3
"""实验：cron 外部 worker 式「写入在飞时进程立刻退出」会不会吃掉记忆。

背景：sync_turn / on_session_end 的落库工作在插件自建的 daemon 线程里；cron 的
外部 worker 进程在 agent.close() 之后 ~0.7s 就退出解释器 → daemon 线程被杀，
正在做 Ollama 嵌入的写入静默消失（这就是 cron 任务"跑完了但没记忆"的形态）。

验收：HERMES_MEMORY_ZVEC_EXIT_DRAIN_S=0 → 落库 0 条（复现丢）；默认（30s）→ 落库 1 条（已修）。

用法：
  python3 exit_loss_experiment.py <插件路径> <标签>
  # 典型：
  HERMES_MEMORY_ZVEC_EXIT_DRAIN_S=0 python3 exit_loss_experiment.py \
      ~/.hermes/hermes-agent/plugins/memory/memory-zvec/__init__.py drain_off

依赖：Ollama 在 11434 且已 pull bge-m3（真实嵌入，不 mock）。
注意：它只往 /tmp/exit_loss_<标签>/ 写临时集合，不动任何生产记忆库。
"""
import json
import os
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~/.hermes")
VENV_PY = os.path.join(HOME, "hermes-agent/venv/bin/python3")
PLUGIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HOME, "hermes-agent/plugins/memory/memory-zvec/__init__.py")
LABEL = sys.argv[2] if len(sys.argv) > 2 else "live"
WORK = f"/tmp/exit_loss_{LABEL}"
MODE = os.environ.get("EXPERIMENT_MODE", "both")   # both | turn | session_end

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK, exist_ok=True)

CHILD = r'''
import os, sys, time
PLUGIN, WORK, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
HOME = os.path.expanduser("~/.hermes")
os.environ.setdefault("HERMES_HOME", os.path.join(HOME, "profiles/default"))
sys.path.insert(0, os.path.join(HOME, "hermes-agent"))
from plugins.memory import load_memory_provider

p = load_memory_provider("memory-zvec")
p._config = {"zvec_dir": WORK + "/zvec_memory", "collection_name": "memories",
             "embedding_model": "bge-m3:latest", "base_url": "http://localhost:11434",
             "vector_dim": 1024, "min_content_len": 50, "enable_hnsw_optimize": True}
p.initialize(session_id="exit_test")

if MODE in ("both", "turn"):
    p.sync_turn("进程退出竞态实验：" + "这是用于触发真实 bge-m3 嵌入的长文本。" * 12,
                "进程退出竞态实验回复：" + "确认写入是否在解释器退出时被 daemon 线程杀死。" * 12,
                session_id="exit_test")

if MODE in ("both", "session_end"):
    now = time.time()
    msgs = []
    for i in range(4):
        msgs.append({"role": "user", "content": f"会话收尾实验第{i}轮：触发真实嵌入。" * 6,
                     "timestamp": now + i * 2})
        msgs.append({"role": "assistant", "content": f"会话收尾实验第{i}轮回复：确认批次完整落库。" * 6,
                     "timestamp": now + i * 2 + 1})
    p.on_session_end(msgs)

# 模拟 cron 外部 worker：不等待任何后台工作，直接让 main 结束、解释器退出
sys.stdout.flush()
'''

with open("/tmp/exit_child_generic.py", "w") as f:
    f.write(CHILD)

print(f"=== [{LABEL}] 插件: {PLUGIN}")
print(f"    退出兜底 env = {os.environ.get('HERMES_MEMORY_ZVEC_EXIT_DRAIN_S', '(默认 30s)')}  mode={MODE}")
t0 = time.time()
r = subprocess.run([VENV_PY, "/tmp/exit_child_generic.py", PLUGIN, WORK, MODE],
                   capture_output=True, text=True, timeout=300)
elapsed = time.time() - t0
if r.returncode != 0:
    print("    子进程 stderr:", r.stderr.strip()[-300:])

code = f'''
import zvec, json
from collections import Counter
c = zvec.open("{{WORK}}/zvec_memory/memories")
rows = c.query(zvec.Query(field_name="vector", vector=[0.0]*1024), topk=max(c.stats.doc_count, 1))
print(json.dumps({{"doc_count": c.stats.doc_count,
                  "roles": dict(Counter((getattr(r, "fields", {{}}) or {{}}).get("role", "") for r in rows))}}))
'''.replace("{WORK}", WORK)
r2 = subprocess.run([VENV_PY, "-c", code], capture_output=True, text=True, timeout=120)
lines = (r2.stdout or "").strip().splitlines()
info = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else {"err": r2.stderr[-200:]}

expected = 0
if MODE in ("both", "turn"):
    expected += 1
if MODE in ("both", "session_end"):
    expected += 4

print(f"    子进程总寿命 {elapsed:.2f}s；落库 {info.get('doc_count')} 条 / 期望 {expected} 条  明细={info.get('roles')}")
ok = info.get("doc_count") == expected
print(f"    => {'✅ 在飞写入存活' if ok else '❌ 写入被进程退出吃掉'}")
sys.exit(0 if ok else 1)
