"""后端 handler 直接调用回归测试 —— 不经过 HTTP,验证 plugin_api.py 的
optimize 端点能真实跑通 run_oneshot。

运行(仓库根 = <repo>/plugins/prompt-optimizer):
    python tests/test_api.py [input-text]
"""
import os
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(TESTS_DIR)          # <repo>/plugins/prompt-optimizer
REPO_ROOT = os.path.dirname(os.path.dirname(PLUGIN_DIR))  # hermes-agent 仓库根
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PLUGIN_DIR)

import importlib.util

api_path = os.path.join(PLUGIN_DIR, 'dashboard', 'plugin_api.py')
spec = importlib.util.spec_from_file_location('po_api', api_path)
mod = importlib.util.module_from_spec(spec)
sys.modules['po_api'] = mod
spec.loader.exec_module(mod)

req = mod.OptimizeRequest(
    input=sys.argv[1] if len(sys.argv) > 1 else '帮我写一个python脚本读取csv',
    instructions='你是一位资深提示词优化专家，请优化用户的提示词。',
    session_id=None,  # 无会话 → auxiliary 后端
    max_tokens=2000,
    temperature=0.3,
    timeout=120,
)
t0 = time.time()
try:
    resp = mod.optimize(req)
    print(f"OK in {time.time()-t0:.1f}s: {resp.text[:120]!r}")
except Exception as e:
    print(f"FAIL in {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
    sys.exit(1)
