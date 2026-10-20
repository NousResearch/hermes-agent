"""辅助后端(auxiliary title_generation)单次调用耗时测试 —— 用于复现
llm.oneshot 慢响应问题与回归验证。

运行(仓库根 = <repo>/plugins/prompt-optimizer):
    python tests/test_oneshot.py
"""
import os
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(TESTS_DIR)          # <repo>/plugins/prompt-optimizer
REPO_ROOT = os.path.dirname(os.path.dirname(PLUGIN_DIR))  # hermes-agent 仓库根
sys.path.insert(0, REPO_ROOT)

from agent.oneshot import run_oneshot

t0 = time.time()
try:
    text = run_oneshot(
        instructions='你是一位资深提示词优化专家，请优化用户的提示词。',
        user_input='帮我写一个python脚本读取csv',
        max_tokens=2000,
        temperature=0.3,
        timeout=60,
    )
    print(f"OK in {time.time()-t0:.1f}s: {text[:200]}")
except Exception as e:
    print(f"FAIL in {time.time()-t0:.1f}s: type={type(e).__name__} msg={str(e)!r}")
    sys.exit(1)
