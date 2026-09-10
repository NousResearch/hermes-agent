"""会话模型继承路径耗时测试 —— 模拟 llm.oneshot 继承 live session 的
provider/model/credentials(即 plugin_api.py 的 main_runtime 分支)。

运行(仓库根 = <repo>/plugins/prompt-optimizer):
    python tests/test_oneshot_session.py
"""
import os
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(TESTS_DIR)          # <repo>/plugins/prompt-optimizer
REPO_ROOT = os.path.dirname(os.path.dirname(PLUGIN_DIR))  # hermes-agent 仓库根
sys.path.insert(0, REPO_ROOT)

from agent.oneshot import run_oneshot

# 模拟会话模型继承(示例:opencode-go / deepseek-v4-flash —— 按需替换为
# 当前会话实际 provider/model/base_url)
main_runtime = {
    "provider": "opencode-go",
    "model": "deepseek-v4-flash",
    "base_url": "https://opencode.ai/zen/go/v1/",
}

t0 = time.time()
try:
    text = run_oneshot(
        instructions='你是一位资深提示词优化专家，请优化用户的提示词。',
        user_input='帮我写一个python脚本读取csv',
        max_tokens=2000,
        temperature=0.3,
        timeout=120,
        main_runtime=main_runtime,
    )
    print(f"OK in {time.time()-t0:.1f}s: {text[:150]}")
except Exception as e:
    print(f"FAIL in {time.time()-t0:.1f}s: type={type(e).__name__} msg={str(e)!r}")
    sys.exit(1)
