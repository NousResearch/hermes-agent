"""One benchmark cell: task x arm x model x rep.

Usage:
    python3 single_run.py <arm> <task_key> <model> <rep>

Arms:
    base  - built-in ``browser_*`` toolset (twelve tools), pinned tree $BUBENCH_BASE_TREE
    pr    - Browser Use CLI mode (single ``browser_exec`` tool), pinned tree $BUBENCH_PR_TREE
    prns  - same as pr but with the schema description stripped to the header only
            (isolates the value of the helpers digest in the tool description)
    stagehand - Stagehand V4 Playwright facade behind the same single
                ``browser_exec`` contract, pinned tree $BUBENCH_PR_TREE

Environment:
    BUBENCH_ROOT       workspace dir (default: dir containing this script)
    BUBENCH_BASE_TREE  checkout used for the ``base`` arm (e.g. a merge-base worktree)
    BUBENCH_BROWSER_USE_TREE exact current-main checkout used for ``pr``/``prns``
    BUBENCH_PR_TREE    checkout containing the ``stagehand`` implementation
    BUBENCH_STAGEHAND_ROOT built Stagehand V4 checkout used by ``stagehand``
    BUBENCH_TASKS      tasks json (default: $BUBENCH_ROOT/tasks/hard.json)
    BENCH_CDP_URL      CDP endpoint both arms drive (default http://127.0.0.1:9333)
    BUBENCH_MODEL_PROVIDER inference route: ai-gateway (default) or openrouter
    AI_GATEWAY_API_KEY Vercel AI Gateway credential for the default route
    OPENROUTER_API_KEY provider credential when using the openrouter route

The run gets a throwaway HERMES_HOME so no local config leaks in, and the
web-fetch credential env vars are stripped so every arm must actually drive
the browser (no web_extract shortcuts).

Prints one line: ``RESULT_JSON:{...}`` consumed by orchestrate.py.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

ARM, TASK_KEY, MODEL, REP = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

ROOT = os.environ.get("BUBENCH_ROOT", os.path.dirname(os.path.abspath(__file__)))
PR_TREE = os.environ["BUBENCH_PR_TREE"]
BASE_TREE = os.environ.get("BUBENCH_BASE_TREE", PR_TREE)
BROWSER_USE_TREE = os.environ.get("BUBENCH_BROWSER_USE_TREE", PR_TREE)
WT = {
    "base": BASE_TREE,
    "pr": BROWSER_USE_TREE,
    "prns": BROWSER_USE_TREE,
    "stagehand": PR_TREE,
}[ARM]

TASKS_PATH = os.environ.get("BUBENCH_TASKS", os.path.join(ROOT, "tasks", "hard.json"))
TASKS = json.load(open(TASKS_PATH, encoding="utf-8"))
task = TASKS[TASK_KEY]

MODEL_PROVIDER = os.environ.get("BUBENCH_MODEL_PROVIDER", "ai-gateway").strip()
MODEL_ROUTES = {
    "ai-gateway": (
        "https://ai-gateway.vercel.sh/v1",
        "AI_GATEWAY_API_KEY",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
    ),
}
if MODEL_PROVIDER not in MODEL_ROUTES:
    raise RuntimeError(
        "BUBENCH_MODEL_PROVIDER must be one of: "
        + ", ".join(sorted(MODEL_ROUTES))
    )
MODEL_BASE_URL, MODEL_API_KEY_ENV = MODEL_ROUTES[MODEL_PROVIDER]
MODEL_API_KEY = os.environ.get(MODEL_API_KEY_ENV, "").strip()
if not MODEL_API_KEY:
    raise RuntimeError(f"{MODEL_API_KEY_ENV} is required for {MODEL_PROVIDER}")

home = tempfile.mkdtemp(prefix=f"buhome-{ARM}-")
hh = os.path.join(home, ".hermes")
os.makedirs(os.path.join(hh, "logs"), exist_ok=True)
cdp = os.environ.get("BENCH_CDP_URL", "http://127.0.0.1:9333")
if ARM == "base":
    browser_cfg = {"cloud_provider": "local", "cdp_url": cdp}
elif ARM == "stagehand":
    stagehand_root = os.environ.get("BUBENCH_STAGEHAND_ROOT", "").strip()
    if not stagehand_root:
        raise RuntimeError(
            "BUBENCH_STAGEHAND_ROOT must point at a built Stagehand V4 checkout"
        )
    browser_cfg = {
        "backend": "stagehand",
        "cloud_provider": "browserbase",
        "stagehand_root": stagehand_root,
    }
else:
    browser_cfg = {"backend": "browser-use"}
cfg = {
    "model": {"provider": MODEL_PROVIDER, "default": MODEL},
    "browser": browser_cfg,
    "display": {"quiet": True},
}
import yaml

with open(os.path.join(hh, "config.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f)
os.environ["HERMES_HOME"] = hh
# Strip web-fetch shortcuts: every arm must drive the browser.
os.environ.pop("BROWSER_USE_API_KEY", None)
for k in ("FIRECRAWL_API_KEY", "NOUS_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"):
    os.environ.pop(k, None)
os.environ["BU_CDP_URL"] = cdp
os.environ["PATH"] = (
    os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")
)

sys.path.insert(0, WT)
import logging

logging.disable(logging.CRITICAL)

import run_agent  # noqa: E402

assert run_agent.__file__.startswith(WT), f"wrong tree: {run_agent.__file__}"

if ARM == "prns":
    # Strip the helpers digest from the schema: header-only description.
    import tools.browser_use_cli as bu  # noqa: E402

    bu._skill_text_fetched = True
    bu._skill_text_cache = None
    bu.BROWSER_EXEC_SCHEMA["description"] = bu._description_header()

from run_agent import AIAgent  # noqa: E402

agent = AIAgent(
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    provider=MODEL_PROVIDER,
    model=MODEL,
    max_iterations=30,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    # NB: "terminal" must be present for the pr arms — since #81958's terminal
    # gate, browser_exec is stripped from sessions whose toolsets exclude
    # terminal. Both arms get the same toolsets for parity; audit
    # tool_call_names in the results for terminal-tool bypasses (curl etc.).
    enabled_toolsets=["browser", "terminal"],
    save_trajectories=False,
)

schema_desc_len = 0
try:
    from model_tools import get_tool_definitions

    for t in get_tool_definitions(agent.enabled_toolsets):
        if t["function"]["name"].startswith("browser"):
            schema_desc_len += len(json.dumps(t["function"]))
except Exception:
    pass

t0 = time.time()
error = None
final = ""
messages = []
try:
    result = agent.run_conversation(task["prompt"])
    final = (
        (result.get("final_response") or "")
        if isinstance(result, dict)
        else str(result)
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
except Exception as e:  # noqa: BLE001
    error = f"{type(e).__name__}: {e}"
    messages = getattr(agent, "messages", []) or []
wall = time.time() - t0

tool_calls = []
for m in messages:
    if isinstance(m, dict) and m.get("role") == "assistant":
        for tc in m.get("tool_calls") or []:
            fn = (
                (tc.get("function") or {}).get("name") if isinstance(tc, dict) else None
            )
            if fn:
                tool_calls.append(fn)


def _ok(text: str) -> bool:
    if task.get("oracle_all"):
        return all(
            re.search(re.escape(x), text, re.IGNORECASE) for x in task["oracle_all"]
        )
    return any(
        re.search(re.escape(x), text, re.IGNORECASE) for x in task.get("oracle_any", [])
    )


out = {
    "arm": ARM,
    "task": TASK_KEY,
    "model": MODEL,
    "rep": int(REP),
    "ok": bool(final) and _ok(final) and error is None,
    "wall_s": round(wall, 1),
    "prompt_tokens": getattr(agent, "session_prompt_tokens", 0),
    "completion_tokens": getattr(agent, "session_completion_tokens", 0),
    "total_tokens": getattr(agent, "session_total_tokens", 0),
    "api_calls": len([
        m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
    ]),
    "tool_calls": len(tool_calls),
    "tool_call_names": tool_calls,
    "browser_schema_chars": schema_desc_len,
    "error": error,
    "final_snippet": (final or "")[-400:],
    "task_file_sha256": hashlib.sha256(
        open(TASKS_PATH, "rb").read()
    ).hexdigest(),
    "runtime_tree": WT,
    "runtime_commit": subprocess.run(
        ["git", "-C", WT, "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip(),
    "model_provider": MODEL_PROVIDER,
    "model_base_url_host": urlsplit(MODEL_BASE_URL).netloc,
}
print("RESULT_JSON:" + json.dumps(out, ensure_ascii=False))
