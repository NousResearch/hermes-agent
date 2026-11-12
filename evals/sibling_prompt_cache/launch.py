"""Resolve a route under the real home, then run worker.py with the credential passed by environment only.

  PYTHONPATH=<tree> python launch.py <route:nous|openai-codex|openrouter> <model> <arm> <out.jsonl> <tree>
"""
import os, subprocess, sys
from hermes_cli.runtime_provider import resolve_runtime_provider
route, model, arm, out, tree = sys.argv[1:6]
rt = resolve_runtime_provider(requested=route)
env = dict(os.environ, SC_TREE=tree, SC_URL=rt["base_url"], SC_KEY=rt["api_key"], SC_PROVIDER=route,
           SC_API_MODE=rt.get("api_mode") or "chat_completions", SC_MODEL=model)
env.pop("PYTHONPATH", None)
worker = os.path.join(os.path.dirname(__file__), "worker.py")
sys.exit(subprocess.run([sys.executable, worker, arm, out], env=env, check=False, timeout=900).returncode)
