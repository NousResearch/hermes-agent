"""Resolve the Nous route under the real home, then run worker.py with the key passed by env only."""
import os, subprocess, sys
from hermes_agent.hermes_cli.runtime_provider import resolve_runtime_provider
arm, out, rep, tree = sys.argv[1:5]
rt = resolve_runtime_provider(requested="nous")
env = dict(os.environ, RL_TREE=tree, RL_URL=rt["base_url"], RL_KEY=rt["api_key"])
env.pop("PYTHONPATH", None)
worker = os.path.join(os.path.dirname(__file__), "worker.py")
sys.exit(subprocess.run([sys.executable, worker, arm, out, rep], env=env, check=False, timeout=600).returncode)
