"""Behavior-parity check for the browser-provider plugin migration (#25214).

Spawns one subprocess per (version, scenario) cell — pinned to either
origin/main (legacy in-tree providers + class-instantiation lookup) or
this PR's worktree (plugin-based registry) via `sys.path[0]`. Each
subprocess clears all browser-related env vars + writes a config.yaml,
loads `tools.browser_tool._get_cloud_provider()`, and emits a reduced
"shape tuple" {is_local, provider_name, is_available} as JSON.

The parent process diffs the shapes per scenario. A diff means the
migration introduced an observable behaviour change vs origin/main —
which would be a real regression for users on the existing config keys.

Run from the PR worktree:

    cd ~/.hermes/hermes-agent/.worktrees/browser-providers-plugin
    python tests/plugins/browser/check_parity_vs_main.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


# Pin one path to current main, one to the PR worktree.
# ``REPO_ROOT`` is ``.../.worktrees/browser-providers-plugin``; the main
# checkout lives two levels up at ``~/.hermes/hermes-agent``.
MAIN_DIR = REPO_ROOT.parent.parent  # ~/.hermes/hermes-agent
PR_DIR = REPO_ROOT  # the worktree we're in
assert (MAIN_DIR / "tools" / "browser_tool.py").exists(), (
    f"MAIN_DIR={MAIN_DIR} doesn't look like a hermes-agent checkout"
)
assert (PR_DIR / "tools" / "browser_tool.py").exists(), (
    f"PR_DIR={PR_DIR} doesn't look like a hermes-agent checkout"
)


# Reduced shape comparison — exact instance addresses obviously differ
# between subprocesses, so we compare the parts that matter for users.
SUBPROCESS_SCRIPT = r"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])

# Isolated HERMES_HOME for the config write.
home = tempfile.mkdtemp()
os.environ["HERMES_HOME"] = home

# Clear every browser-related env var so is_available() is deterministic.
for k in (
    "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID", "BROWSERBASE_BASE_URL",
    "BROWSER_USE_API_KEY", "BROWSER_USE_GATEWAY_URL",
    "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "FIRECRAWL_BROWSER_TTL",
    "TOOL_GATEWAY_DOMAIN", "TOOL_GATEWAY_USER_TOKEN",
):
    os.environ.pop(k, None)

# Apply per-scenario env (passed as JSON via argv[2]).
scenario_env = json.loads(sys.argv[2])
os.environ.update(scenario_env)

# Apply per-scenario config (passed as YAML body via argv[3]).
config_yaml = sys.argv[3]
config_path = os.path.join(home, "config.yaml")
with open(config_path, "w") as f:
    f.write(config_yaml)

# Fresh import — must not have any browser modules cached.
for name in list(sys.modules):
    if name.startswith("tools.") or name.startswith("agent.") or name.startswith("plugins."):
        sys.modules.pop(name, None)

from tools.browser_tool import _get_cloud_provider, _is_local_mode

provider = _get_cloud_provider()

# Pull the human-readable backend name via the API that exists on BOTH
# legacy (origin/main: CloudBrowserProvider.provider_name()) and the new
# ABC (BrowserProvider exposes provider_name() as a backward-compat alias
# returning display_name). Both shapes resolve to the same string —
# 'Browserbase' / 'Browser Use' / 'Firecrawl' — so we can compare safely.
provider_name = None
is_available = None
if provider is not None:
    pn = getattr(provider, "provider_name", None)
    if callable(pn):
        provider_name = pn()
    elif isinstance(pn, str):
        provider_name = pn
    is_conf = getattr(provider, "is_configured", None)
    if callable(is_conf):
        is_available = bool(is_conf())

shape = {
    "is_local": _is_local_mode(),
    "provider_name": provider_name,
    "is_available": is_available,
}
print(json.dumps(shape))
"""


SCENARIOS: list[tuple[str, str, dict[str, str]]] = [
    # (label, config.yaml body, extra env vars)
    ("no-config-no-env", "", {}),
    ("explicit-local-no-env", "browser:\n  cloud_provider: local\n", {}),
    (
        "explicit-browserbase-no-creds",
        "browser:\n  cloud_provider: browserbase\n",
        {},
    ),
    (
        "explicit-browserbase-with-creds",
        "browser:\n  cloud_provider: browserbase\n",
        {"BROWSERBASE_API_KEY": "x", "BROWSERBASE_PROJECT_ID": "y"},
    ),
    (
        "explicit-browser-use-no-creds",
        "browser:\n  cloud_provider: browser-use\n",
        {},
    ),
    (
        "explicit-browser-use-with-creds",
        "browser:\n  cloud_provider: browser-use\n",
        {"BROWSER_USE_API_KEY": "k"},
    ),
    (
        "explicit-firecrawl-no-creds",
        "browser:\n  cloud_provider: firecrawl\n",
        {},
    ),
    (
        "explicit-firecrawl-with-creds",
        "browser:\n  cloud_provider: firecrawl\n",
        {"FIRECRAWL_API_KEY": "k"},
    ),
    (
        "no-config-bu-creds",
        "",
        {"BROWSER_USE_API_KEY": "k"},
    ),
    (
        "no-config-bb-creds",
        "",
        {"BROWSERBASE_API_KEY": "x", "BROWSERBASE_PROJECT_ID": "y"},
    ),
    (
        "no-config-both-creds",
        "",
        {
            "BROWSER_USE_API_KEY": "k",
            "BROWSERBASE_API_KEY": "x",
            "BROWSERBASE_PROJECT_ID": "y",
        },
    ),
    (
        "no-config-firecrawl-only",
        "",
        {"FIRECRAWL_API_KEY": "k"},
    ),
    (
        "no-config-firecrawl-and-bb",
        "",
        {
            "FIRECRAWL_API_KEY": "k",
            "BROWSERBASE_API_KEY": "x",
            "BROWSERBASE_PROJECT_ID": "y",
        },
    ),
]


def _run_scenario(repo_path: Path, label: str, config_yaml: str, env: dict) -> dict:
    """Run one (version, scenario) cell. Returns the shape dict."""
    venv_python = repo_path / ".venv" / "bin" / "python"
    if not venv_python.exists():
        # Worktrees share the main repo's venv.
        venv_python = MAIN_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path("python3")

    out = subprocess.run(
        [
            str(venv_python),
            "-c",
            SUBPROCESS_SCRIPT,
            str(repo_path),
            json.dumps(env),
            config_yaml,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        return {
            "error": "subprocess failed",
            "stdout": out.stdout,
            "stderr": out.stderr[-500:],
        }
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"error": f"could not parse output: {exc}", "stdout": out.stdout}


def _reduce_for_comparison(shape: dict) -> dict:
    """Reduce a shape dict to the parts that matter for user-visible parity.

    We compare ``(is_local, provider_name, is_available)`` — the trio that
    decides what the dispatcher does with each tool call. ``provider_name``
    is the legacy ``provider_name()`` return value ('Browserbase' / 'Browser
    Use' / 'Firecrawl'), which is identical between legacy and plugin
    classes (the plugin's ``display_name`` matches the legacy
    ``provider_name()`` return).
    """
    return {
        "is_local": shape.get("is_local"),
        "provider_name": shape.get("provider_name"),
        "is_available": shape.get("is_available"),
    }


def main() -> int:
    print(f"main:    {MAIN_DIR}")
    print(f"pr:      {PR_DIR}")
    print()

    failures: list[str] = []
    errors: list[str] = []
    for label, config_yaml, env in SCENARIOS:
        main_shape = _run_scenario(MAIN_DIR, label, config_yaml, env)
        pr_shape = _run_scenario(PR_DIR, label, config_yaml, env)

        if "error" in main_shape or "error" in pr_shape:
            print(f"  [ERR ] {label}: subprocess failed")
            print(f"    main: {main_shape}")
            print(f"    pr:   {pr_shape}")
            errors.append(label)
            continue

        main_reduced = _reduce_for_comparison(main_shape)
        pr_reduced = _reduce_for_comparison(pr_shape)

        if main_reduced == pr_reduced:
            print(f"  [OK]   {label}: {main_reduced}")
        else:
            print(f"  [FAIL] {label}")
            print(f"    main: {main_reduced}")
            print(f"    pr:   {pr_reduced}")
            failures.append(label)

    print()
    if errors:
        print(f"SUBPROCESS ERRORS in {len(errors)} scenario(s):")
        for e in errors:
            print(f"  - {e}")
    if failures:
        print(f"BEHAVIOUR REGRESSION in {len(failures)} scenario(s):")
        for f in failures:
            print(f"  - {f}")
    if failures or errors:
        return 1
    print(f"PARITY OK across {len(SCENARIOS)} scenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
