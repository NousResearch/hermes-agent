"""Offline real-worktree prompt prefix probe (no provider requests).

Run with the checkout's Python and PROBE_OUT pointing at a new empty directory.
The subprocesses use isolated HOME/HERMES_HOME and a localhost dummy provider.
"""

import os, sys, json, subprocess, pathlib, hashlib, dataclasses

BASE = pathlib.Path(os.environ["PROBE_OUT"])
SRC = pathlib.Path(__file__).resolve().parents[2]


def git(*args, cwd):
    p = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return p.stdout.strip()


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


if len(sys.argv) == 1:
    BASE.mkdir(parents=True, exist_ok=True)
    home = BASE / "home"
    home.mkdir(exist_ok=True)
    hh = home / "hermes"
    hh.mkdir(exist_ok=True)
    (hh / "config.yaml").write_text(
        "context_file_max_chars: 100000\nagent:\n  coding_context: auto\n  environment_probe: false\n"
    )
    fixture = BASE / "fixture"
    fixture.mkdir(exist_ok=True)
    git("init", "-b", "main", cwd=fixture)
    content = "# Shared project guidance\n" + (
        "Use deterministic local operations and preserve project conventions.\n" * 1100
    )
    content = content[:66000]
    (fixture / "AGENTS.md").write_text(content)
    (fixture / "app.py").write_text('print("fixture")\n')
    git("add", "AGENTS.md", "app.py", cwd=fixture)
    git(
        "-c",
        "user.name=Prompt Probe",
        "-c",
        "user.email=probe@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "fixture",
        cwd=fixture,
    )
    for name in ("worktree-a", "worktree-b"):
        git("worktree", "add", "-b", name, str(BASE / name), cwd=fixture)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "HERMES_HOME": str(hh),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(SRC),
        "PYTHONHASHSEED": "0",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "TMPDIR": str(BASE),
        "PROBE_OUT": str(BASE),
    }
    for name in ("worktree-a", "worktree-b"):
        e = env | {"TERMINAL_CWD": str(BASE / name), "TERMINAL_ENV": "local"}
        with (BASE / (name + ".log")).open("w") as log:
            p = subprocess.run(
                [sys.executable, "-B", __file__, name],
                env=e,
                cwd=BASE / name,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=150,
            )
        print(name, "exit", p.returncode, flush=True)
        if p.returncode:
            raise SystemExit(p.returncode)
    a, b = [
        json.loads((BASE / (name + ".json")).read_text())
        for name in ("worktree-a", "worktree-b")
    ]

    def lcp(x, y):
        return next(
            (i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y))
        )

    diff = {
        "source_sha": git("rev-parse", "HEAD", cwd=SRC),
        "source_dirty": bool(git("status", "--porcelain", cwd=SRC)),
        "source_diff_sha256": digest(git("diff", "HEAD", cwd=SRC)),
        "fixture_git_worktrees": git("worktree", "list", "--porcelain", cwd=fixture),
        "context_file_chars": len(content),
        "context_file_sha256": digest(content),
        "stable_equal": a["parts"]["stable"] == b["parts"]["stable"],
        "full_common_prefix_chars": lcp(a["full"], b["full"]),
        "stable_common_prefix_chars": lcp(a["parts"]["stable"], b["parts"]["stable"]),
        "project_context_equal": a["project_context"] == b["project_context"],
        "same_agent_rebuild_equal": [
            a["same_agent_rebuild_equal"],
            b["same_agent_rebuild_equal"],
        ],
        "per_worktree": [x["metrics"] for x in (a, b)],
        "provider_usage": None,
        "billing_measured": False,
    }
    (BASE / "measurements.json").write_text(json.dumps(diff, indent=2))
    print(json.dumps(diff, indent=2))
else:
    # Fresh credential-free process, real production imports; no prompt seams mocked.
    sys.path.insert(0, str(SRC))
    from run_agent import AIAgent
    from agent.system_prompt import build_system_prompt, build_system_prompt_parts
    from agent.prompt_builder import build_context_files_prompt
    from agent.prompt_caching import build_prompt_cache_plan

    agent = AIAgent(
        api_key="offline-not-a-credential",
        base_url="http://127.0.0.1:9/v1",
        provider="openai-compat",
        model="offline-probe",
        enabled_toolsets=["terminal", "file"],
        quiet_mode=True,
        skip_context_files=False,
        skip_memory=True,
        skip_background_review=True,
        save_trajectories=False,
        platform="cli",
        session_id="20260907_120000_probe",
    )
    parts = build_system_prompt_parts(agent)
    full = build_system_prompt(agent)
    repeated = build_system_prompt(agent)
    project = build_context_files_prompt(cwd=os.getcwd(), skip_soul=True)
    plan = build_prompt_cache_plan(
        [
            {"role": "system", "content": full},
            {"role": "user", "content": "Offline prompt inspection only."},
        ],
        agent.tools,
        static_system_prefix=agent._cached_system_prompt_static,
    )
    plan_data = dataclasses.asdict(plan)
    from agent.conversation_loop import _stored_prompt_matches_runtime

    decoy = "\n\nOperator instructions:\nHost: Example\nUser home directory: /example\nCurrent working directory: /example\n"
    restore_matches = _stored_prompt_matches_runtime(agent, full)
    trailing_decoy_matches = _stored_prompt_matches_runtime(agent, full + decoy)
    (BASE / "different-worktree").mkdir(exist_ok=True)
    os.environ["TERMINAL_CWD"] = str(BASE / "different-worktree")
    drift_rejected = not _stored_prompt_matches_runtime(agent, full)
    os.environ["TERMINAL_CWD"] = os.getcwd()
    name = sys.argv[1]
    metrics = {
        "restore_matches": restore_matches,
        "trailing_decoy_matches": trailing_decoy_matches,
        "drift_rejected": drift_rejected,
        "cwd": os.getcwd(),
        "full_chars": len(full),
        "stable_chars": len(parts["stable"]),
        "context_chars": len(parts["context"]),
        "project_context_chars": len(project),
        "cwd_line_offset": full.index("Current working directory:"),
        "workspace_offset": full.find("Workspace (snapshot at session start"),
        "root_offset": full.find("- Root:"),
        "project_context_offset": full.index("# Project Context"),
        "stable_has_cwd": os.getcwd() in parts["stable"],
        "stable_sha256": digest(parts["stable"]),
        "prompt_sha256": digest(full),
        "valid_tool_names": sorted(agent.valid_tool_names),
    }
    out = {
        "parts": parts,
        "full": full,
        "project_context": project,
        "same_agent_rebuild_equal": full == repeated,
        "metrics": metrics,
        "cache_plan": plan_data,
    }
    (BASE / (name + ".json")).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(metrics, indent=2))
