# AgentCyber Repair Audit

Timestamp (UTC): `2026-06-21T13:34:03Z`
Repo: `/home/kbun/Desktop/hermes-agentcyber`
Plan: `/home/kbun/Desktop/hermes-agentcyber/.hermes/plans/2026-06-21_0918-repair-agentcyber-standalone-edition.md`

## Scope and safety boundary

This audit is the pre-repair baseline for restoring AgentCyber as a standalone Hermes Agent Cyber Edition. It records the current repository, runtime status, visible cyber toolsets, and focused regression tests before runtime-boundary implementation.

No default Hermes config, gateway, profile, cron job, credential file, or service was modified by this audit.

## Repository state

Command:

```bash
git status --short
git branch --show-current
git log -1 --oneline
git remote -v
git rev-list --count HEAD..upstream/main
git rev-list --count upstream/main..HEAD
git rev-list --count HEAD..origin/main
git rev-list --count origin/main..HEAD
```

Output:

```text
## git status --short
?? docs/AGENTCYBER_REPAIR_CRON_LEDGER.md
## git branch --show-current
main
## git log -1 --oneline
977a476bc Merge pull request #20 from breakingcircuits1337/docs/agentcyber-breakglass-operator-workflow
## git remote -v
origin	https://github.com/breakingcircuits1337/hermes-agentcyber.git (fetch)
origin	git@github.com:breakingcircuits1337/hermes-agentcyber.git (push)
upstream	https://github.com/NousResearch/hermes-agent.git (fetch)
upstream	DISABLED (push)
## git rev-list --count HEAD..upstream/main
0
## git rev-list --count upstream/main..HEAD
59
## git rev-list --count HEAD..origin/main
0
## git rev-list --count origin/main..HEAD
0
```

Notes:

- The checkout is on `main` at `977a476bc`.
- The only pre-existing dirty item observed in this run was the untracked repair ledger: `docs/AGENTCYBER_REPAIR_CRON_LEDGER.md`.
- The fork is even with `origin/main` and 59 commits ahead of `upstream/main`.

## Current AgentCyber status command

Command:

```bash
uv run --frozen hermes agentcyber status --json
```

Output:

```json
{
  "agent_cyber": {
    "allow_hosted_override": true,
    "local_open_weight": {
      "api_key_env": "",
      "api_key_present": false,
      "api_mode": "chat_completions",
      "base_url_present": true,
      "context_length": 131072,
      "model": "qwen3-coder:30b",
      "provider": "ollama"
    },
    "require_local_for_sensitive": true,
    "routing_enabled": true
  },
  "assets": {
    "builtin_enabled": true,
    "count": 3,
    "source": "builtin:breaking-circuits"
  },
  "git": {
    "ahead_upstream_main": 59,
    "behind_origin_main": 0,
    "behind_upstream_main": 0,
    "branch": "main",
    "dirty": true,
    "head": "977a476bcb573ad76ea06b190369835443a9cdeb"
  },
  "local_runtime_health": {
    "model_present": true,
    "models_count": 7,
    "ok": true,
    "url": "http://192.168.1.120:11434/api/tags"
  },
  "toolsets": {
    "cyber_enabled": true,
    "cyber_visible": true,
    "live_usb_enabled": false,
    "live_usb_visible": true,
    "platform": "cli",
    "registered_tools": [
      "extract_iocs",
      "ir_incident",
      "live_usb",
      "network_scan",
      "threat_intel",
      "vuln_triage"
    ]
  }
}
```

The command emitted this environment warning before the JSON:

```text
warning: `VIRTUAL_ENV=/home/kbun/.hermes/hermes-agent/venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
```

Notes:

- The AgentCyber status command exists and reports routing enabled.
- Local/open-weight routing points at Ollama model `qwen3-coder:30b` and the model health check was green at audit time.
- `cyber` is visible/enabled; `live_usb` is visible/disabled.
- `git.dirty` is true because this repair run has an untracked ledger file.
- This command was still run from the normal process environment. It proves the repo command works, but it does **not** yet prove a standalone `HERMES_HOME` boundary.

## Toolset visibility

Command:

```bash
uv run --frozen hermes tools list | grep -Ei 'cyber|live_usb|enabled|disabled'
```

Relevant output:

```text
  ✓ enabled  web  🔍 Web Search & Scraping
  ✓ enabled  browser  🌐 Browser Automation
  ✓ enabled  terminal  💻 Terminal & Processes
  ✓ enabled  file  📁 File Operations
  ✓ enabled  code_execution  ⚡ Code Execution
  ✓ enabled  vision  👁️  Vision / Image Analysis
  ✗ disabled  video  🎬 Video Analysis
  ✓ enabled  image_gen  🎨 Image Generation
  ✗ disabled  video_gen  🎬 Video Generation
  ✗ disabled  x_search  🐦 X (Twitter) Search
  ✗ disabled  moa  🧠 Mixture of Agents
  ✓ enabled  tts  🔊 Text-to-Speech
  ✓ enabled  skills  📚 Skills
  ✓ enabled  todo  📋 Task Planning
  ✓ enabled  memory  💾 Memory
  ✗ disabled  context_engine  🧩 Context Engine
  ✓ enabled  session_search  🔎 Session Search
  ✓ enabled  clarify  ❓ Clarifying Questions
  ✓ enabled  delegation  👥 Task Delegation
  ✓ enabled  cronjob  ⏰ Cron Jobs
  ✗ disabled  homeassistant  🏠 Home Assistant
  ✗ disabled  spotify  🎵 Spotify
  ✓ enabled  cyber  🛡️  AgentCyber Operations
  ✗ disabled  live_usb  💽 AgentCyber Live USB
  ✗ disabled  yuanbao  🤖 Yuanbao
  ✓ enabled  computer_use  🖱️  Computer Use (macOS)
  foundry  all tools enabled
  azure_manager  all tools enabled
```

## Focused regression baseline

Command:

```bash
uv run --frozen python -m pytest \
  tests/agent/test_cyber_routing.py \
  tests/agent/test_agentcyber_routing_guard.py \
  tests/agent/test_cyber_breakglass.py \
  tests/hermes_cli/test_agentcyber_cmd.py \
  -q -o addopts= --tb=short
```

Output:

```text
...........................                                              [100%]
=============================== warnings summary ===============================
tests/hermes_cli/test_agentcyber_cmd.py::test_agentcyber_status_reports_tool_visibility_and_safe_runtime
  /home/kbun/Desktop/hermes-agentcyber/.venv/lib/python3.11/site-packages/discord/player.py:30: DeprecationWarning: 'audioop' is deprecated and slated for removal in Python 3.13
    import audioop

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
27 passed, 1 warning in 1.67s
```

## Audit conclusion

AgentCyber is not empty or nonfunctional: the status command, routing config, local model health check, cyber toolset visibility, break-glass tests, and focused routing/gating tests are currently green from the repo command path.

The remaining repair target is the operator/runtime boundary: AgentCyber still needs an unmistakable standalone launch path that uses repo-local code and a dedicated `HERMES_HOME` instead of relying on whichever default Hermes environment happens to invoke the command.
