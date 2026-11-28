## What does this PR do?

A keyless provider row with a Python post-setup (DuckDuckGo `ddgs`, faster-whisper, KittenTTS, Piper) reads
`needs_setup` in `GET /api/tools/toolsets/<name>/config` until Hermes restarts, even after its install finished.

On upstream/main 0431192d69:

- The row's status comes from `_POST_SETUP_READY` (`hermes_cli/tools_config_post_setup.py:352`), which is
  `_module_installed(module)` (`:315`), an `importlib.util.find_spec` in the dashboard's own process;
  `hermes_cli/tools_config_providers.py:208` maps it to `"ready" if predicate() else "needs_setup"`.
- The post-setup installs through PM into a new dependency generation (`_post_setup_python`, `:114`;
  `pm.sync_venv([spec["extra"]], explicit=True)` at `:121`) and prints "Restart Hermes to use them." (`:126`).
- A running process never switches generations (`pm/environments.py:463-467`, `activate_dependencies`:
  "Already-running processes are never switched after a dependency install."). So the dashboard that ran the
  install keeps answering `needs_setup`, and a client offers Install again right after a successful install.

Reproduced with a dashboard started before `hermes tools post-setup ddgs` on a fresh home: `needs_setup` before the
install, right after "Post-setup 'ddgs' complete" and 5 s later; `ready` only after a restart.

The fix adds a row status, `needs_restart`, for "installed, in use after a restart":

- `_module_in_committed_generation(module)` looks for the module in the site-packages of the generation PM
  committed for this install (`pm.environments.committed_venv`, the one the next boot loads) with
  `importlib.machinery.PathFinder.find_spec`, so nothing is imported. An unreadable selection record counts as no
  install. `post_setup_missing_status(key)` returns `needs_restart` for a Python post-setup whose module that
  generation holds, else `needs_setup`.
- When the installed-check fails, `provider_readiness_status` returns `post_setup_missing_status(post_setup)`
  instead of a fixed `needs_setup`.

`ready`, `needs_keys`, `needs_auth` and the non-Python post-setups keep their meaning; a module in neither the
process nor the committed generation still reads `needs_setup`.

Note for the desktop app: `ToolProviderStatus` (`apps/desktop/src/types/hermes.ts:1267`) does not list
`needs_restart`, and `toolset-config-panel.tsx` shows "installed" only for `ready`, so the panel draws no pill for
the new status and keeps its Install button. A follow-up can give it its own pill; this PR keeps to the server.

## Related Issue

None for this status. Related: #75118 (post-setup can break a running gateway by upgrading dependencies), which
is about the same "a running process does not switch generations" boundary.

## Type of Change

- [x] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 🔒 Security fix (undisclosed vulnerabilities in [SECURITY.md](https://github.com/NousResearch/hermes-agent/blob/main/SECURITY.md) §3.1 scope go to [Security Advisories](https://github.com/NousResearch/hermes-agent/security/advisories/new) first, not a public PR)
- [ ] 📝 Documentation update
- [ ] ✅ Tests (adding or improving test coverage)
- [ ] ♻️ Refactor (no behavior change)
- [ ] 🎯 New skill (bundled or hub)

## Changes Made

- `hermes_cli/tools_config_post_setup.py`: `_module_in_committed_generation` and `post_setup_missing_status`.
- `hermes_cli/tools_config_providers.py`: `provider_readiness_status` returns `post_setup_missing_status(...)` when
  the installed-check fails.
- `tests/hermes_cli/test_tools_config_post_setup.py`: `test_a_finished_install_this_process_cannot_load_needs_a_restart`.

## How to Test

1. With `_module_installed` patched to False and a committed generation written under the test's HERMES_HOME, the
   DuckDuckGo row reads `needs_restart` when the generation holds `ddgs`, and `needs_setup` when it holds another
   module or nothing is committed:
   `pytest tests/hermes_cli/test_tools_config_post_setup.py -q`
2. On upstream/main: 1 failed (`'needs_setup' == 'needs_restart'`). With this change: passes.
3. Neighbours pass unchanged (291 passed, 6 skipped): `tests/hermes_cli/test_web_server.py`,
   `test_tools_config.py`, `test_post_setup_gating.py`, `test_web_routers_tools_install_on_enable.py`,
   `test_imagegen_managed_gateway.py`, `test_image_gen_picker.py`, `tests/tools/test_web_picker_active_rows.py`.
4. Live: on a fresh home, a dashboard started before `hermes tools post-setup ddgs` answers `needs_restart` right
   after the install and `ready` after a restart; without the fix it answers `needs_setup` until the restart.

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(scope):`, `feat(scope):`, etc.)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [ ] I've run `pytest tests/ -q` and all tests pass (not the whole suite: the changed tests and their neighbours, listed under How to Test)
- [x] `python scripts/check` passes (the blocking lint and code-health checks CI runs)
- [x] I've added tests for my changes (required for bug fixes, strongly encouraged for features)
- [x] I've tested on my platform: Ubuntu (Linux), Python 3.14

### Documentation & Housekeeping

- [x] I've updated relevant documentation (README, `docs/`, docstrings), or N/A: N/A
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys, or N/A: N/A
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows, or N/A: N/A
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility), or N/A: the generation and its site-packages come from pm.environments (committed_venv, site_packages), which already own the per-platform layout
- [x] I've updated tool descriptions/schemas if I changed tool behavior, or N/A: N/A

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01851juoqk7upryW25W6xyoX
