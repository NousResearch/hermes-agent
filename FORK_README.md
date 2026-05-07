# Hermes Agent — Multi-Provider Memory Fork

> A fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
> that adds **multi-provider memory support** — run multiple memory backends
> simultaneously instead of being locked to a single provider.

**Fork:** [someaka/hermes-agent](https://github.com/someaka/hermes-agent) —
branch `feat/multi-provider-memory`

**Base:** upstream `main` at tag `v2026.5.7`

---

## What this adds

Upstream Hermes Agent only supports one memory provider at a time
(`memory.provider: mnemosyne`). This fork adds:

1. **Multi-provider config key** — `memory.providers` (list) alongside the
   legacy `memory.provider` (string, still works as fallback)

2. **Concurrent provider loading** — `MemoryManager.add_provider()` registers
   all configured providers at agent startup; each gets its own tool namespace

3. **Thread-safe MemoryManager** — `RLock`-guarded registration + removal;
   safe for concurrent subagent and gateway use

4. **Runtime provider removal** — `remove_provider()` cleanly deregisters a
   provider: cancels background tasks, releases tools, updates config

5. **Provider removal UX** — CLI checklist, curses picker, and web API
   endpoint for removing providers interactively

6. **Schema-first registration** — `add_provider()` wraps `get_tool_schemas()`
   in try/except so a provider that fails schema loading is never registered

7. **Tool budget warning** — warns at 20+ active tools across all providers

8. **Namespace-prefixed tools** — each provider's tools are prefixed with its
   name (e.g. `holographic_store`, `mem0_search`) to avoid collisions

9. **91 tests** covering all of the above

## Install

### From fork (recommended)

```bash
uv pip install "hermes-agent @ git+https://github.com/someaka/hermes-agent@feat/multi-provider-memory"
```

Or clone and install locally:

```bash
git clone -b feat/multi-provider-memory https://github.com/someaka/hermes-agent
cd hermes-agent
uv pip install -e ".[all]"
```

### Configure multi-provider memory

Edit `~/.hermes/config.yaml`:

```yaml
memory:
  providers:
    - mnemosyne
    - holographic
    - mem0
```

Or use the CLI:

```bash
hermes setup   # memory provider step now supports multi-select
```

Legacy single-provider config still works:

```yaml
memory:
  provider: mnemosyne
```

### Install upstream + patch

If you prefer to track upstream releases:

```bash
# Install upstream
uv pip install "hermes-agent @ git+https://github.com/NousResearch/hermes-agent@v2026.5.7"

# Apply fork changes
git remote add someaka https://github.com/someaka/hermes-agent
git fetch someaka
git cherry-pick someaka/feat/multi-provider-memory
```

> **Note:** cherry-pick may conflict if upstream has diverged since the
> fork's base. In that case, read the diff and apply manually — the
> changes are concentrated in `agent/memory_manager.py`,
> `plugins/memory/__init__.py`, `run_agent.py`, and `hermes_cli/plugins_cmd.py`.

## Upstream sync strategy

This fork rebases on upstream release tags. When a new upstream tag drops:

```bash
git fetch upstream
git rebase --onto v<NEW_TAG> v<OLD_TAG> feat/multi-provider-memory
# resolve conflicts (usually 1-3 files)
# run tests: uv run pytest tests/ -x --timeout=60
git tag v<NEW_TAG>-multi-mem
```

Conflicts are typically in:
- `hermes_cli/plugins_cmd.py` (CLI config helpers)
- `plugins/memory/__init__.py` (provider discovery)
- `run_agent.py` (agent init)

The rebase strategy keeps the fork linear and auditable — no merge commits
polluting the history.

## Maintenance commitment

This fork tracks upstream releases on a best-effort basis. The goal is to
get multi-provider memory merged upstream, making this fork unnecessary.

**Scope:** only multi-provider memory features. No unrelated changes, no
experimental branches, no divergent opinions on upstream architecture.

**Stability:** the 31 commits on this branch are clean, tested (91/91 pass),
and designed for easy cherry-pick or upstream merge.

## Commit summary

```
4b647518c build: add dev dependency group with server and test utilities
0b2e841f5 feat(memory): add 'Remove a provider...' option to setup wizard
f1450da0d feat(memory): add provider removal UX — CLI, curses checklist, web API
ffc088aa0 fix: memory setup wizard asks add-vs-replace when providers are already active
a9bc363c8 fix(cli): show all active memory providers in UI, not just first
f927fdc49 test(agent): add performance regression tests for MemoryManager
e1ede63d0 test(acp): skip adapter tests when agent-client-protocol is missing
27bb73572 feat(agent): make MemoryManager thread-safe + add concurrency tests
7f059cbf9 test(agent): cover remove_provider() shutdown exception path
7c54feb75 refactor(memory): remove dead code _get_active_memory_provider()
51b1b7654 fix(tests): update test assertions after upstream rebase
d0f9c0e87 chore: remove implementation plan files (not for upstream)
25375d92e test: add memory setup and get_active_memory_providers config tests
a159f831f test(agent): add schema failure, namespace, and budget warning tests
823cd973e fix(cli): multi-provider compat in dump, plugins_cmd, honcho setup
a35a8a696 fix(agent): wrap schema loading in try/except to prevent registration on failure
1874ebaf1 fix(tests): update plugin CLI tests for multi-provider API
829176646 fix(cli): hermes doctor iterates all active memory providers
4082c1d02 fix(docs): update stale single-provider references
a4abf1fbd fix(plugins): use load_config, multi-provider CLI support
3800166a6 test(agent): add multi-provider memory tests
a1c67d31e refactor(plugins): rename holographic tools with provider prefix and aliases
d58452ba5 feat(agent): add toolset filtering for memory provider tools
96c5119a4 feat(agent): add tool budget warning and namespace validation
ccdc911d0 feat(agent): add remove_provider() for runtime provider deregistration
72b27548c feat(agent): remove single-external-provider guard for multi-provider support
778eb1295 feat(agent): load all configured memory providers in agent init
47314d8c7 feat(config): add memory.providers list key for multi-provider support
307dbae56 feat(plugins): add get_active_memory_providers() for multi-provider loading
6fc02f234 fix(agent): capture on_pre_compress return value
afaf1cbac docs: add multi-provider memory implementation plans
```

## License

Same as upstream: [MIT](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE)
