---
applyTo: "**"
description: >
  Patch-authoring guidance for changes inside the hermes-agent fork. Derived
  from the canonical Rosenblatt standard at the monorepo level and tailored to
  hermes-agent's extension points (auto-discovered tools, plugin system with
  lifecycle hooks, skills, gateway adapters, env-var config), code style, and
  file layout. Use for every PR that modifies this fork's source.
canonical_hash: 192b2b62cf077e90
canonical_source: rosenblatt monorepo — docs/standards/contribution/upstream-fork-contribution.md
generated_by: rosenblatt monorepo — .claude/skills/fork-instruction-sync
generated_at: 2026-05-01T18:10:00Z
fork_main_sha: 3ff3dfb5ac97c7a746d2c54a9b8eefb9f6279a75
---

> **Generated file — do not edit by hand.** Produced by the `fork-instruction-sync` skill in the Rosenblatt monorepo from the canonical standard at `docs/standards/contribution/upstream-fork-contribution.md`. Edits made directly here will be overwritten on the next sync. To change the guidance, edit the canonical doc and re-run the skill. Tier-routing rules (T1/T2/T3) live in the monorepo's `upstream-vs-harness.instructions.md` — this file only covers *how to write* a patch once you've decided it belongs in this fork.

# Contributing patches to hermes-agent

You are writing a patch that will land in the Rosenblatt fork of `NousResearch/hermes-agent`. Every line you change is a line a future maintainer (often you) will have to re-resolve when upstream evolves. The principles below all reduce to one thing: **make patches that survive upstream evolution with the least possible friction**.

## The five principles, in priority order

### 1. Use upstream's extensibility seams before patching core

`hermes-agent` has a deep extension surface, and most of it is **auto-discovered** — meaning you can add new functionality without editing any existing file. Before modifying anything, check whether one of these seams fits:

- **Tool registration (auto-discovery, no manual list edit).** Drop a new `*.py` file in `tools/` containing a top-level `registry.register(...)` call. The registry's `discover_builtin_tools()` function (in `tools/registry.py`) uses AST inspection to find every `tools/*.py` file with such a call and imports it at startup. Called from `model_tools.py` during module import. **You do not need to edit `model_tools.py` or any other registry file** to add a tool — just drop the file. Pattern (see `tools/web_tools.py` for a real example):
  ```python
  from tools.registry import registry

  def my_tool(...): ...

  MY_TOOL_SCHEMA = {"name": "my_tool", ...}

  registry.register(
      name="my_tool",
      toolset="my_toolset",
      schema=MY_TOOL_SCHEMA,
      handler=lambda args, **kw: my_tool(...),
      check_fn=...,
  )
  ```
  Files excluded from auto-discovery: `__init__.py`, `registry.py`, `mcp_tool.py`. MCP tools have their own discovery path (`tools/mcp_tool.py:discover_mcp_tools()`).

- **Full plugin system (auto-discovered, with lifecycle hooks).** `hermes_cli/plugins.py` implements a proper plugin system with four discovery sources, loaded in this order:
  1. **Bundled plugins:** built-in plugins shipped with `hermes-agent` under `<repo>/plugins/<name>/` (flat) or `<repo>/plugins/<category>/<backend>/` (category, e.g. `image_gen/openai/`). `memory/` and `context_engine/` are skipped at the top level — they have their own discovery (see below).
  2. **User plugins:** `$HERMES_HOME/plugins/<name>/` (default `~/.hermes/plugins/<name>/`)
  3. **Project plugins:** `./.hermes/plugins/<name>/` (opt-in via `HERMES_ENABLE_PROJECT_PLUGINS`)
  4. **Pip plugins:** Python packages exposing the `hermes_agent.plugins` entry-point group

  Later sources override earlier ones on key collision (user > bundled, project > user).

  Each directory plugin requires `plugin.yaml` (manifest) and `__init__.py` with a `register(ctx)` function. The `PluginContext` (in `hermes_cli/plugins.py`) lets plugins register tools (delegates to `tools.registry.register`), commands, and **lifecycle hooks** spanning tool calls, LLM calls, API requests, and session events. The authoritative list is `hermes_cli/plugins.py`'s `VALID_HOOKS` set — refer to it for the current hook names; new hooks land there as they're added upstream.

  This is the strongest extension seam in the codebase — a plugin needs zero source edits and gets full lifecycle access. **Strongly prefer this over patching agent internals.**

- **Bundled skills** (`skills/<category>/<skill>/SKILL.md`). Most "new capabilities" should be skills, not tools or core code — see `CONTRIBUTING.md`'s "Should it be a Skill or a Tool?" section. A skill is a SKILL.md plus optional `scripts/` and `references/`. Zero source patches.

- **Optional skills** (`optional-skills/`). Same structure as `skills/` but ship un-activated; users opt in via `hermes skills install`. Use this for skills that are official-quality but not universal.

- **Context-engine plugins** (`plugins/context_engine/<name>/`). Auto-discovered subdirectories implementing the `ContextEngine` ABC. One active at a time, selected via `context.engine` in `config.yaml`. See `plugins/context_engine/__init__.py`.

- **Memory-provider plugins** (`plugins/memory/<name>/`). Auto-discovered subdirectories implementing the `MemoryProvider` ABC. Bundled providers live in-tree; user-installed providers live in `$HERMES_HOME/plugins/<name>/` (note this is the same dir as the full plugin system above — bundled providers take precedence on name collisions). Selected via `memory.provider` in `config.yaml`. See `plugins/memory/__init__.py`.

- **Slash commands** (`hermes_cli/commands.py`). Add a `CommandDef` entry to the `COMMAND_REGISTRY` list. Aliases are an `aliases=("short",)` field on the existing `CommandDef`. New commands appear in autocomplete automatically.

- **Toolsets** (`toolsets.py`). Tools are grouped into named bundles (`web`, `terminal`, `file`, `browser`, etc.) that can be enabled per platform. Add a toolset entry rather than hardcoding tool lists at call sites.

- **Gateway platform adapters** (`gateway/platforms/`). Each messaging platform is a self-contained adapter inheriting from `gateway/platforms/base.py`. New platform support is a new file, not a patch to existing platforms.

- **Terminal execution backends** (`tools/environments/`). Backends (local, docker, ssh, singularity, modal, daytona) all subclass `BaseEnvironment`. New execution targets go here as new files.

- **`HERMES_*` env vars** (~30 in active use across the codebase: `HERMES_HOME`, `HERMES_ENABLE_PROJECT_PLUGINS`, `HERMES_AGENT_TIMEOUT`, `HERMES_GATEWAY_*`, `HERMES_SESSION_*`, `HERMES_INTERACTIVE`, `HERMES_YOLO_MODE`, `HERMES_TIMEZONE`, etc.). If your patch's behavior should be flippable or parameterizable, add a new `HERMES_*` env var in the relevant module rather than hardcoding. Document it in `cli-config.yaml.example` and `website/docs/user-guide/configuration.md` if user-facing.

- **Provider abstraction.** Hermes works with any OpenAI-compatible API. Provider resolution lives in `hermes_cli/auth.py` and `agent/auxiliary_client.py`. Custom providers should plug in via the existing resolution flow, not a parallel code path.

If your change can be expressed as one of the above, it should be — even when a direct edit would be slightly shorter. The verbosity is paid once; the merge debt is paid forever.

If after a thorough check no extension point fits, that's a real signal. Consider opening a T1 PR upstream that adds the extension point you need (the plugin system in particular has an extensible hook list — adding a hook is a clean upstream contribution), then implementing your feature against it.

### 2. Minimal diff principle

Every patch is the **smallest change that achieves the stated goal**. Specifically:

- **No reformatting** of existing code, even if it offends your style sensibilities. The upstream file's style is the contract.
- **No import reordering** unless your change requires a new import.
- **No whitespace-only changes**, including trailing-whitespace cleanup or line-ending normalization.
- **No "while I'm here" refactors.** Open a separate T1 PR if the refactor is genuinely valuable.
- **No dependency upgrades** bundled with feature work. `pyproject.toml` dependency bumps are their own T1 PR with their own review (Hermes pins to known-good ranges to limit supply-chain attack surface — bumps need careful review).
- **No test-framework changes** bundled with feature work. New tests using `pytest` are fine; switching frameworks is not.

A reviewer should be able to read your diff and answer "what does this change do?" in one sentence. If they can't, the diff is too big.

### 3. Modularity / isolation

When you must add Rosenblatt-specific code:

- **Prefer new files** over editing existing ones. New files are clean additions; edits to existing files are merge-conflict surface. Most extension seams above (tools registry, plugin system, gateway platforms, terminal backends) explicitly support "drop a new file" patterns — use them.
- **When you must edit an existing file**, change the minimum surface and delegate to a new file. Pattern: add one import + one call site that dispatches into your new module.
- **Avoid editing the high-churn files below** unless you genuinely have no other seam. These files change frequently upstream and any patch in them will conflict on most upstream merges:
  - `run_agent.py`
  - `gateway/run.py`
  - `cli.py`
  - `hermes_cli/main.py`, `hermes_cli/config.py`, `hermes_cli/setup.py`, `hermes_cli/models.py`, `hermes_cli/gateway.py`
  - `tools/terminal_tool.py`
  - `agent/auxiliary_client.py`
  - `gateway/platforms/base.py`, `gateway/platforms/telegram.py`, `gateway/platforms/discord.py`
  - `cron/scheduler.py`
  - `website/docs/user-guide/configuration.md`

  For changes that need to *invoke* logic in these files, add your logic in a new file under `tools/`, `plugins/`, `agent/`, `gateway/platforms/`, or in a plugin (via `hermes_cli/plugins.py`'s lifecycle hooks) and add the **smallest possible** dispatch hook (one import + one call) into the high-churn file — or, ideally, none at all if a plugin hook can intercept the relevant lifecycle event.
- **Group related patches into the smallest number of files.** One patched file with five edits is easier to merge than five patched files with one edit each.

### 4. Match upstream conventions

In this repo, **upstream's conventions win** — full stop:

- **Style:** PEP 8 with practical exceptions; line length is not strictly enforced. No formatter is run in CI (no Black, no Ruff format, no `pre-commit-config.yaml`) — match the surrounding file's apparent style. Don't run a formatter across files you didn't otherwise touch.
- **Comments:** Only when explaining non-obvious intent, trade-offs, or API quirks. Don't narrate what the code does.
- **Error handling:** Catch specific exceptions. Use `logger.warning()` / `logger.error()` (Python `logging`, not `print`). Pass `exc_info=True` for unexpected errors.
- **Cross-platform:** Never assume Unix. Test against Windows + macOS + Linux paths and processes wherever practical. Upstream values cross-platform portability highly.
- **Tests:** Use `pytest`. Tests live under `tests/` mirroring the source layout (`tests/agent/`, `tests/cli/`, `tests/gateway/`, etc.). Mark tests requiring external services with `@pytest.mark.integration`. CI runs `pytest tests/` against the unit-test set with `--ignore=tests/integration --ignore=tests/e2e -n auto` (see `.github/workflows/tests.yml` for the exact invocation, including any current `--tb=*` flags) — keep tests fast and parallelizable. The `[tool.pytest.ini_options]` `addopts` in `pyproject.toml` excludes the `integration` marker by default.
- **Type annotations:** Used inconsistently across the codebase — match the surrounding file. Don't introduce `mypy` strictness or annotation backfills as part of feature work (no `[tool.mypy]` config exists).
- **Enforced CI checks:** `tests.yml` (pytest), `contributor-check.yml`, `docker-publish.yml`, `nix.yml`, `supply-chain-audit.yml`, `docs-site-checks.yml`, `skills-index.yml`, `deploy-site.yml`. **No lint, format, or type-check workflow.** This means the burden of style consistency is on you as the patch author — read the file you're editing first.

### 5. Make patches conspicuous

Every Rosenblatt patch must be **trivially identifiable** by anyone (human or AI) doing a future upstream merge.

**Commit hygiene (monorepo-wide policy):**
- Prefix every Rosenblatt-only commit with `[rosenblatt]`. This is the standard across all forks — do not vary it. `git log --grep '\[rosenblatt\]'` must reliably surface every internal patch.
- Each commit message explains **why** the patch exists, not just what it does.
- Reference the upstream issue/PR (if T1-pending) or the internal ticket / Slack discussion (if T2-permanent).
- One logical change per commit.

**Code markers:**
- For non-trivial in-source patches, add sentinel comments marking the start and end of Rosenblatt-modified regions:
  ```python
  # === ROSENBLATT PATCH START: short description ===
  # Reason: why this patch exists
  # Upstream: link or "internal-only"
  ...patched code...
  # === ROSENBLATT PATCH END ===
  ```
- Sentinel comments are **mandatory** for any patch larger than ~10 lines or any patch in a high-churn file (see Principle 3).
- For one-line edits, an inline `# ROSENBLATT: <reason>` is sufficient.

These markers serve three purposes: (a) `git log -G "ROSENBLATT"` and `grep -r ROSENBLATT` find every patch in seconds, (b) merge-conflict resolvers immediately see the patch boundary, and (c) periodic audits to drop accepted-upstream patches become trivial.

## Forward-port mindset

Write every patch assuming it will be rebased on a future upstream version you haven't seen yet:

- **Depend on documented public APIs**, not private internals. The tools registry (`tools.registry.register`), the plugin system (`PluginContext`, `register(ctx)`, the `VALID_HOOKS` set), env vars, `hermes_constants.get_hermes_home()`, and the `BaseEnvironment` / `ContextEngine` / `MemoryProvider` ABCs are the safest dependencies. If you must use a private (leading-underscore) function, comment why and what the public alternative would be.
- **Pin upstream version assumptions** in the commit message ("requires hermes-agent ≥ 0.10 because the X plugin hook landed in 0.10.0") so a future merger knows when the patch can be re-evaluated.
- **Don't copy upstream code into your patch.** If you find yourself duplicating a function to wrap it, you're creating a future drift bug. Use a hook, subclass, or registry registration instead.

## Merge-debt awareness

Every line of fork patch is a permanent tax on every future upstream sync. Budget accordingly:

- **Audit `rosenblatt/main` patches** at every upstream version bump. For each patch: has upstream merged this? Is it still needed? Can it now be replaced by a newly-existing extension point or env var?
- **Drop patches as soon as upstream accepts the equivalent T1 PR.** The next `git merge upstream/main` is the natural place — accept upstream's version and remove your patch.
- **Re-bias toward harness (T3) at every audit.** Hermes adds new env vars, hooks, and config options frequently. A patch that was necessary six months ago may now be expressible as a `HERMES_*` env var, a plugin lifecycle hook, or a config option read by `entrypoint-hermes.sh` in the monorepo harness.

## When this guidance bends

A few cases override the principles above:

- **Security fixes.** Ship immediately, minimize later. Open the upstream PR in parallel.
- **Build breaks.** If upstream is broken on `main` and we need a working build, patch first, file the upstream issue second. Drop the patch as soon as upstream resolves the underlying issue.
- **`.gitmodules` rewrites for transitive forks** (e.g., the existing rewrite that points `tinker-atropos` at our fork). Structurally Rosenblatt-only, can't go upstream, no extensibility seam applies.

## This fork at a glance

| Aspect | Value |
|---|---|
| Upstream | [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent) |
| Language | Python ≥ 3.11 |
| Package manager | `uv` |
| Test framework | `pytest` (with `pytest-asyncio`, `pytest-xdist`); tests under `tests/` mirror source layout |
| Lint / format | None enforced. PEP 8 informally, no Black/Ruff/mypy/pre-commit in CI |
| CI workflows | `tests.yml`, `contributor-check.yml`, `docker-publish.yml`, `nix.yml`, `supply-chain-audit.yml`, `docs-site-checks.yml`, `skills-index.yml`, `deploy-site.yml` |
| Public extension surfaces | Auto-discovered tools registry, full plugin system with lifecycle hooks, bundled & optional skills, context-engine plugins, memory-provider plugins, slash-command registry, gateway platform adapters, terminal execution backends, `HERMES_*` env vars |
| Contributor docs | `CONTRIBUTING.md` (root), `AGENTS.md` (AI assistant guide), `website/docs/` |
| User config dir | `$HERMES_HOME` (default `~/.hermes/`). Resolved via `hermes_constants.get_hermes_home()` — **always document via the env var, never hardcode the fallback**. Layout: `config.yaml`, `.env`, `auth.json`, `skills/`, `memories/`, `state.db`, `sessions/`, `cron/`, `plugins/`. |
| User plugin dir | `$HERMES_HOME/plugins/<name>/` (resolved from the env var above). Used by both the full plugin system (`hermes_cli/plugins.py`) and the memory-provider plugin loader (with bundled providers winning name collisions). |
| Project plugin dir | `./.hermes/plugins/<name>/` (opt-in via `HERMES_ENABLE_PROJECT_PLUGINS`). |

## Cross-references

- **Canonical contribution standard:** `docs/standards/contribution/upstream-fork-contribution.md` in the Rosenblatt monorepo.
- **Tier classification (T1 / T2 / T3):** `.github/instructions/upstream-vs-harness.instructions.md` in the Rosenblatt monorepo.
- **Submodule sync mechanics:** `upstream/README.md` in the Rosenblatt monorepo.
- **Upstream contributor guide:** `CONTRIBUTING.md` and `AGENTS.md` in this repo.
