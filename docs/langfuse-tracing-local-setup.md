# Langfuse Tracing for Hermes

Opt-in tracing plugin that sends LLM calls, tool calls, and per-turn spans to
Langfuse.  The plugin lives **outside** the hermes-agent repo so pulling
upstream updates never causes conflicts.

---

## Quick start (copy-paste recipe)

This gets you from zero to working traces.  Every command is meant to be run
in order in a single terminal session.

```bash
# ── 1. Prerequisites ──────────────────────────────────────────────────
cd /path/to/hermes-agent
source .venv/bin/activate
pip install langfuse                     # into the repo venv, not global

# ── 2. Fetch the plugin source ────────────────────────────────────────
# The plugin lives on the fork branch feat/langfuse_tracing.
# Pick ONE of the two fetch commands depending on your remote setup:

# (a) Your origin IS the fork (kshitijk4poor/hermes-agent):
git fetch origin feat/langfuse_tracing
PLUGIN_REF="origin/feat/langfuse_tracing"

# (b) Your origin is upstream (NousResearch/hermes-agent):
git fetch git@github.com:kshitijk4poor/hermes-agent.git \
  feat/langfuse_tracing:refs/remotes/fork/feat/langfuse_tracing
PLUGIN_REF="fork/feat/langfuse_tracing"

# ── 3. Determine your plugin directory ────────────────────────────────
# Hermes loads user plugins from $HERMES_HOME/plugins/.
# HERMES_HOME defaults to ~/.hermes for the default profile.
# If you use `hermes -p <name>`, it becomes ~/.hermes/profiles/<name>/.
# The CLI sets HERMES_HOME internally — it may not be in your shell env.

# Default profile:
PLUGIN_DIR="$HOME/.hermes/plugins/langfuse_tracing"

# Named profile (uncomment and edit):
# PLUGIN_DIR="$HOME/.hermes/profiles/<YOUR_PROFILE>/plugins/langfuse_tracing"

# ── 4. Install the plugin ────────────────────────────────────────────
mkdir -p "$PLUGIN_DIR"
git show "$PLUGIN_REF:.hermes/plugins/langfuse_tracing/__init__.py" \
  > "$PLUGIN_DIR/__init__.py"
git show "$PLUGIN_REF:.hermes/plugins/langfuse_tracing/plugin.yaml" \
  > "$PLUGIN_DIR/plugin.yaml"

# ── 5. Set credentials ───────────────────────────────────────────────
# Add these to your shell profile (~/.zshrc, ~/.bashrc, etc.) or .env.
# Tracing is completely dormant without them — no errors, no network calls.
export HERMES_LANGFUSE_ENABLED=true
export HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
export HERMES_LANGFUSE_SECRET_KEY=sk-lf-...

# ── 6. Verify ─────────────────────────────────────────────────────────
# Start a NEW terminal / hermes process (plugins load at startup only).
hermes plugins list                      # should show langfuse_tracing: enabled
HERMES_LANGFUSE_DEBUG=true hermes chat -q "hello"
# Look for: "Langfuse tracing: started trace ..." in stderr
```

That's it.  The plugin is outside the repo tree, so `git pull upstream main`
will never touch it.

---

## Updating hermes without breaking tracing

The plugin hooks into hermes via the standard plugin system and uses `**_` in
every hook signature to absorb new kwargs.  Per-API-call tracing uses
`pre_api_request` / `post_api_request` (not `pre_llm_call` / `post_llm_call`, which
are once per user turn).  Those hooks receive **summary fields only** (message
counts, tool counts, token usage dict, etc.) — not full `messages`, `tools`, or
raw provider `response` objects — so keep span metadata small and the contract
stable.

This means:

```bash
# Just pull upstream as usual
git fetch upstream
git merge upstream/main
# or: git pull upstream main
```

Nothing else is needed.  The plugin at `$PLUGIN_DIR` is not inside the repo,
so there are no merge conflicts.

### Updating the plugin itself

When the plugin code on `feat/langfuse_tracing` is updated:

```bash
git fetch origin feat/langfuse_tracing   # or the fork fetch from step 2b
git show "$PLUGIN_REF:.hermes/plugins/langfuse_tracing/__init__.py" \
  > "$PLUGIN_DIR/__init__.py"
git show "$PLUGIN_REF:.hermes/plugins/langfuse_tracing/plugin.yaml" \
  > "$PLUGIN_DIR/plugin.yaml"
# Restart hermes to pick up changes
```

---

## Alternative: symlink for plugin development

If you're actively editing the plugin and want it version-controlled separately:

```bash
# Create a standalone plugin repo
mkdir -p ~/Projects/hermes-langfuse-plugin/langfuse_tracing
git show "$PLUGIN_REF:.hermes/plugins/langfuse_tracing/__init__.py" \
  > ~/Projects/hermes-langfuse-plugin/langfuse_tracing/__init__.py
git show "$PLUGIN_REF:.hermes/plugins/langfuse_tracing/plugin.yaml" \
  > ~/Projects/hermes-langfuse-plugin/langfuse_tracing/plugin.yaml
cd ~/Projects/hermes-langfuse-plugin && git init && git add -A && git commit -m "init"

# Symlink into hermes plugin dir (remove existing dir/link first)
rm -rf "$PLUGIN_DIR"
ln -s ~/Projects/hermes-langfuse-plugin/langfuse_tracing "$PLUGIN_DIR"
```

Edits to `~/Projects/hermes-langfuse-plugin/langfuse_tracing/` take effect on
next hermes restart.  Upstream hermes updates are still conflict-free.

---

## Environment variables reference

All variables are optional.  Tracing does nothing unless `ENABLED` + both keys are set.

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `HERMES_LANGFUSE_ENABLED` | yes | `false` | Must be `true`/`1`/`yes`/`on` |
| `HERMES_LANGFUSE_PUBLIC_KEY` | yes | — | Langfuse project public key |
| `HERMES_LANGFUSE_SECRET_KEY` | yes | — | Langfuse project secret key |
| `HERMES_LANGFUSE_BASE_URL` | no | `https://cloud.langfuse.com` | Self-hosted Langfuse URL |
| `HERMES_LANGFUSE_ENV` | no | — | Environment tag (e.g. `development`) |
| `HERMES_LANGFUSE_RELEASE` | no | — | Release tag |
| `HERMES_LANGFUSE_SAMPLE_RATE` | no | `1.0` | Float 0.0-1.0 |
| `HERMES_LANGFUSE_MAX_CHARS` | no | `12000` | Max chars per traced value |
| `HERMES_LANGFUSE_DEBUG` | no | `false` | Verbose logging to stderr |

Each variable also accepts `CC_LANGFUSE_*` and bare `LANGFUSE_*` prefixes as
fallbacks (checked in order: `HERMES_` > `CC_` > bare).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `hermes plugins list` doesn't show `langfuse_tracing` | Plugin files not in the right dir | Check `$PLUGIN_DIR` matches your profile.  Must contain both `__init__.py` and `plugin.yaml`. |
| Listed as `disabled` | In `plugins.disabled` in config.yaml | Run `hermes plugins enable langfuse_tracing` |
| No trace output with `HERMES_LANGFUSE_DEBUG=true` | Plugin loaded but dormant | Verify all 3 required env vars are set and exported |
| `"Could not initialize Langfuse client: ..."` | Bad credentials or unreachable server | Check public/secret keys; check base URL if self-hosted |
| Traces appear but background reviews aren't tagged | `feat/turn-type-hooks` not merged upstream | Plugin still works — `turn_type` defaults to `"user"`.  Background reviews just won't be filterable until the upstream PR lands. |
| Plugin works in `hermes` but not `hermes -p coder` | Profile-scoped plugin dirs | Install plugin into `~/.hermes/profiles/coder/plugins/langfuse_tracing/` |

---

## Disabling tracing

Three options, from least to most permanent:

1. **Unset env vars** — unset `HERMES_LANGFUSE_ENABLED`.  Plugin loads but does nothing.
2. **CLI toggle** — `hermes plugins disable langfuse_tracing`.  Plugin is skipped at startup.
3. **Remove files** — `rm -rf "$PLUGIN_DIR"`.

---

## What gets traced

Each user turn becomes a root trace with nested child observations:

```
Hermes turn  (or "Hermes background review")
 |-- LLM call 0  (generation — with usage/cost)
 |-- Tool: search_files  (tool — with parsed JSON output)
 |-- Tool: read_file  (tool — head/tail preview, not raw content)
 |-- LLM call 1  (generation)
 \-- ...
```

Root trace metadata: `source`, `task_id`, `session_id`, `platform`, `provider`,
`model`, `api_mode`, `turn_type`.

Tags: `hermes`, `langfuse`, plus `background_review` for auto-generated passes.

Data normalization applied:
- Tool result JSON strings parsed into dicts
- Trailing `[Hint: ...]` extracted into `_hint` key
- `read_file` content replaced with head/tail line preview
- `base64_content` omitted (replaced with length)
- Usage/cost extracted when `agent.usage_pricing` is available

---

## Running tests

Tests live on the fork branch only — not on upstream or `main`.

```bash
git checkout feat/langfuse_tracing
source .venv/bin/activate
python -m pytest tests/test_langfuse_tracing_plugin.py -q
```

12 tests covering payload parsing, observation nesting, tool call aggregation,
and `turn_type` propagation.  No credentials or network access needed.

---

## Project history

### Branches

| Branch | Remote | Purpose |
|--------|--------|---------|
| `feat/turn-type-hooks` | `origin` (fork) | Upstream PR: `turn_type` hook plumbing in `run_agent.py` + `model_tools.py` |
| `feat/langfuse_tracing` | `origin` (fork) | Plugin code, tests, optional skill, skills hub changes |

Fork remote: `git@github.com:kshitijk4poor/hermes-agent.git`
Upstream remote: `https://github.com/NousResearch/hermes-agent.git`

### Commit log (chronological)

| Date | Commit | Description |
|------|--------|-------------|
| 2026-03-28 | `b0a64856` | Initial plugin + hook emission patches + langfuse dependency |
| 2026-03-28 | `e691abda` | Parse JSON tool payloads into structured data |
| 2026-03-28 | `00dbff19` | Handle trailing `[Hint: ...]` after JSON in tool outputs |
| 2026-03-28 | `fd54a008` | Fix child observation nesting (use parent span API) |
| 2026-03-28 | `8752aed1` | Format read_file traces as head/tail previews |
| 2026-03-28 | `93f9c338` | Aggregate tool calls onto root trace output |
| 2026-03-29 | `dd714b2a` | Optional skill installer + skills hub enhancements |
| 2026-03-29 | `4b2f865e` | Distinguish background review traces via `turn_type` |
| 2026-03-29 | `aef4b44d` | Upstream-clean `turn_type` hook plumbing (2 files only) |

### File inventory

**Plugin** (`$HERMES_HOME/plugins/langfuse_tracing/`):
`__init__.py` (hook handlers + `register()`), `plugin.yaml` (manifest)

**Upstream PR** (`feat/turn-type-hooks`):
`run_agent.py` (+`_turn_type` attr, hook propagation), `model_tools.py` (+`turn_type` param)

**Fork branch** (`feat/langfuse_tracing`):
`.hermes/plugins/langfuse_tracing/` (plugin source),
`optional-skills/observability/` (installer skill),
`tools/skills_hub.py` + `hermes_cli/skills_hub.py` (hub enhancements),
`tests/test_langfuse_tracing_plugin.py` + `tests/tools/test_skills_hub.py` (tests)

### Known limitations

1. `pre_llm_call`/`post_llm_call` fire once per user turn. Hermes (this branch) adds `pre_api_request`/`post_api_request` per actual LLM HTTP request; the Langfuse plugin on `feat/langfuse_tracing` should register those names and read the summary kwargs documented above.
2. No session-level parent trace — turns are independent, linked by `session_id` in metadata.
3. Background review filtering requires the `feat/turn-type-hooks` upstream PR.
4. Plugin is profile-scoped — must be installed per Hermes profile.
