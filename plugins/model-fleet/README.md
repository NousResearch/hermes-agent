# model-fleet

Repoint a whole Hermes install at one provider/model in a single command.

A Hermes install accumulates several **independent** model pins:

| Where | Key |
|---|---|
| active profile | `model.default` / `model.provider` |
| sub-agents | `delegation.model` / `delegation.provider` |
| every named profile | both of the above, in `profiles/<name>/config.yaml` |
| every cron job | `provider` / `model` (+ `*_snapshot` twins) |
| auxiliary tasks | `auxiliary.<task>.provider` / `.model` |

Keeping those aligned by hand means editing a dozen files, then repeating the whole
edit next time you change model. This plugin does it in one step and prints exactly
what it touched.

## Install

```bash
hermes plugins install model-fleet     # or copy this directory to ~/.hermes/plugins/model-fleet
hermes plugins enable model-fleet
/restart                               # slash commands register at gateway start
```

## Usage

```
/model-fleet                            # numbered list of authenticated providers
/model-fleet 3                          # numbered model list for provider #3
/model-fleet 3 5                        # apply provider #3 / model #5 across the install
/model-fleet nous z-ai/glm-5.3-flash    # same, addressed by slug
/model-fleet status                     # current model map (profiles, aux, crons)
/model-fleet auxiliary nous <model>     # also repoint auxiliary.* task models
```

Add `--dry-run` to any apply form to preview the full plan without writing.

## What one apply changes

1. **Default model** — written through Hermes' own `switch_model` +
   `persist_model_selection` pipeline, so `base_url`, `api_key`, `api_mode` and the
   context pin are *re-resolved* for the target route instead of being carried over
   from the old provider. This is the reason the plugin does not hand-write
   `model.provider`.
2. **Sub-agents** — `delegation.*`; stale `delegation.base_url` / `api_key` /
   `api_mode` are cleared when the provider changes.
3. **Every in-scope profile** — both blocks above.
4. **Every in-scope cron job** that is not `no_agent` — pure-script jobs are skipped
   because no model is in that loop. Writes go through `cron.jobs.save_jobs` under
   `use_cron_store(home)` so the cross-process jobs lock and the shrink-merge guard
   still apply. Never hand-edit `jobs.json`.
5. **Auxiliary tasks** — only via the `auxiliary` subcommand or
   `include_auxiliary: true`. Off by default: those models are usually pinned to a
   cheap tier on purpose.

## Safety

- `--dry-run` first is the intended habit; the plan names every file.
- Every touched file is copied to `<name>.bak-model-fleet-<UTC stamp>` before the
  write (disable with `backup: false`).
- A refused switch (bad model, missing credentials) is reported and **nothing** is
  written — the check runs before the first mutation.
- A new model reaches new sessions and the next cron fire. The chat that ran the
  command keeps its current model until a new session starts.
- `no_agent` cron jobs are never modified.

## Configuration

All in `config.yaml` under `plugins.entries.model-fleet`:

```yaml
plugins:
  entries:
    model-fleet:
      include_profiles: true      # also repoint profiles/<name>/config.yaml
      include_cron: true          # also repoint cron jobs
      include_auxiliary: false    # also repoint auxiliary.* models
      profile_allowlist: []       # when non-empty, ONLY these profiles
      profile_blocklist: []       # profiles to skip
      model_allowlist: []         # when non-empty, only these providers are offered
      backup: true                # write .bak-model-fleet-<stamp> copies
```

An install with no profiles and no cron jobs simply gets a smaller blast radius —
no code change needed.

## Notes for contributors

- `tests/` runs standalone: `PYTHONPATH=<hermes-repo> pytest plugins/model-fleet/tests`.
  `conftest.py` puts the repo root on `sys.path`; the plugin imports `cron.jobs`,
  `hermes_cli.*` and `utils` at call time.
- Provider listing passes `non_blocking_catalogs=True` deliberately: with it false a
  cold provider catalog blocks the slash handler for 15-30s on a live `/models` fetch.
- The command name must not collide with a built-in (`/model`, `/cron`, …);
  `register_command` drops collisions with only a log warning. Check
  `hermes_cli.commands.resolve_command(name)` before renaming.
