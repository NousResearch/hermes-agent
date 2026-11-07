# Hermes Self-Evolution Companion Workflow

This checkout now includes a thin bridge for using the companion repo:

- upstream optimizer: `https://github.com/NousResearch/hermes-agent-self-evolution`
- local bridge: `agent/self_evolution_bridge.py`
- local runner: `scripts/run-self-evolution.sh`

## Why a bridge exists

`hermes-agent-self-evolution` currently expects Hermes history in the legacy
`~/.hermes/sessions/*.json` format.

Current Hermes stores live session history in `state.db`.

The bridge exports selected sessions from `state.db` into the legacy JSON shape
on demand, inside a temporary HOME, so the optimizer can mine real Hermes
history without changing Hermes runtime storage.

## Typical usage

Synthetic dataset:

```bash
scripts/run-self-evolution.sh \
  --skill github-code-review \
  --iterations 3 \
  --eval-source synthetic
```

Mine real Hermes session history first:

```bash
scripts/run-self-evolution.sh \
  --skill github-code-review \
  --iterations 3 \
  --eval-source sessiondb
```

## Behavior

The runner will:

1. Clone `NousResearch/hermes-agent-self-evolution` into
   `${HERMES_SELF_EVOLUTION_REPO:-$HERMES_HOME/hermes-agent-self-evolution}` if missing.
2. Create a dedicated `.venv` for the companion repo via `uv`.
3. Install the companion repo with its `dev` dependencies.
4. For `--eval-source sessiondb`, export recent sessions from `state.db` into a
   temporary `~/.hermes/sessions/` tree.
5. Run `python -m evolution.skills.evolve_skill` against this checkout via
   `HERMES_AGENT_REPO=<this repo>`.

## Useful overrides

- `HERMES_SELF_EVOLUTION_REPO` — custom companion repo path
- `SELF_EVO_SESSION_LIMIT` — cap how many sessions are exported for sessiondb runs

## Model defaults

Upstream defaults target OpenAI models. The local runner keeps those defaults if
`OPENAI_API_KEY` is present.

If `OPENAI_API_KEY` is missing but `ANTHROPIC_API_KEY` is available, the runner
automatically adds:

- `--optimizer-model anthropic/claude-sonnet-4.6`
- `--eval-model anthropic/claude-sonnet-4.6`

You can always override both explicitly on the command line.

## Notes

- This is an offline maintenance workflow, not a runtime dependency.
- It is intentionally scoped to skill evolution first.
- Guardrails and PR review remain in the companion repo's flow.
