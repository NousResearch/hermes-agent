# Doubleword.ai Hermes Agent Skill

The `doubleword` skill helps Hermes and OpenClaw agents run cost-aware LLM
inference on Doubleword.ai. It gives agents a practical operating procedure for
choosing between realtime, async, and 24-hour batch inference, validating local
JSONL payloads, submitting jobs through the `dw` CLI, and retrieving results.

Doubleword.ai provides high-performance inference for generation, extraction,
classification, OCR, embeddings, evals, and large data-processing workflows.

## Get a Doubleword.ai Account

You need a Doubleword.ai account and API key to use this skill.

If you do not already have an account, sign up at
[doubleword.ai](https://doubleword.ai/). Signup is free and includes free tokens
so you can try the platform before adding paid usage.

After creating an account, create or copy an API key. When this skill is loaded
in an interactive Hermes session, Hermes requests `DOUBLEWORD_API_KEY` through
its masked secret prompt and stores it in the active profile's `.env` file.

For manual or non-interactive setup, add the key to
`${HERMES_HOME:-~/.hermes}/.env`:

```bash
DOUBLEWORD_API_KEY=your_api_key_here
```

Do not put the key in `config.yaml`, pass its literal value on a command line,
or commit it to a repository.

For headless use, configure the CLI by referencing the environment variable:

```bash
dw login --api-key "$DOUBLEWORD_API_KEY"
```

The CLI validates the key and stores it in `~/.dw/credentials.toml` with
restricted permissions. The command contains the variable name rather than the
secret value.

The skill supports two authentication readiness paths. With browser login, run
the interactive login and then verify identity and organization:

```bash
dw login
dw whoami
```

For headless/API-key login, `dw whoami` and `dw models list` are unavailable
because API-key credentials do not include admin API access. In that mode, the
skill validates JSONL payloads locally and uses a non-interactive,
token-limited realtime request, such as
`MODEL="${MODEL:-openai/gpt-oss-20b}"; dw realtime "$MODEL" "Reply with OK." --temperature 0 --max-tokens 2 --no-stream`,
as the inference-authentication probe before uploading or submitting jobs.

## What This Skill Does

- Routes small interactive prompts to realtime inference.
- Sends same-session background workflows to async inference.
- Uses 24-hour batch inference for large jobs when cost savings matter more than
  immediate completion.
- Validates JSONL payloads locally before upload.
- Encourages deterministic request IDs so batch results can be joined back to
  source data.
- Avoids long idle polling loops that are unsafe for Hermes agents.
- Points agents to detailed command, model, and pricing references only when
  needed.

## Repository Layout

```text
SKILL.md
references/
  cli-recipes.md
  models-and-pricing.md
```

In the Hermes repository, this skill lives at
`optional-skills/mlops/doubleword-ai/`. Installing it through the official
skills source copies the bundle into the active Hermes profile.

The skill uses progressive disclosure:

- **Level 0:** skill list metadata from `SKILL.md`
- **Level 1:** operating procedure in `SKILL.md`
- **Level 2:** detailed command and model references under `references/`

References are intentionally separate because command recipes, model tables, and
pricing notes are detail-heavy and may change more frequently than the core
skill procedure.

## Key Files

- [`SKILL.md`](./SKILL.md) - Main skill definition and agent procedure.
- [`references/cli-recipes.md`](./references/cli-recipes.md) - Doubleword CLI
  recipes, official command reference link, and Hermes-specific execution
  guardrails.
- [`references/models-and-pricing.md`](./references/models-and-pricing.md) -
  Model selection guidance, pricing tables, and batch limits.

## Requirements

- A Doubleword.ai account.
- `DOUBLEWORD_API_KEY` configured through Hermes' secure prompt or the active
  profile's `.env` file.
- The Doubleword `dw` CLI available wherever the agent runs Doubleword jobs.
- Terminal access for Hermes/OpenClaw skill execution.

## Documentation

- Doubleword.ai: [https://doubleword.ai](https://doubleword.ai/)
- Doubleword API base URL: `https://api.doubleword.ai/v1`
- `dw` CLI command reference:
  [https://doublewordai.github.io/dw/commands.html](https://doublewordai.github.io/dw/commands.html)

## License

See [`LICENSE`](./LICENSE).
