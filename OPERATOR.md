# Hermes Agent — Local Operator Guide

This checkout is self-contained under `/home/len/hermes-agent`. Development
dependencies, caches, build outputs, and test state stay inside the checkout.

## Start here

```bash
cd /home/len/hermes-agent
source .venv/bin/activate
hermes doctor
hermes
```

`hermes setup` configures a production profile. It writes to the selected
`HERMES_HOME`; review that path before running setup if strict filesystem
containment matters.

For a disposable repository-local profile:

```bash
export HERMES_HOME=/home/len/hermes-agent/.tmp/hermes-home
hermes doctor
hermes
```

## Local model status

Ollama is available at `http://127.0.0.1:11434/v1`. The installed models have
these Hermes compatibility limits:

- `qwen3:8b` and `hermes-qwen:8b`: 40,960-token contexts, below the required
  64K minimum.
- `gemma3:12b`: 131K context, but its Ollama template does not support tools.
- `llama3.1:8b`: 131K context and accepts tools, but did not reliably follow
  the schema-heavy prompt in smoke testing.

Use a model with a declared context of at least 64K and reliable structured
tool calling. Configure it through `hermes model` or `hermes setup` instead
of editing secrets into tracked files.

## Verification

```bash
scripts/run_tests.sh -j 8 -q
npm run check
npm run build --workspace web
npm run build --workspace ui-tui
npm run build --workspace apps/desktop
npm run build --workspace apps/bootstrap-installer
npm audit --audit-level=high
hermes security audit --fail-on high
docker compose config -q
hermes-acp --check
```

Hermes intentionally does not build wheels or source distributions. Supported
distribution paths are the shell installer, Docker, and Nix; development uses
an editable install.

## Services on this machine

- Open WebUI: `http://127.0.0.1:3000`
- Firecrawl API: `http://127.0.0.1:3002`
- Ollama API: `http://127.0.0.1:11434`

Those services currently listen beyond loopback at the host level. Treat them
as LAN-accessible unless host/router firewall policy says otherwise.

## Repository hygiene

- Keep secrets in the selected Hermes home, never in tracked files.
- Keep `.venv`, `.cache`, `.tmp`, and `.tools` untracked.
- Update `TODO.md` with evidence whenever a completion gate changes.
- Commit one coherent behavior or verification slice at a time.
