# Plan 001: Formalize GHCR runtime plus voice wiring architecture

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Working directory**: Run all commands from the repo root: `c:\Users\1\github-pr\hermes-agent`.
>
> **Shell assumption**: All commands in this plan are written for Windows PowerShell, matching the current operator environment for this repo.
>
> **Drift check (run first)**: `git diff --stat 7b162e2da..HEAD -- docker-compose.yml docker-compose.upstream.yml docker-compose.windows.yml README.md INSTALL.md Dockerfile data/config.yaml`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: tech-debt
- **Planned at**: commit `7b162e2da`, 2026-08-07

## Supported architecture

The supported runtime architecture for voice in this repo is:

1. **Hermes containers run from the published GHCR image**: `ghcr.io/jzkk720/hermes-agent:latest`.
2. **Voice behavior is layered at runtime**, not baked into a repo-local image:
   - active provider and voice selection come from mounted `data/config.yaml`
   - optional Python packages for voice backends are made available at container startup and/or via the durable lazy-install target
3. **Qwen3-TTS is a separate host-side sidecar**, not part of the container image:
   - it runs from the external `agent-meow` environment
   - Hermes reaches it over `http://host.docker.internal:17494`

This plan is to formalize that architecture in compose comments and top-level docs. It is **not** to convert the stack to a local-build lane.

## Why this matters

The repo has drifted into two competing mental models: "voice is part of a local build image" vs. "voice is runtime wiring on top of the GHCR image." The live stack proves the second model is the real one today. If the docs and compose comments continue to leave that ambiguous, future changes to fallback logic, package ownership, or Qwen integration will keep landing in the wrong layer and creating regressions.

## Current state

- Relevant files:
  - `docker-compose.upstream.yml` — repo-specific deployment lane; uses the GHCR image and runtime startup wiring for voice packages.
  - `docker-compose.yml` — default compose lane; same image/provenance model.
  - `docker-compose.windows.yml` — Windows deployment lane; same image/provenance model.
  - `README.md` — top-level user-facing runtime/install overview.
  - `INSTALL.md` — top-level Docker install and update guide.
  - `data/config.yaml` — mounted runtime config holding the active provider and default voice.
- Current excerpts that establish the lane:
  - `docker-compose.upstream.yml:12` — `image: ${HERMES_FORK_IMAGE:-ghcr.io/jzkk720/hermes-agent:latest}`
  - `docker-compose.yml:33` — `image: ${HERMES_FORK_IMAGE:-ghcr.io/jzkk720/hermes-agent:latest}`
  - `docker-compose.windows.yml:16` — `image: ${HERMES_FORK_IMAGE:-ghcr.io/jzkk720/hermes-agent:latest}`
  - `docker-compose.upstream.yml:49` and `:74`, `docker-compose.yml:77` and `:101`, `docker-compose.windows.yml:26` and `:43` — `command:` entries that begin with `"sh", "-lc", "(/opt/hermes/.venv/bin/python3 -m pip show edge-tts ...` and reinforce voice package availability at runtime.
  - `README.md:131` contains `You can still bring your own keys per-tool whenever you want — the gateway is per-backend, not all-or-nothing.`
  - `README.md:135` begins `## CLI vs Messaging Quick Reference`
  - `INSTALL.md:47` contains `This keeps the local \`data/.env\`, \`data/config.yaml\`, sessions, memories, and PostgreSQL data while pulling the fork's GHCR-published image.`
  - Live runtime audit showed both `hermes-gateway` and `hermes-web` running `ghcr.io/jzkk720/hermes-agent:latest`, and `/opt/data/config.yaml` inside the gateway reported `tts.provider = edge` and `tts.edge.voice = zh-CN-XiaoxiaoNeural`.
- What is missing today:
  - The compose comments establish the GHCR image lane, but do **not** explain that voice provider choice comes from mounted runtime config.
  - The top-level docs do **not** contain a concise repo-local explanation of the split between image provenance, mounted voice config, startup package wiring, and the host-side Qwen sidecar.

## Commands you will need

| Purpose                    | Command                                                                                                                                                                                                        | Expected on success                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Compose parse (upstream)   | `docker compose -f docker-compose.upstream.yml config > $null`                                                                                                                                                 | exit 0                                                            |
| Compose parse (default)    | `docker compose -f docker-compose.yml config > $null`                                                                                                                                                          | exit 0                                                            |
| Compose parse (windows)    | `docker compose -f docker-compose.windows.yml config > $null`                                                                                                                                                  | exit 0                                                            |
| Runtime image check        | `docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > $env:TEMP\hermes-ps.txt; findstr /i "hermes" $env:TEMP\hermes-ps.txt`                                                                        | shows `ghcr.io/jzkk720/hermes-agent:latest` for Hermes containers |
| Mounted voice-config check | `docker exec hermes-gateway python3 -c "import yaml; cfg=yaml.safe_load(open('/opt/data/config.yaml', encoding='utf-8')); print(cfg['tts']['provider']); print(cfg['tts']['edge']['voice'])"`                  | prints `edge` and `zh-CN-XiaoxiaoNeural`                          |
| README anchor lookup       | `findstr /n /c:"You can still bring your own keys per-tool whenever you want — the gateway is per-backend, not all-or-nothing." README.md` and `findstr /n /c:"## CLI vs Messaging Quick Reference" README.md` | both anchor lines found                                           |
| INSTALL anchor lookup      | `findstr /n /c:"PostgreSQL data while pulling the fork's GHCR-published image." INSTALL.md`                                                                                                                    | anchor line found                                                 |
| README wording check       | `findstr /i /c:"### Voice Runtime Model" README.md`                                                                                                                                                            | exact header found once                                           |
| INSTALL wording check      | `findstr /i /c:"### Voice runtime note" INSTALL.md`                                                                                                                                                            | exact header found once                                           |
| Compose wording check      | `findstr /i /c:"Voice package availability is currently reinforced at container startup." docker-compose.upstream.yml docker-compose.yml docker-compose.windows.yml`                                           | phrase found in all three files                                   |

## Scope

**In scope**:

- `docker-compose.upstream.yml`
- `docker-compose.yml`
- `docker-compose.windows.yml`
- `README.md`
- `INSTALL.md`

**Out of scope**:

- `tools/tts_tool.py` fallback logic
- `scripts/qwen3-tts-server.py`
- `data/config.yaml` provider values
- Converting the stack to a local-build / agent-meow image lane
- Reworking package ownership into the image build

## Git workflow

- Branch: `advisor/001-voice-runtime-source`
- Commit style: conventional/imperative, one logical unit per commit
- Do not push or open a PR unless explicitly instructed

## Steps

### Step 1: Add explicit GHCR-lane wording to the compose files

In each of the three compose files, add a short comment block near the top that makes these four points explicit:

- Hermes runtime comes from the published GHCR image
- voice behavior comes from mounted `data/config.yaml`
- `edge-tts` and `piper-tts` are runtime-wired packages, not evidence of a local-build voice image
- Qwen3-TTS remains a separate host-side sidecar on `host.docker.internal:17494`

Use this exact wording block:

```yaml
# Supported voice architecture for this lane:
# - Hermes runtime comes from the published GHCR image in this compose file.
# - Active voice provider and default voice come from mounted data/config.yaml.
# - edge-tts and piper-tts are runtime-wired packages, not part of a repo-local build lane.
# - Qwen3-TTS remains a separate host-side sidecar reached via host.docker.internal:17494.
```

Place it in the opening comment block of each file, immediately before the first YAML key that starts the compose body.

**Verify**:

- `docker compose -f docker-compose.upstream.yml config > $null` → exit 0
- `findstr /i /c:"Supported voice architecture for this lane:" docker-compose.upstream.yml docker-compose.yml docker-compose.windows.yml` → header found in all three files

### Step 2: Add one explicit runtime note to README at a fixed location

Locate the anchors first:

```powershell
findstr /n /c:"You can still bring your own keys per-tool whenever you want — the gateway is per-backend, not all-or-nothing." README.md
findstr /n /c:"## CLI vs Messaging Quick Reference" README.md
```

Insert a new subsection in `README.md` immediately **after** the first anchor and immediately **before** the second anchor.

Use this exact wording:

```markdown
### Voice Runtime Model

The supported Docker runtime uses `ghcr.io/jzkk720/hermes-agent:latest` for the Hermes containers.
Voice provider choice and the default voice come from mounted `data/config.yaml`, while optional voice packages such as `edge-tts` and `piper-tts` are wired in at runtime.
`Qwen3-TTS` is not baked into the container image; it runs as a separate host-side sidecar reached at `host.docker.internal:17494`.
```

**Verify**:

- `findstr /i /c:"### Voice Runtime Model" README.md` → exact header found once
- `findstr /i /c:"Voice provider choice and the default voice come from mounted" README.md` → exact sentence found
- `findstr /i /c:"Qwen3-TTS is not baked into the container image; it runs as a separate host-side sidecar reached at host.docker.internal:17494." README.md` → exact sentence found

### Step 3: Add one explicit runtime note to INSTALL at a fixed location

Locate the anchor first:

```powershell
findstr /n /c:"PostgreSQL data while pulling the fork's GHCR-published image." INSTALL.md
```

Insert a new subsection in `INSTALL.md` immediately **after** that sentence.

Use this exact wording:

```markdown
### Voice runtime note

The supported Docker lane still runs the published GHCR Hermes image.
Voice provider choice, default voice selection, and optional voice backends are layered at runtime through `data/config.yaml` and the startup/lazy-install path.
`Qwen3-TTS` remains a separate host-side sidecar, not a service baked into the container image.
```

**Verify**:

- `findstr /i /c:"### Voice runtime note" INSTALL.md` → exact header found once
- `findstr /i /c:"Voice provider choice, default voice selection, and optional voice backends are layered at runtime through" INSTALL.md` → exact sentence found
- `findstr /i /c:"Qwen3-TTS remains a separate host-side sidecar, not a service baked into the container image." INSTALL.md` → exact sentence found

### Step 4: Ensure the runtime precondition for live verification

Steps 5 and 6 assume the repo-specific Hermes containers are already running.

If they are not running, start them first with:

```powershell
docker compose -f docker-compose.upstream.yml up -d --pull=missing
```

This is only a verification precondition, not a change to the supported architecture.

**Verify**: `docker ps --format "table {{.Names}}\t{{.Status}}" | findstr /i "hermes"` → shows `hermes-gateway` and `hermes-web` running

### Step 5: Add an explicit startup-wrapper comment at the actual wrapper locations

Add the runtime/package split comment immediately **above** each `command:` entry that currently begins with `"sh", "-lc", "(/opt/hermes/.venv/bin/python3 -m pip show edge-tts ...`.

The exact locations are:

- `docker-compose.upstream.yml` in the `hermes-web:` service block
- `docker-compose.upstream.yml` in the `hermes-gateway:` service block
- `docker-compose.yml` in the `hermes-web:` service block
- `docker-compose.yml` in the `hermes-gateway:` service block
- `docker-compose.windows.yml` in the `gateway:` service block
- `docker-compose.windows.yml` in the `dashboard:` service block

Use this exact wording:

```text
Voice package availability is currently reinforced at container startup. This does not change the supported runtime lane: the containers still run from the published GHCR image.
```

**Verify**:

- `findstr /i /c:"Voice package availability is currently reinforced at container startup." docker-compose.upstream.yml docker-compose.yml docker-compose.windows.yml` → phrase found in all three files
- `docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | findstr /i "hermes"` → image names still show `ghcr.io/jzkk720/hermes-agent:latest`

### Step 6: Verify docs match the live mounted voice config

After updating docs/comments, confirm that the described runtime matches the actual mounted config and active default voice. The docs must attribute the default voice to mounted config, not to the image itself.

**Verify**:

- `docker exec hermes-gateway python3 -c "import yaml; cfg=yaml.safe_load(open('/opt/data/config.yaml', encoding='utf-8')); print(cfg['tts']['provider']); print(cfg['tts']['edge']['voice'])"` → prints `edge` and `zh-CN-XiaoxiaoNeural`
- `findstr /i /c:"Voice provider choice and the default voice come from mounted" README.md` → exact sentence found
- `findstr /i /c:"Voice provider choice, default voice selection, and optional voice backends are layered at runtime through" INSTALL.md` → exact sentence found

## Test plan

- No new unit tests expected for this plan.
- Validation is compose parsing + grep-able text checks + a runtime image/config check.
- If any Python file is touched as part of bootstrap ownership, syntax-check it with `python -m py_compile`.

## Done criteria

- [ ] The supported runtime lane is explicitly documented as GHCR image + runtime voice wiring
- [ ] Each compose file contains the exact `Supported voice architecture for this lane:` comment block
- [ ] `README.md` contains the exact `Voice Runtime Model` note at the specified insertion point
- [ ] `INSTALL.md` contains the exact `Voice runtime note` at the specified insertion point
- [ ] Each wrapped compose `command:` entry has the explicit startup-wrapper comment above it
- [ ] `docker compose ... config` succeeds for all in-scope compose files
- [ ] A live `docker ps` image check matches the documented GHCR lane
- [ ] The docs/comments correctly distinguish image provenance from mounted runtime voice settings
- [ ] No out-of-scope files modified

## STOP conditions

- The intended runtime is actually controlled from a different repo/file not present in this workspace
- The user reverses the architectural decision and wants a true local-build lane instead
- The live deployment environment diverges from the compose files in a way the repo cannot express (for example, an external orchestrator rewriting image/command settings)

## Maintenance notes

- Every future voice/runtime change should target the GHCR-image-plus-runtime-wiring lane unless the architecture decision is explicitly revisited.
- Reviewer should scrutinize whether comments, image/build settings, mounted config expectations, and actual runtime all tell the same story.
