# Galt MCP Integration Final Spec

> **Final reviewed spec.** Draft researched externally/internally, passed to Claude Code CLI for oppositional review, then independently adjudicated.
> **Scope:** Galt local stack on macOS: Hermes Agent, Codex CLI, Claude Code CLI, subagents, Docker/Colima.
> **Decision:** Use MCP aggressively where it improves tool quality, but not as global tool bloat.

---

## 1. Executive Decision

Build the MCP layer in three tiers:

1. **Docker MCP Gateway profiles for local/containerized MCP servers** where isolation, lifecycle management, secrets handling, and profile selection are valuable.
2. **Direct remote HTTP MCP connections for SaaS/vendor-hosted servers** where OAuth/callbacks are first-class and Docker Gateway adds indirection.
3. **Native Hermes/Codex/Claude built-in tools remain primary** where MCP would duplicate existing capabilities.

Do **not** install a global kitchen-sink MCP profile. Profiles are capability bundles, not decorations.

Current local state verified on 2026-05-14:

| Component | Verified state |
|---|---|
| Hermes MCP | `hermes mcp list` -> no servers configured |
| Claude Code MCP | `claude mcp list` -> no servers configured |
| Codex MCP | bundled `computer-use` only |
| Docker | Docker CLI 29.4.3, Compose 5.1.3 |
| Docker context | `colima` active; endpoint `unix:///Users/johngalt/.colima/default/docker.sock` |
| Docker MCP plugin | `docker mcp` not installed/available in current CLI |
| Node/npx | available at `/opt/homebrew/bin` |
| uvx | available at `/opt/homebrew/bin` |

The first implementation step is therefore **enable/install Docker MCP Gateway**, not profile creation.

---

## 2. Final Review Adjudication

Claude Code returned a useful but imperfect opposition review.

Accepted from Claude:

- Stdio Docker Gateway is **per-client process**, not one shared daemon. The final architecture must show that honestly.
- Remote SaaS MCP servers such as Vercel/Sentry/Expo should generally connect **directly** to Claude/Codex rather than be forced through Docker Gateway.
- Expo MCP requires an **EAS paid plan**. Treat as a hard gate.
- Context7 remote endpoint requires `CONTEXT7_API_KEY`; local/npm usage may work differently, but remote config must name the credential.
- GitHub MCP base profile needs server-level read-only/toolset restriction, not just a polite instruction.
- DB Toolbox requires an explicit `tools.yaml`; it is not plug-and-play.
- Playwright MCP is powerful enough to be dangerous; restrict it to coding-agent project profiles, not Hermes global.
- Rollback must include OAuth revocation, not only deleting local MCP config.

Rejected from Claude after verification against Docker source docs:

- Claude claimed `docker mcp gateway run --dry-run` does not exist. Docker gateway source docs list `--dry-run` as a valid flag.
- Claude claimed `docker mcp feature list` is undocumented. Docker profile source docs document it as the verification command after `docker mcp feature enable profiles`.
- Claude claimed `docker mcp profile create --server ...` is likely invalid. Docker profile source docs explicitly show `profile create --name ... --server ...` examples.

Takeaway: use Claude's architectural/security critique, but do not blindly copy its command corrections.

---

## 3. Architecture

### 3.1 Actual default model: stdio per client

```text
Hermes process ───────────────┐
                              ├─ starts its own `docker mcp gateway run --profile X` over stdio
Claude Code process ──────────┤
                              ├─ starts its own `docker mcp gateway run --profile Y` over stdio
Codex process ────────────────┘

Each gateway process uses Docker to launch/manage MCP server containers as needed.
```

This is **not** one shared daemon. It is a shared configuration/control model, launched per MCP client. That is acceptable and safer than exposing a long-running HTTP gateway.

### 3.2 Optional future model: local HTTP/streaming gateway

Only if there is a concrete need for one shared gateway endpoint:

```bash
docker mcp gateway run --transport streaming --port <port> --profile <profile>
```

Rules:

- Bind localhost only if possible.
- Never expose publicly without explicit approval.
- Use only narrow read-only/observe profiles unless deliberately activated.
- Re-review auth, logs, and reachable tools before enabling.

Default remains stdio.

---

## 4. Server Placement Matrix

| Server | First use? | Placement | Why |
|---|---:|---|---|
| Context7 | Yes | Docker Gateway/local stdio **or** direct remote HTTP | Current docs for frameworks/libraries |
| GitHub official MCP | Yes | Docker Gateway local/containerized, with read-only/toolset restrictions | PRs/issues/actions/repo context |
| Playwright MCP | Yes for web projects | Project-scoped Claude/Codex; preferably via Docker/local profile | Browser inspection via accessibility snapshots; risky tools |
| MCP Toolbox for Databases | Yes, but gated | Project-local, DB-read-only profile; requires `tools.yaml` | Safer DB tools than random Postgres MCPs |
| Sentry MCP | Project-specific | Direct remote HTTP | Official Sentry production issue/trace context |
| Expo MCP | RN/Expo projects only | Direct remote HTTP; EAS paid plan required | Expo/EAS/RN workflows |
| Vercel MCP | Project-specific | Direct remote HTTP | Deployments/logs/docs/projects with OAuth |
| Cloudflare MCP | Project-specific | Direct remote HTTP | Workers/Pages/R2/D1/observability |
| Linear MCP | Only if Linear is canonical | Direct remote HTTP | Avoid competing with Hermes Kanban unless needed |
| Supabase MCP | Project-specific | Direct remote HTTP or project-local config | Supabase project/table/config/querying |
| Neon MCP | Project-specific | Direct remote HTTP | Neon Postgres branching/querying |
| MongoDB MCP | Project-specific | Project-local | MongoDB/Atlas only |
| Redis MCP | Project-specific | Project-local | Redis/cache/vector/search only |
| Stripe MCP | Rare, project-specific | Direct remote HTTP; never global | Billing/payment mutation risk |
| Figma MCP | UI-heavy projects | Direct remote HTTP/Desktop MCP | Design-to-code context |
| Browserbase MCP | Escalation only | Direct remote HTTP | Cloud browser fallback; local browser first |

Skip globally:

- Filesystem MCP — redundant; Hermes/Codex/Claude already have file tools.
- Memory MCP — avoid split-brain memory; Hermes/Hindsight is memory.
- Sequential Thinking MCP — tool/schema bloat; use plan/review discipline.
- JetBrains npm MCP proxy — deprecated relative to built-in JetBrains MCP.
- Magic/21st.dev — optional scaffolding, not foundational.

---

## 5. Profiles

### 5.1 `galt-base-dev`

Purpose: safe default coding-agent profile.

Servers:

- Context7.
- GitHub official MCP in read-only mode.

Rules:

- No database.
- No deployment providers.
- No Stripe.
- No browser automation.
- No workflow dispatch unless explicitly required.

GitHub restrictions:

- Prefer read-only mode server argument/config.
- Restrict toolsets to context/repository/issues/pull-request read use.
- Avoid `workflow`, `admin:*`, hook management, or write-capable tools in base.

Expose to:

- Claude Code user scope, if not duplicated by project profile.
- Codex user scope, if not duplicated by project profile.
- Hermes only if Benjamin wants GitHub/Context7 from chat; otherwise keep Hermes lean.

### 5.2 `galt-web-dev`

Purpose: web app implementation/debugging.

Servers through Docker Gateway/local:

- Context7.
- GitHub read/write as appropriate for project.
- Playwright MCP with unsafe tool filtering where supported.

Direct remote HTTP, project-local:

- Sentry if project uses Sentry.
- Vercel if project uses Vercel.
- Cloudflare if project uses Cloudflare.

Rules:

- Prefer project-local Claude/Codex config.
- Do not stack `galt-base-dev` and `galt-web-dev` in one session if that duplicates Context7/GitHub tools. Use one profile per session.

### 5.3 `galt-rn-dev`

Purpose: React Native/Expo work.

Servers:

- Context7.
- GitHub.
- Expo MCP direct remote HTTP **only if EAS paid plan is verified**.
- Sentry if app uses Sentry.

Gate:

```bash
npx eas-cli account:view
# or equivalent Expo/EAS account check
```

Skip Expo MCP if no paid EAS plan.

### 5.4 `galt-db-readonly`

Purpose: schema and safe query analysis.

Server:

- MCP Toolbox for Databases.

Hard requirements:

- Write `tools.yaml` before adding profile.
- Use local/dev DB first.
- Use read-only DB credentials.
- Explicit SELECT-only tools.
- Result limits.
- No migrations.
- No production write tools.

Minimal Postgres `tools.yaml` skeleton:

```yaml
sources:
  local_pg:
    kind: postgres
    host: 127.0.0.1
    port: 5432
    database: ${PGDATABASE}
    user: ${PGUSER}
    password: ${PGPASSWORD}

tools:
  list_tables:
    kind: postgres-sql
    source: local_pg
    description: List visible public schema tables.
    statement: |
      SELECT table_schema, table_name
      FROM information_schema.tables
      WHERE table_schema = 'public'
      ORDER BY table_name
      LIMIT 200;

  describe_table:
    kind: postgres-sql
    source: local_pg
    description: Describe columns for a table in public schema.
    parameters:
      - name: table_name
        type: string
        description: Public schema table name.
    statement: |
      SELECT column_name, data_type, is_nullable
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = $1
      ORDER BY ordinal_position
      LIMIT 200;

toolsets:
  readonly:
    - list_tables
    - describe_table
```

Do not use this against production until a separate review defines credentials, network reachability, query limits, and audit logging.

### 5.5 `galt-prod-observe`

Purpose: production observation.

Direct remote/project-specific servers:

- Sentry.
- GitHub Actions read-only.
- Vercel/Cloudflare logs/read-only if supported.

Rules:

- No deploys.
- No DB writes.
- No secret retrieval.
- No billing mutations.

### 5.6 `galt-deploy-admin`

Purpose: rare explicit deployment/change workflow.

Rules:

- Not persistent globally.
- Activated only in a deliberate session.
- Removed/disabled after use.
- Human approval required for prod deploys, DNS changes, billing changes, data mutation, workflow dispatch, or destructive local commands.

Activation pattern:

1. Add session/project-local MCP config.
2. Perform bounded task.
3. Verify result.
4. Remove MCP config.
5. Revoke OAuth/token if temporary.

---

## 6. Implementation Plan

### Phase 0 — No-secret inventory

Run:

```bash
docker --version
docker compose version || true
docker context ls
command -v docker node npx uvx hermes claude codex
hermes mcp list
claude mcp list
codex mcp list
```

Do not print config files containing secrets.

### Phase 1 — Enable Docker MCP Gateway

Current system has Docker CLI but no `docker mcp`. Choose one path.

#### Path A: Docker Desktop MCP Toolkit

Use if Docker Desktop 4.62+ is installed/acceptable. Docker docs/source references have shown 4.59+ and 4.62+ in different places; use the conservative threshold.

1. Enable MCP Toolkit / MCP Working Sets in Docker Desktop.
2. Verify:

```bash
docker mcp --help
docker mcp feature list || true
docker mcp profile list
```

#### Path B: Standalone Docker MCP CLI plugin with Colima

Use if staying Colima-first.

1. Install/build Docker MCP plugin from Docker's `mcp-gateway` source/release.
2. Place executable at:

```text
~/.docker/cli-plugins/docker-mcp
```

3. Verify:

```bash
docker mcp --help
```

4. If profiles are unavailable:

```bash
docker mcp feature enable profiles
docker mcp feature list
```

5. Because Colima uses a nonstandard socket path, include Docker context/socket handling in clients that launch Docker MCP from sanitized environments:

```text
DOCKER_HOST=unix:///Users/johngalt/.colima/default/docker.sock
```

For Hermes MCP subprocesses, explicitly add env if required:

```yaml
env:
  DOCKER_HOST: "unix:///Users/johngalt/.colima/default/docker.sock"
```

Acceptance:

```bash
docker mcp profile list
docker mcp gateway run --dry-run --profile galt-base-dev || true
```

Note: `--dry-run` is documented in Docker gateway source docs, but the local plugin version must still be checked. If it fails on installed version, use profile show/list plus a bounded client smoke test instead.

### Phase 2 — Discover catalog server IDs

Do not invent server IDs.

```bash
docker mcp catalog server ls mcp/docker-mcp-catalog
```

Map actual IDs for:

- Context7, if present.
- GitHub official MCP, if present.
- Playwright MCP, if present.

If a server is not in the Docker catalog, use:

- `docker://<image>` if a maintained image exists.
- `file://<local-server-definition.yaml>` for a local Docker MCP server entry.
- Direct client MCP config if it is a remote HTTP SaaS server.

### Phase 3 — Create Docker profiles

Command shape supported by Docker profile source docs:

```bash
docker mcp profile create --name galt-base-dev \
  --server catalog://mcp/docker-mcp-catalog/<context7-id> \
  --server catalog://mcp/docker-mcp-catalog/<github-id>
```

Equivalent two-step form if local plugin requires it:

```bash
docker mcp profile create --name galt-base-dev
docker mcp profile server add galt-base-dev --server catalog://mcp/docker-mcp-catalog/<context7-id>
docker mcp profile server add galt-base-dev --server catalog://mcp/docker-mcp-catalog/<github-id>
```

Configure non-sensitive settings via Docker profile config. Use Docker secrets/OAuth/local secret stores for sensitive values.

For GitHub base profile, enforce read-only/tool restrictions using the mechanisms supported by the selected server definition:

- GitHub MCP server `--read-only` flag if exposed.
- `GITHUB_TOOLSETS` environment variable if using Docker image/local server.
- Docker profile tool enable/disable controls, e.g. disable write tools.

### Phase 4 — Configure clients

#### 4.1 Hermes

Use YAML for precision. `hermes mcp add --args` does accept multiple args, but YAML is clearer and reviewable.

```yaml
mcp_servers:
  docker_base:
    command: "docker"
    args: ["mcp", "gateway", "run", "--profile", "galt-base-dev"]
    env:
      DOCKER_HOST: "unix:///Users/johngalt/.colima/default/docker.sock"
    timeout: 180
    connect_timeout: 180
```

If Hermes should not have GitHub MCP globally, do not add this yet. Context7-only for Hermes is safer.

Restart Hermes/Gateway after changing MCP config. Hermes discovers MCP tools at startup.

#### 4.2 Claude Code

Stdio Docker profile:

```bash
claude mcp add --transport stdio --scope user MCP_DOCKER_BASE \
  -- docker mcp gateway run --profile galt-base-dev
```

Project-local web profile:

```bash
claude mcp add --transport stdio --scope local MCP_DOCKER_WEB \
  -- docker mcp gateway run --profile galt-web-dev
```

Remote HTTP SaaS examples:

```bash
claude mcp add --transport http --scope local sentry https://mcp.sentry.dev/mcp
claude mcp add --transport http --scope local vercel https://mcp.vercel.com
claude mcp add --transport http --scope local expo https://mcp.expo.dev/mcp
```

Claude Code supports `-s/--scope`; use long form in docs/specs for clarity.

#### 4.3 Codex

Stdio Docker profile:

```bash
codex mcp add MCP_DOCKER_BASE -- docker mcp gateway run --profile galt-base-dev
```

Remote HTTP SaaS examples:

```bash
codex mcp add sentry --url https://mcp.sentry.dev/mcp
codex mcp add vercel --url https://mcp.vercel.com
codex mcp add expo --url https://mcp.expo.dev/mcp
```

Codex config locations:

- Global/user: `~/.codex/config.toml`.
- Project-scoped trusted projects: `.codex/config.toml`.

Remote OAuth may require:

```bash
codex mcp login <server-name>
```

Use `codex mcp --help` and current docs before adding OAuth servers.

### Phase 5 — Verification

For every client/profile:

```bash
hermes mcp list
claude mcp list
codex mcp list
```

Functional smoke tests:

- Context7: fetch current docs for a known library/version.
- GitHub: list open PRs/issues only; no mutation.
- Playwright: inspect a local dev app accessibility snapshot; avoid unsafe JS execution.
- Sentry: list unresolved issues for a known project.
- DB Toolbox: list local/dev tables only.

Operational checks:

```bash
docker ps --filter label=com.docker.mcp || true
docker ps -a --filter label=com.docker.mcp || true
```

Acceptance criteria:

- Client lists server as connected/enabled.
- Intended tool call succeeds.
- No secrets printed in client output/logs.
- Tool set is bounded and not duplicated.
- Containers do not accumulate unexpectedly.
- Remote OAuth grants are visible/revocable in vendor account settings.

---

## 7. Security Controls

### 7.1 Prompt injection

MCP tools that fetch docs, GitHub issues, webpages, logs, or traces return untrusted text. Treat that text as data, not instructions.

### 7.2 Playwright MCP

Official Playwright MCP docs warn it is not a security boundary; unsafe code execution tools are RCE-equivalent in browser context. Therefore:

- No Hermes-global Playwright MCP by default.
- Use only in project-scoped Claude/Codex sessions.
- Prefer tool filtering to disable unsafe tools where supported.
- Do not run against authenticated production sessions unless explicitly needed.

### 7.3 GitHub MCP

Base profile must be read-only.

Minimum posture:

- Fine-grained PAT or OAuth with least privilege.
- Read-only server mode if available.
- Toolset restriction.
- No workflow dispatch in base.
- No repo writes in base.

### 7.4 Databases

- Read-only/dev first.
- Tool-defined SQL only.
- No arbitrary query tool for production.
- Result caps.
- Audit logs.
- Production write profile requires separate approval.

### 7.5 Docker Gateway exposure

- Default stdio only.
- No public ports.
- HTTP/streaming only with explicit approval and localhost binding.

### 7.6 Secrets

- Do not put tokens in Markdown, git-tracked config, or chat.
- Use Docker secrets, OAuth, Keychain, or local env files with permissions.
- Hermes MCP subprocess env is sanitized; explicitly pass only required env keys.
- Rollback includes OAuth/token revocation.

---

## 8. Rollback

Before removal, capture no-secret state:

```bash
hermes mcp list
claude mcp list
codex mcp list
docker mcp profile list || true
```

Remove client configs:

```bash
claude mcp remove MCP_DOCKER_BASE
codex mcp remove MCP_DOCKER_BASE
hermes mcp remove docker-base
```

Remove project-local SaaS servers as applicable:

```bash
claude mcp remove sentry
claude mcp remove vercel
claude mcp remove expo
codex mcp remove sentry
codex mcp remove vercel
codex mcp remove expo
```

Remove Docker profiles:

```bash
docker mcp profile list
docker mcp profile remove galt-base-dev
docker mcp profile remove galt-web-dev
docker mcp profile remove galt-rn-dev
docker mcp profile remove galt-db-readonly
```

Revoke vendor OAuth/app grants manually in each provider account:

- GitHub authorized OAuth apps / PATs.
- Vercel account integrations.
- Sentry account auth/integrations.
- Expo/EAS tokens/integrations.
- Cloudflare OAuth/API tokens.
- Stripe restricted keys/OAuth.
- Figma connected apps.

Do not delete Docker images/volumes globally without separate approval.

---

## 9. Immediate Next Steps

1. Decide Docker path:
   - Docker Desktop MCP Toolkit, or
   - standalone Docker MCP CLI plugin with Colima.
2. Install/enable `docker mcp`.
3. Create only `galt-base-dev` first.
4. Connect to Claude Code and Codex.
5. Smoke-test Context7 and GitHub read-only.
6. Decide whether Hermes gets Context7-only or `galt-base-dev`.
7. Add project profiles only when working in a project that actually needs them.

Recommended first implementation: **Codex + Claude Code get `galt-base-dev`; Hermes waits until we see tool count/context impact.**

---

## 10. Sources

Primary docs/source consulted:

- Docker MCP Catalog and Toolkit: https://docs.docker.com/ai/mcp-catalog-and-toolkit/
- Docker MCP CLI: https://docs.docker.com/ai/mcp-catalog-and-toolkit/cli/
- Docker MCP Gateway: https://docs.docker.com/ai/mcp-catalog-and-toolkit/mcp-gateway/
- Docker MCP Gateway source docs: https://github.com/docker/mcp-gateway
- Codex MCP docs: https://developers.openai.com/codex/mcp
- Claude Code MCP docs: https://code.claude.com/docs/en/mcp
- Hermes MCP docs/source: https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp and local Hermes CLI help/source
- Context7: https://github.com/upstash/context7 and npm `@upstash/context7-mcp`
- GitHub MCP: https://github.com/github/github-mcp-server
- Playwright MCP: https://playwright.dev/docs/getting-started-mcp and https://github.com/microsoft/playwright-mcp
- MCP Toolbox for Databases: https://github.com/googleapis/mcp-toolbox
- Sentry MCP: https://docs.sentry.io/ai/mcp/
- Expo MCP: https://docs.expo.dev/eas/ai/mcp/
- Vercel MCP: https://vercel.com/docs/agent-resources/vercel-mcp
- Cloudflare MCP: https://developers.cloudflare.com/agents/model-context-protocol/mcp-servers-for-cloudflare/
- Linear MCP: https://linear.app/docs/mcp
- Supabase MCP: https://supabase.com/docs/guides/getting-started/mcp
- Figma MCP: https://developers.figma.com/docs/figma-mcp-server/
