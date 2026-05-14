# Hermes Agent ACP Registry + Zed Integration Implementation Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task.

Goal: Make Hermes Agent installable from Zed's official ACP Registry, so users can add Hermes from Zed's agent panel without manual custom `agent_servers` settings.

Architecture: Use the official `agentclientprotocol/registry` flow instead of the deprecated Zed Agent Server Extension path. Ship a registry-compatible launcher distribution, advertise valid ACP auth methods during every handshake, validate against official registry schema and auth CI, then submit a registry PR for `hermes-agent`.

Tech Stack: Hermes Agent Python package, ACP adapter (`hermes acp` / `hermes-acp`), npm launcher package, official ACP Registry JSON schema, Zed external agent UI.

---

## Compliance constraints

- Zed v0.221.x+ prefers the ACP Registry for external agents; do not use Zed Agent Server Extensions for distribution.
- Registry repo layout is top-level `hermes-agent/agent.json` and `hermes-agent/icon.svg`, not `agents/hermes-agent/`.
- Registry metadata must use the official schema: `id`, `name`, `version`, `description`, `distribution`, optional `repository`, `website`, `authors`, `license`.
- Distribution must be exactly one supported type unless intentionally adding another: `binary`, `npx`, or `uvx`.
- Hermes must advertise at least one valid `authMethods` entry on a clean first-run handshake. No-provider/no-auth is not compliant.
- Terminal Auth must be explicit and deterministic: `id: hermes-setup`, `type: terminal`, `args: ["--setup"]`.
- `icon.svg` must be 16x16, square, monochrome, and use only `currentColor` / `none` for fill/stroke; no gradients, hardcoded colors, or `url(#...)` paints.
- ACP server mode must reserve stdout for JSON-RPC only. Diagnostics/logs go to stderr. `--version`, `--check`, and `--setup` are not server mode and may print normally.
- Published npm package must exist and be runnable before the upstream registry PR references it.

---

## Tasks

1. Verify/implement ACP auth methods.
   - Always return terminal setup auth from `initialize()`.
   - Return configured provider auth in addition when provider credentials are resolvable.
   - Add tests for provider auth, terminal fallback auth, and authenticate behavior before/after provider setup.

2. Add non-interactive ACP commands.
   - `hermes acp --version`
   - `hermes acp --check`
   - `hermes acp --setup`
   - Same behavior through `hermes-acp`.

3. Build npm launcher package.
   - Package: `@nousresearch/hermes-agent-acp@<version>`.
   - Command: `uvx --from 'hermes-agent[acp]==<version>' hermes-acp ...args`.
   - Fallback: `uv tool run --from ...` when only `uv` exists.
   - Forward all args, including `--setup`, `--version`, and `--check`.
   - Preserve stdio in server mode.
   - Print actionable stderr error when `uv`/`uvx` is missing.

4. Replace local registry metadata.
   - Convert `acp_registry/agent.json` from old command-style local format to official registry schema.
   - Replace `acp_registry/icon.svg` with compliant 16x16 currentColor icon.
   - Add tests rejecting old fields (`schema_version`, `display_name`, `distribution.type`, `distribution.command`) and unknown distribution keys.

5. Update docs.
   - Zed docs show official ACP Registry install first: Add Agent / `zed: acp registry` -> search Hermes Agent -> install.
   - Manual `agent_servers` JSON remains only as local-development fallback.
   - Docs include `uv` prerequisite and `hermes acp --check` troubleshooting.
   - Developer internals mention npm launcher and terminal setup auth.

6. Validate locally.
   - `python -m pytest tests/acp/test_auth.py tests/acp/test_server.py tests/acp/test_entry.py tests/acp/test_registry_manifest.py -q`
   - `(cd packages/hermes-agent-acp && npm test)`
   - `(cd packages/hermes-agent-acp && npm pack --dry-run)`
   - `hermes acp --version`
   - `hermes acp --check`

7. Validate against official registry tooling before PR.
   - In a clone/fork of `agentclientprotocol/registry`, copy files into top-level `hermes-agent/`.
   - Run official dry-run build, e.g. `uv run --with jsonschema .github/workflows/build_registry.py --dry-run`.
   - Run official auth check if available, e.g. `.github/workflows/scripts/run-registry-docker.sh python3 .github/workflows/verify_agents.py --auth-check`.
   - Fix any schema/auth issues before submitting.

8. Publish and submit.
   - Publish `@nousresearch/hermes-agent-acp@<version>`.
   - Verify published package:
     - `npx @nousresearch/hermes-agent-acp@<version> --version`
     - `npx @nousresearch/hermes-agent-acp@<version> --check`
     - ACP initialize/authMethods smoke test through the published package.
   - Open PR to `agentclientprotocol/registry` adding `hermes-agent/agent.json` and `hermes-agent/icon.svg`.

9. End-to-end Zed verification.
   - Install Hermes Agent through Zed's ACP Registry.
   - Start a Hermes thread.
   - Verify workspace cwd, file tools, terminal tools, tool rendering, and approval prompts.

---

## Acceptance criteria

- Hermes appears in Zed's official ACP Registry UI.
- Install starts Hermes without custom Zed settings.
- Registry CI passes schema and auth validation.
- ACP stdout remains JSON-RPC only; all logs go to stderr.
- `authMethods` are present and valid on clean first run.
- Terminal Auth can launch Hermes provider/model setup with `--setup`.
- Zed workspace cwd is honored by Hermes file and terminal tools.
- Docs describe registry install first and manual custom config second.
- Package/release automation prevents registry entries from pointing at unpublished versions.
