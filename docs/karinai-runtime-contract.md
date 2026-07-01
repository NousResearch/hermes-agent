# KarinAI agent runtime contract

Status: Accepted planning direction
Date: 2026-06-22

This document captures the runtime-local contract for `karinai-agent`. The canonical platform architecture remains in `karinai-backend/docs/architecture/per-user-container-runtime.md`; this file describes what the agent image/runtime must expose inside each user workspace container.

## Terminology

- Product-facing runtime name: KarinAI agent.
- Upstream engine/base: Hermes Agent.
- Container: one isolated KarinAI agent container for one active user workspace.
- Backend product run id: durable KarinAI run id stored by `karinai-backend`.
- Agent runtime run id: private execution handle returned by the KarinAI agent API server.

Inside a KarinAI user container, product-facing docs, prompts, logs, and UI events should call the runtime the KarinAI agent. Technical docs may still mention Hermes when referring to upstream internals, compatibility headers, repository history, or engine-specific file paths such as `HERMES_HOME`.

## Request path

The public UI must not call a user container directly.

```text
Forked Open WebUI or future KarinAI UI
  -> karinai-backend public API
  -> backend security/prompt filter and policy checks
  -> backend/runtime-manager
  -> private KarinAI agent API server inside the user container
  -> POST /v1/runs
```

The forked Open WebUI frontend may start with OpenAI-compatible chat semantics. The backend adapts those requests into durable KarinAI runs and streams normalized runtime events back to the UI.

## Canonical runtime API

The canonical backend-to-agent execution API is private `/v1/runs`.

Expected flow:

```text
backend creates karinai_run_id
runtime-manager ensures workspace container is warm
runtime-manager calls POST /v1/runs on the private agent endpoint
agent returns agent_run_id
runtime-manager/backend consume /v1/runs/{agent_run_id}/events
backend persists messages, tool calls, artifacts, usage, errors, and final status
```

The backend product run id is the source of truth. The agent runtime run id is only an execution handle inside the current container/runtime instance.

Runtime calls may pass compatibility headers understood by the upstream engine:

```text
Authorization header carrying runtime-manager internal auth
X-Hermes-Session-Id: <backend conversation/thread id>
X-Hermes-Session-Key: <backend-minted memory scope>
X-KarinAI-Run-Id: <backend product run id>
X-KarinAI-Workspace-Id: <backend workspace id>
```

`X-Hermes-Session-Id` and `X-Hermes-Session-Key` are not auth, tenant isolation, or routing authority. They are compatibility inputs for upstream session/memory behavior. Auth, tenancy, container routing, and workspace locks belong to the backend/runtime-manager.

## Managed runtime mode

`karinai-agent` should expose an explicit managed runtime mode for containers started by KarinAI runtime-manager.

Managed mode assumptions:

- Private API server enabled.
- `/v1/runs` available.
- Dashboard disabled by default.
- Public bind disabled.
- Runtime accepts calls only from backend/runtime-manager using an injected internal key/token.
- Tool policy is rendered by backend/runtime-manager, not trusted from user-editable state.
- Local autonomous cron is disabled or bridged to backend-owned schedules.
- Plugin installation is disabled unless product-allowlisted.
- Raw provider, cloud, storage, GitHub, and platform secrets are not present in the container.
- Model/tool access goes through trusted gateways with scoped runtime tokens.
- One active run per workspace is enforced outside the container by backend/runtime-manager.

Representative runtime inputs:

```text
KARINAI_MANAGED_RUNTIME=true
KARINAI_USER_ID=<backend user id>
KARINAI_WORKSPACE_ID=<backend workspace id>
KARINAI_WORKSPACE_DIR=/workspace
KARINAI_RUNTIME_STATE_DIR=/workspace/.hermes
HERMES_HOME=/workspace/.hermes
HOME=/workspace/.hermes/home
HERMES_WRITE_SAFE_ROOT=/workspace
API_SERVER_ENABLED=true
API_SERVER_HOST=<private bind>
API_SERVER_PORT=<private port>
API_SERVER_KEY=<container-internal key>
KARINAI_ENABLED_TOOLSETS=<backend-rendered beta tool policy>
KARINAI_MODEL_GATEWAY_URL=<trusted internal model gateway>
KARINAI_MODEL_GATEWAY_MODEL=<backend-selected model alias>
KARINAI_IMAGE_GATEWAY_URL=<trusted internal image gateway>
KARINAI_IMAGE_GATEWAY_PROVIDER=<backend-selected image provider alias>
KARINAI_IMAGE_GATEWAY_MODEL=<backend-selected image model alias>
KARINAI_TOOL_GATEWAY_URL=<trusted internal tool gateway>
KARINAI_RUNTIME_TOKEN=<scoped runtime token>
KARINAI_LOCAL_CRON_ENABLED=false
KARINAI_PLUGIN_INSTALL_ENABLED=false
KARINAI_DASHBOARD_ENABLED=false
```

Implementation detail: non-secret behavior should eventually be represented in rendered runtime config files where that fits upstream Hermes conventions. Environment variables are acceptable for container identity, injected secrets/tokens, and bootstrap handoff from runtime-manager.

## Beta tool policy

The first beta should make the KarinAI agent useful inside the sandbox while keeping platform boundaries strict.

Allowed:

- Agent reasoning and chat turns.
- Workspace file read/search/write/patch under the approved workspace path.
- Terminal and code execution inside the user container.
- Git inspection and local repo operations inside the workspace.
- Controlled web/documentation lookup if allowed by backend policy.
- Structured event streaming for tool calls, file edits, commands, errors, and artifacts.

Restricted:

- Network egress should be policy-gated and should block metadata/control-plane endpoints.
- Git push and other external side effects should require approval or backend-mediated credentials.
- Long-running/background processes should be tracked by the current run or registered with backend/runtime-manager.
- Memory and skills should be product-scoped and auditable before becoming broadly user-editable.

Disabled by default:

- Container-local durable cron scheduling.
- Arbitrary plugin installation.
- Host Docker/Kubernetes/container runtime access.
- Host filesystem mounts or shared tenant volumes.
- Raw model/provider/cloud/SaaS credentials.
- Public dashboard/API exposure.
- Writes outside approved workspace/runtime-state paths.

## Cron and scheduled work

The KarinAI agent may help the user create scheduled work, but the backend owns schedules.

Managed mode should not let the upstream local scheduler become a hidden durable scheduler. Instead, scheduling should use a backend-mediated capability such as:

```text
create_schedule_intent(title, prompt, schedule, timezone, workspace_id, conversation_id)
update_schedule_intent(schedule_id, ...)
delete_schedule_intent(schedule_id)
list_schedules(workspace_id)
```

The backend validates auth, quota, schedule frequency, workspace ownership, security policy, billing status, and confirmation requirements. At due time, the backend scheduler creates a product run, wakes/restores the container through runtime-manager, and invokes `/v1/runs` with a self-contained scheduled task payload.

## Secrets and gateways

The user container is not a trusted place for platform-wide secrets. Code running inside the container must not be able to read provider API keys, cloud credentials, object-storage admin credentials, GitHub app secrets, or backend service credentials.

The KarinAI agent should receive only scoped runtime tokens for trusted model/tool gateways. Gateways hold real provider credentials outside the user container, enforce policy and quota, and report usage back to backend run records.

When `KARINAI_MODEL_GATEWAY_URL` is set, managed startup renders the upstream-compatible model config inside `HERMES_HOME/config.yaml` with a single custom provider named `karinai-model-gateway`. That config stores `key_env: KARINAI_RUNTIME_TOKEN`, not the token value and not raw upstream provider keys. The backend-selected `KARINAI_MODEL_GATEWAY_MODEL` becomes the default model for the agent process.

When `KARINAI_IMAGE_GATEWAY_URL` is set, managed startup renders `image_gen.provider=karinai-image-gateway` so the `image_generate` tool calls the trusted image gateway with `KARINAI_RUNTIME_TOKEN`. Image provider credentials and routing stay in the gateway service; the managed agent receives only the gateway URL, optional provider/model hints, and scoped runtime token. If no image gateway URL is configured, managed runtime removes stale `image_gen` config and the image tool fails closed rather than falling back to direct upstream providers inside the user container.

## Prompt and branding contract

Product-facing identity must be template/config-driven, not hardcoded through upstream files. Managed mode should render KarinAI prompts from product-owned templates and runtime variables such as assistant name, product name, company/brand name, workspace id, and policy mode.

See `docs/karinai-prompt-branding.md` for the prompt/branding plan.

## Upstream sync contract

This repo should remain an upstream-tracking fork. Keep KarinAI-specific integration under `karinai/`, docs, config/templates, runtime entrypoints, and product tests where possible. If a core upstream file must change, document it in `docs/karinai-patches.md`.

Before adding substantial runtime code, sync from upstream Hermes and run targeted tests around the API server, Docker/runtime startup, tool policy, cron behavior, and any touched files.

## Tests to add with implementation

Future product tests should verify:

- Managed mode starts only with internal API auth configured.
- `/v1/runs` is available in managed mode.
- Product prompts render KarinAI identity and do not leak product-facing “You are Hermes Agent” text.
- Dashboard/public bind are disabled by default.
- Local cron is disabled or bridged to backend schedule intent.
- Plugin installation is disabled by default.
- Beta tool policy exposes only approved tools/toolsets.
- Workspace file operations are scoped to the approved workspace path.
- Runtime config does not require raw provider keys inside the container.
- Schedule creation produces backend schedule intent rather than local durable cron state.
