# OneCLI, MCP, Notion, and Memory Integration Changes

Date: 2026-04-19

This document records the root-cause fixes made to Hermes' MCP launch path and
the local operational changes needed to make Notion MCP and MemPalace work
reliably behind OneCLI. It also captures the current memory-policy split
between MemOS and MemPalace after alignment with the local usage spec.

## Scope

There are two classes of changes:

1. Tracked `hermes-agent` code changes in this repository
2. Local deployment changes in `~/.hermes` that are not currently tracked by
   this repository

The tracked code changes are the portable root-cause fixes for stdio MCP
subprocess environment construction. The local deployment changes are necessary
to keep the installed Notion MCP package and the custom `memos_palace` provider
working in this machine's live runtime.

## Root cause: unresolved proxy placeholders leaked into MCP subprocesses

The original failure mode for Notion MCP was that stdio subprocesses were
receiving environment values like:

- `HTTPS_PROXY=${https_proxy}`
- `http_proxy=$http_proxy`

Those values are literal strings by the time a subprocess sees them. Node does
not recursively expand shell placeholders inside inherited environment values,
so outbound HTTP clients tried to resolve `${https_proxy}` as a host and failed.

### Tracked fix in `tools/mcp_tool.py`

The launcher now does three things:

1. Drops unresolved shell-style placeholders from configured subprocess env
2. Supports an explicit opt-in for OneCLI-derived proxy and CA injection via
   `NODE_USE_ENV_PROXY=1`
3. Prefers already-correct parent proxy and CA env, then falls back to
   container-config bootstrap discovery when the parent env is missing

Important properties:

- Explicit env values still win
- Real API keys are not injected into the subprocess env
- OneCLI remains the credential-isolation boundary

### New helper module

`hermes_cli/onecli_bootstrap.py` was added to normalize OneCLI bootstrap data.
It is responsible for:

- fetching the first working container-config endpoint
- extracting proxy URL and CA material from multiple payload shapes
- injecting proxy auth tokens into the bootstrap proxy URL
- rewriting `host.docker.internal` to `172.17.0.1` when Hermes is running on
  the host and the Docker alias is not resolvable
- materializing CA PEM data into temp files when only inline PEM is available

### Test coverage

The tracked tests now cover:

- dropping unresolved env placeholders
- inheriting parent proxy and CA env when already present
- falling back to OneCLI bootstrap discovery
- preserving explicit proxy and CA overrides
- bootstrap payload normalization and shell/json output paths

Files:

- `tools/mcp_tool.py`
- `tests/tools/test_mcp_tool.py`
- `hermes_cli/onecli_bootstrap.py`
- `tests/hermes_cli/test_onecli_bootstrap.py`

## Local Notion MCP runtime fix

The installed `@notionhq/notion-mcp-server` package was still failing in this
environment even after the launcher env fix. OneCLI proxying and auth rewrite
were working, but the package's outbound Node transport was not behaving
correctly under the local runtime.

Local operational fix applied on this machine:

- patched the installed package transport under:
  - `~/.hermes/notion-mcp/node_modules/@notionhq/notion-mcp-server/src/openapi-mcp-server/client/http-client.ts`
- regenerated the package CLI bundle:
  - `~/.hermes/notion-mcp/node_modules/@notionhq/notion-mcp-server/bin/cli.mjs`
- updated local Hermes config to launch the real Notion MCP binary directly
  instead of routing through the earlier shell-wrapper workaround

Result:

- `API-post-search` returned real Notion data
- `API-post-page` succeeded
- test page created under the target workspace:
  - `Hermes MCP Transport Fix`

These package-level changes are outside this repository and should be converted
into either:

- an upstream fix to the Notion MCP package
- a tracked local wrapper package
- or a reproducible local patch step

The current state works, but it is still deployment-local.

## Local MemPalace TLS fix

The TLS error was not on MemOS proper. MemOS is configured as plain HTTP on the
Tailscale network. The certificate failure occurred on the MemPalace HTTPS MCP
bridge.

Observed error:

- `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`

The root service-level issue was that the Hermes gateway had drifted away from
the OneCLI startup path and was launching Python directly, so the live process
was no longer guaranteed to inherit the correct proxy and CA environment.

Local operational fix applied on this machine:

- restored gateway startup through the OneCLI wrapper:
  - `~/.config/systemd/user/hermes-gateway.service`
  - `ExecStart=~/.hermes/bin/hermes-gateway-onecli`
- updated the wrapper to publish a stable combined CA path:
  - `/tmp/onecli-combined-ca.pem`

Result:

- the live gateway again starts with OneCLI proxy and CA env
- direct `httpx` probes stopped failing with certificate verification errors
- the failure class moved from TLS trust to higher-level request behavior

This matters for Python clients such as the custom `memos_palace` provider,
which use `httpx` and need a deterministic CA bundle path when talking to
MemPalace over HTTPS.

## Memory policy alignment

The local `MemOS+MemPalace Usage Specs.md` established the intended split:

- MemOS is the primary semantic memory layer
- MemPalace is the verbatim archive and fallback exact-recall layer
- Markdown memory remains the human-auditable continuity source of truth

Hermes' custom provider was then aligned to that policy, but with one important
clarification taken from NanoClaw / Deme:

- MemPalace should still receive ordinary non-trivial turn archival writes
- MemOS remains the semantic authority and promotability-gated store

Current live behavior in the local `memos_palace` provider:

- MemOS:
  - primary semantic recall
  - promotability-gated turn storage
- MemPalace:
  - fire-and-forget verbatim turn archive via `add_drawer`
  - fallback retrieval for exact wording, transcript-style recall, and
    evidence-oriented questions

Current MemPalace archival shape:

- content:
  - `User: <sanitized text>`
  - `Assistant: <sanitized text>`
- location:
  - `wing=agent_main`
  - `room=conversations`
- tags:
  - source tag such as `hermes`
  - `verbatim_archive`
  - `scope:<...>`
  - `viewer:<...>`
  - `subject:<...>`
  - `conversation:<...>`
  - `participant:<...>`
  - `platform:<...>`
  - `context:<...>`

This mirrors the Deme/NanoClaw archival model while keeping MemOS as the
primary semantic layer.

## What is tracked vs not tracked

Tracked in this repository:

- MCP env sanitization and OneCLI bootstrap support
- tests for the launcher and bootstrap logic
- this design/ops note

Not tracked in this repository:

- `~/.hermes/config.yaml`
- `~/.hermes/onecli-backend/config.yaml`
- `~/.hermes/bin/hermes-gateway-onecli`
- `~/.config/systemd/user/hermes-gateway.service`
- `~/.hermes/plugins/memos_palace/__init__.py`
- installed Notion MCP package patches under `~/.hermes/notion-mcp/...`

If these local operational changes should be reproducible from source, the next
step is to move them into one or more tracked homes:

1. vendor or wrap the Notion MCP transport patch
2. move `memos_palace` into a tracked plugin repo or into `hermes-agent`
3. check in the gateway/OneCLI wrapper and service-unit policy in a repo that
   owns local deployment

## Recommended follow-ups

1. Create a tracked home for the `memos_palace` provider so memory behavior is
   not defined only by mutable local files.
2. Upstream or vendor the Notion transport fix so rebuilds do not discard it.
3. Treat the OneCLI wrapper as the mandatory gateway launch path in deployment
   docs and templates.
4. Add a reproducible smoke test that validates:
   - stdio MCP sees sane proxy env
   - Notion MCP can complete a search
   - MemPalace HTTPS no longer fails certificate verification

## Hermes Spark client-node cutover status

Update: 2026-04-20

Hermes Spark on `rj-spark` has now been cut over to the dedicated client-node
OneCli handoff model with runtime-specific proxy identity and explicit CA
material supplied through a local env file.

Operationally, the cutover was completed with:

- a dedicated local env file at `~/.config/onecli/hermes-spark-proxy.env`
- dedicated CA files under `~/.config/onecli/certs/`
- wrapper changes that make explicit env-file CA paths win over bootstrap CA
  material

Important result:

- bootstrap-derived proxy material is no longer required for Hermes Spark
- bootstrap-derived CA material is now fallback-only when explicit CA env is
  present

Validated outcomes:

- the live gateway process runs with the dedicated `ONECLI_PROXY_URL` from the
  installed env file
- the live gateway process keeps explicit CA paths from
  `~/.config/onecli/certs/` instead of rewriting them to temp bootstrap files
- a real Notion MCP provider call succeeds through the dedicated env file

Failure classification during cutover:

- dedicated proxy identity alone was not sufficient at first
- the first failure class was TLS trust, not connectivity, proxy auth, or
  upstream Notion behavior
- the dedicated handoff CA bundle resolved that TLS failure

The validated end state for Hermes Spark is:

- dedicated proxy identity: active
- real provider validation: passed
- bootstrap dependency for proxy material: removed
- bootstrap dependency for CA material: reduced to fallback only
