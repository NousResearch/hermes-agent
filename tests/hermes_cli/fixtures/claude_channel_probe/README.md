# ESN-0407 disposable Claude Channel proposal

This is a review-only acceptance fixture excluded from production packaging.
Two live attempts ran on 2026-10-25 after durable subscription login. The first
used background mode and blocked at Claude's mandatory interactive development-
Channel confirmation; its question expired without delivery or reply. The
second used the reviewed foreground launcher with human consent. Its private
socket and exact idle provider session were verified, and one bounded question
was admitted. At 29.906 seconds the ad-hoc sender received a non-matching
envelope but failed to preserve it before validation, so timing cannot establish
whether it was an expiry, another error, or a late reply. Both provider sessions,
sockets and temporary dependencies were removed. These attempts establish
consent, expiry-bound cleanup and an evidence gap; neither establishes a
successful two-way Channel reply. `send-once.mjs`, the private debug file and
the evidence requirements below close that gap for any separately authorized
repeat.

## Dependency and consent boundary

The fixture pins `@modelcontextprotocol/sdk` 1.32.2 from the npm registry:

- source: `https://registry.npmjs.org/@modelcontextprotocol/sdk/-/sdk-1.32.2.tgz`
- integrity: `sha512-SWsVtH/0vULSQStoZ6xxH2K09Li8fXYTMC64ZeogCdpZ3XSi1eCPOByV1OpuJxuI3OcmHhosDWYyii07tpPsLw==`
- license: MIT (`@modelcontextprotocol/sdk` package metadata)

`package-lock.json` pins all 95 transitive packages and their registry
integrities. `prepare.sh` first scrubs API/provider credential environment,
requires `claude auth status --json` to report first-party `claude.ai`
subscription authentication, verifies the lock and SDK integrity, then runs
`npm ci --ignore-scripts --no-audit --no-fund`. API-key, Bedrock, Vertex and
Foundry fallback are refused.

Claude Code Channels are a research preview. Custom Channels are not on the
approved allowlist, so the official reference requires this launch gate:

```text
--dangerously-load-development-channels server:esn0407_channel
```

That flag is explicit security consent to load unapproved Channel code. The
operator authorized one disposable install and launch on 2026-10-25. An
independent safety review still holds the launch; do not install or run this
fixture until that review passes. Team and Enterprise organizations may also
need an administrator to enable Channels.

Official references:

- https://code.claude.com/docs/en/channels
- https://code.claude.com/docs/en/channels-reference

## Intended disposable launch

After independent review passes:

1. Copy this directory into a fresh child of the OS temporary directory, owned
   by the current user and mode `0700`. Preparation refuses a directory outside
   the OS temporary root, a directory inside a Git repository, or any directory
   accessible by group or other users.
2. Run `./prepare.sh`. It installs exactly the lockfile into that private copy
   with lifecycle scripts disabled. Do not run `npm install` or update the lock.
3. Replace every placeholder in `mcp.json.template` from one authenticated
   Roots ticket read. Generate a fresh 256-bit nonce and one correlation ID for
   this single question. Do not write either value to logs or use an operator
   or production project directory.
4. Generate and record a fresh provider session UUID, then launch one foreground
   subscription-authenticated Claude session from the stable empty work
   directory inside the durable provider-authentication profile. That directory
   is mode `0700`, contains no project files, and has one reviewed persistent
   trust record in the same profile namespace. It is not a per-run temporary
   workspace. The manifest's `production_service_active: false` is a declared
   invariant, not process evidence: immediately before launch, run
   `claude agents --json` under this `CLAUDE_CONFIG_DIR` and refuse the probe if
   any session is active; `launch.sh` enforces that check immediately before
   starting Claude while holding an atomic profile-scoped lease for the full
   process lifetime. A concurrent launcher is refused, and a dead owner's lease
   is recovered only by an atomic rename plus owner recheck. Custom development
   Channels require an
   interactive warning: the operator must select **I am using this for local
   development** before the Channel registers. `launch.sh` intentionally does
   not use `--bg`, because a background launch blocks at that confirmation and
   silently leaves the Channel unable to deliver. After the terminal displays
   the Channels notice, keep it open while the separate sender submits the one
   bounded question. `launch.sh` repeats the credential scrub
   and subscription-auth check immediately before `exec` so the Claude child
   cannot fall back to an API or cloud-provider credential. It makes only
   `mcp__esn0407_channel__reply` visible with `--tools` and explicitly permits
   that exact tool with `--allowedTools`, so `dontAsk` cannot deny the required
   first reply. It refuses inbound cross-session messages and loads only the
   temporary MCP config. It also writes MCP-only debug diagnostics to a new
   mode-0600 file in the disposable fixture directory. The launcher opens that
   file with exclusive creation and no symlink following, validates the opened
   inode, and passes the still-open descriptor to Claude as `/dev/fd/3`; Claude
   never reopens the directory pathname. The official Channel
   reference says a successful `mcp.notification()` write is not delivery
   confirmation: an unregistered or organization-blocked Channel may be dropped
   silently. Before sending, the terminal must show the exact dim Channels
   registration notice. After sending, retain the debug file and the sender's
   raw-response and verdict evidence until their hashes and bounded findings are
   recorded:

```text
ESN0407_MCP_CONFIG=/PRIVATE_TMP/mcp.json \
ESN0407_WORK_DIR=/PERSISTENT_HERMES_AUTH_PROFILE/work \
ESN0407_PROFILE_ROOT=/PERSISTENT_HERMES_AUTH_PROFILE \
ESN0407_SESSION_ID=/PRE_REGISTERED_UUIDV4 \
ESN0407_DEBUG_FILE=/PRIVATE_TMP/mcp-debug.log \
CLAUDE_CONFIG_DIR=/PERSISTENT_HERMES_AUTH_PROFILE/config \
  /PRIVATE_TMP/launch.sh
```

5. Only after the interactive Channels notice is visible, run `send-once.mjs`
   with absolute paths in `ESN0407_CHANNEL_SOCKET`, `ESN0407_QUESTION_FILE`,
   `ESN0407_MCP_CONFIG`, `ESN0407_RAW_RESPONSE_EVIDENCE`, and
   `ESN0407_RESPONSE_VERDICT`. The sender writes the first bounded response as
   base64 plus its SHA-256 to a new mode-0600 evidence file and fsyncs it before
   parsing or validating it. It writes a separate verdict referencing that hash.
   Terminal output contains only an outcome and hash, never raw response text.
   Then send one envelope,
   at most 4 KiB, over the private Unix socket. It must bind
   the exact Roots ticket, assignment, attempt, owner, revision and authority
   event digest; its authority is `answer-only-no-changes`; TTL is at most 30
   seconds. The envelope must also echo the one-time nonce and pre-registered
   correlation ID. Ask for a fixed short marker and no action.
6. Keep the socket open for the correlated reply. For cancellation, send the
   cancel envelope or close the connection. The Channel cannot retract an event
   already delivered to Claude; it invalidates the waiter and rejects a late
   reply instead.
7. Exit the foreground Claude session and verify that exact UUID is no longer
   active. The installed CLI exposes deletion only for background sessions, so
   the bounded foreground transcript remains durable provider state under the
   pre-registered UUID; do not claim it was deleted. It may contain only the
   startup/consent event, the one bounded question and its reply-tool result.
   Remove the private temporary MCP, socket and dependency directory and confirm
   they no longer exist. The durable provider-authentication profile, its stable
   empty work directory, reviewed trust record and bounded transcript remain.

Do not add `--bare`. Claude Code 2.1.306, installed during the initial review,
documented that bare mode
never reads OAuth or the keychain and accepts only `ANTHROPIC_API_KEY` or an API
key helper. That conflicts with this probe's subscription-only requirement and
credential scrub. The foreground live attempt later reported Claude Code
2.1.310; its subscription preflight passed and the launcher still omitted
`--bare`.

### Superseded isolated trust bootstrap experiment

The launcher requires workspace trust even for its empty restricted
working directory. Installed 2.1.306 identifies the state as
`projects[<absolute path>].hasTrustDialogAccepted` in `.claude.json`.
`trust-bootstrap-plan.mjs` validates two empty, current-user-owned mode-0700
sibling directories under the OS temporary root and emits this exact proposed
write without performing it:

```json
{
  "projects": {
    "/PRIVATE_DISPOSABLE_PARENT/work": {
      "hasTrustDialogAccepted": true
    }
  }
}
```

The target is `/PRIVATE_DISPOSABLE_PARENT/config/.claude.json`, created mode
`0600`, with `CLAUDE_CONFIG_DIR` set to that `config` directory only for the
disposable Claude process. The exact additional consent proposed by the plan is:
"Authorize one trust record for the empty disposable Claude workspace
`/PRIVATE_DISPOSABLE_PARENT/work`, written only to
`/PRIVATE_DISPOSABLE_PARENT/config/.claude.json`, removed with its temporary
parent, and never copied to global config?" The before state is an absent file;
the only intended semantic diff is the path-scoped boolean above.
Cleanup removes the shared disposable parent, including config, work,
dependencies and socket. Nothing is copied to `~/.claude.json`.

This experiment established that copying `.claude.json` cannot move OAuth or
keychain identity into a new namespace. It is retained as a negative control,
not as the current launch path. The operator instead completed the supported
interactive `claude auth login --claudeai` flow in a durable, private
`CLAUDE_CONFIG_DIR`; no OAuth token was extracted or copied. Per-run cleanup
must never remove that durable authentication directory. Provider login remains
separate from the Roots ticket, role, revision and tool authority required for
each managed worker start.

This reply-only transport probe intentionally loads no `CLAUDE.md`, skill,
hook, project MCP server or Roots tool. Its authority and route are supplied by
the explicit reviewed Channel configuration, and the only visible and allowed
tool is the correlated reply. A future managed-worker bootstrap is a separate
contract: its receipt must bind the actual governing instruction source, role,
task, ticket, revision, DoD and context, plus authenticated scoped Roots MCP
tool authorization. A self-attested prompt or this transport receipt cannot
prove those properties.

The managed-worker authentication design is governed by **P-0593**, “Onboarding
a principal: declare the grant, deploy, and issue a credential only agents
need,” and owned by **ESN-0030**, “Retire OpenClaw completely; rebuild agents as
principals.” **ESN-0169** preserves the boundary that Claude Code acts through
operator delegation and no agent may hold `write_ticket`; **ESN-0171** requires
an unattended credential to be bounded by both its delegator and the receiving
agent's declared ceiling. The concrete proposal is recorded in ESN-0407 comment
`80063559-7e14-4eab-90c6-c0169e02cd84`; its dependency on the owning parallel
authentication work is recorded in ESN-0407 comment
`83aa9075-e97f-4837-bcc9-a5da4f110413`.

This fixture does **not** implement parallel authentication, mint a principal
credential, load an operator credential, or prove the P-0593/ESN-0030 managed
bootstrap. It only tests the separate reply-only Channel transport boundary.

## Permission and transport scope

- Local Unix socket only; there is no TCP, HTTP, cloud or external listener.
- Parent directory is pre-created as `0700`; the socket is changed to `0600`.
- The MCP child communicates with Claude over stdio.
- The fixture declares `claude/channel` and one reply tool. It deliberately
  does not declare `claude/channel/permission`, so it cannot relay approvals.
- Restricted mode, `dontAsk`, the tool allowlist and empty setting sources keep
  the disposable worker from gaining file, shell, browser or inherited project
  authority. Administrator-managed settings still apply and must be inventoried
  before launch; stop if they widen the disposable worker's authority. These
  launch controls supplement provider identity; prompt wording alone cannot
  establish authority or non-interruption.

## Required proof

Acceptance must record the disposable provider session identity, Channel child
identity, Roots route binding and correlation ID. It must show:

1. the operator accepted the custom development-Channel warning in the exact
   foreground session before the question was sent;
2. the question arrived in the exact disposable session at a provider safe
   point;
3. the reply tool echoed the correlation ID and exact marker;
4. no file, command, browser, agent or approval tool ran;
5. cancellation stopped the caller wait and a later reply was rejected without
   claiming delivery retraction;
6. the exact pre-registered Claude session UUID is no longer active, the Unix
   socket and temporary dependency directory were removed, and durable provider
   authentication, its stable empty workspace trust and its explicitly bounded
   retained transcript remained intact.

Socket readiness proves only that Claude spawned the MCP child. It does not
prove Channel registration, notification delivery, model processing, or a reply.
For a live proof, preserve the startup registration notice, the inbound `←`
terminal line, the disposable MCP debug log, raw response evidence and verdict.
Classify `remote_expired`, `remote_error`, `invalid_response`, and `local_timeout`
separately; timing alone is not a transport verdict.

No result from this fixture establishes access to a Claude session that was not
started with this Channel.
