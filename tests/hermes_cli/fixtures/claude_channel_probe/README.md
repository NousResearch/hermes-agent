# ESN-0407 disposable Claude Channel proposal

This is a review-only acceptance fixture. It is excluded from production
packaging and has not been installed or launched.

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
4. Launch one background subscription-authenticated Claude session from an
   empty temporary working directory. `launch.sh` repeats the credential scrub
   and subscription-auth check immediately before `exec` so the Claude child
   cannot fall back to an API or cloud-provider credential. It makes only
   `mcp__esn0407_channel__reply` visible with `--tools` and explicitly permits
   that exact tool with `--allowedTools`, so `dontAsk` cannot deny the required
   first reply. It refuses inbound cross-session messages and loads only the
   temporary MCP config:

```text
ESN0407_MCP_CONFIG=/PRIVATE_TMP/mcp.json \
ESN0407_WORK_DIR=/PRIVATE_EMPTY_WORK_DIR \
  /PRIVATE_TMP/launch.sh
```

5. Send one envelope, at most 4 KiB, over the private Unix socket. It must bind
   the exact Roots ticket, assignment, attempt, owner, revision and authority
   event digest; its authority is `answer-only-no-changes`; TTL is at most 30
   seconds. The envelope must also echo the one-time nonce and pre-registered
   correlation ID. Ask for a fixed short marker and no action.
6. Keep the socket open for the correlated reply. For cancellation, send the
   cancel envelope or close the connection. The Channel cannot retract an event
   already delivered to Claude; it invalidates the waiter and rejects a late
   reply instead.
7. Stop and remove the disposable Claude background session, then remove the
   private temporary directory. Confirm that the socket, session and dependency
   directory no longer exist.

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

1. the question arrived in the exact disposable session at a provider safe
   point;
2. the reply tool echoed the correlation ID and exact marker;
3. no file, command, browser, agent or approval tool ran;
4. cancellation stopped the caller wait and a later reply was rejected without
   claiming delivery retraction;
5. the Claude session, Unix socket and temporary dependency directory were
   removed.

No result from this fixture establishes access to a Claude session that was not
started with this Channel.
