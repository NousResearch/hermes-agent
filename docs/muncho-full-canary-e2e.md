# Full Muncho canary E2E evidence contract

`gateway.canonical_full_canary_e2e` is the packaged, deterministic verifier for
the final isolated canary. It is intentionally **not** a model or Discord
executor. A green verifier receipt means that a separate approved live driver
supplied exact service-generated evidence which satisfied every invariant; it
does not mean that an offline fixture called GPT or Discord.

## Required live sequence

1. The sealed canary lifecycle starts the privileged Discord edge, Canonical
   writer, and gateway from one exact release SHA and release-artifact digest,
   records their identities, and installs the owner-bound one-shot scope.
2. An approved external driver submits the fixture prompt to the authenticated
   loopback API server. Its bearer is read from a systemd credential file and
   is never placed in config, argv, environment, evidence, or logs. The gateway
   uses the canary's real GPT-5.6 Codex credential, and the privileged edge
   alone uses the distinct canary Discord token. This is control-plane ingress:
   the canary has no Discord ingress and does not claim to test one.
3. The sealed collector plugin captures bounded hook projections and performs
   the deterministic private-target edge probe; the external driver captures
   authenticated API/lifecycle facts. Neither path copies a credential into
   the fixture, evidence, logs, or command line.
4. The lifecycle invokes the packaged offline verifier with exact file
   digests after the API run has durably revoked its scope. It verifies the
   atomic tombstone/revocation receipt, then stops gateway, writer, and edge in
   that order even when verification fails.

There is no synthetic fallback. If the live driver, model credential, distinct
Discord application token, public allowlisted channel, Canonical readback, or
any receipt is unavailable, verification fails closed and the services are
stopped. The operator must never describe unit-test fixtures as live evidence.

## Invocation

Run from the sealed interpreter with bytecode disabled and isolated imports:

```bash
<sealed-python> -B -I -m gateway.canonical_full_canary_e2e verify \
  --start-receipt-sha256 <exact-full-canary-start-receipt-file-sha256> \
  --fixture /etc/muncho/full-canary/fixture.json \
  --fixture-sha256 <exact-fixture-file-sha256> \
  --fixture-gid <exact-gateway-primary-gid> \
  --evidence /run/muncho-full-canary/e2e-evidence.json \
  --evidence-sha256 <exact-evidence-file-sha256>
```

Both inputs must be absolute, normalized, single-link regular files of at most
8 MiB. Symlinks, duplicate JSON keys, non-JSON numeric constants, digest
mismatches, unknown schema fields, or malformed identifiers are rejected. The
module does not write a file, open a socket, query a database, invoke systemd,
or make a network request.

Success exits `0`; failure exits `2`. Standard output is one canonical JSON
object containing `schema`, `ok`, `fixture_sha256`, `evidence_sha256`, and
`invariant_receipt_sha256`. Success also carries the release/run IDs and fixed
invariant names, plus the exact lifecycle start-receipt file digest already
bound into runtime provenance. Failure carries one fixed non-secret
`failure_code`. Paths,
receipt bodies, model output, Discord content, environment values, and crypto
errors are never echoed.

## Fixture authority

The owner-approved `muncho-full-canary-e2e-fixture.v1` fixes only operator
policy and nonsemantic correlation:

- release SHA, sealed release-artifact SHA-256, run/case correlation, owner
  Discord ID, and execution window;
- the exact authenticated loopback session-create plus
  `127.0.0.1:8642/api/sessions/{session_id}/chat/stream` routes and their
  protocol version. The stream's `run.completed.messages` carries the full
  assistant/tool transcript needed for receipt collection;
- a bounded one-hour execution window;
- the exact verified route: `openai-codex`, `codex_responses`,
  `https://chatgpt.com/backend-api/codex`, `gpt-5.6-sol`;
- initial `high` and model-requested `xhigh` effort;
- a bounded exact prompt plus its digest and a minimum of three completed
  steps, without supplying their meaning or IDs;
- the public Discord target and canonical route-back idempotency key;
- the canary writer capability and edge receipt Ed25519 public keys.

Session/turn IDs, plan ID, step/criterion/verification IDs, plan content, and
route-back content/digest are deliberately absent: GPT authors or the live
runtime creates them. The fixture contains no evidence and grants no service
authority. Private keys, tokens, passwords, API keys, and database credentials
are forbidden from it.

## Required evidence

`muncho-full-canary-e2e-evidence.v1` is bound to the exact fixture file digest
and must say `execution_mode=live_isolated_canary` and `synthetic=false`. It
contains these externally produced receipts:

- the exact full-canary start-receipt digest and all three service identity
  digests already checked by the privileged lifecycle;
- process-bound Canonical writer startup readiness with a real typed PING;
- an authenticated loopback API request receipt containing only request/run,
  session/turn, peer/control and systemd-credential provenance. It carries no
  authentication bearer value or digest. Fresh random session/epoch digests
  are retained only to bind the one-shot writer scope and exact fixture prompt;
- Canonical readback proving the scope was owner-preapproved for this sealed
  release/fixture/run, claimed once by the exact API session generation, and
  revoked with `api_server_run_finished`. The revocation event must carry the
  transaction-bound `session_tombstone_recorded=true` receipt, while all three
  append-only lifecycle events remain readable to the trusted projection. The
  sealed SQL contract and real PostgreSQL 18 test prove that this tombstone
  denies the exact retired session generation;
- ordered sealed-collector model-call receipts proving the first live request
  used `high`, an assistant-authored `todo` call requested `xhigh` in the same
  turn, and all later requests in that turn used `xhigh`. Every receipt binds
  the actual sanitized `post_api` response payload digest, response model,
  tool-call IDs, and observation time; it never invents a provider response ID
  that the hook does not expose. Nonterminal calls must contain tool calls and
  the final receipt must contain none, matching Hermes's real agent loop;
- Canonical writer readback of GPT's own plan. The verifier derives its plan,
  step, dependency, criterion, and verification IDs, then proves acyclic
  dependencies, monotonic revisions, one completed step per transition,
  passing criterion coverage, the completed head, and its exact receipt set;
- the full writer-authorized Discord request and edge-signed receipt. The
  verifier checks both Ed25519 signatures, canonical authority, target,
  idempotency, model-authored content digest, adapter acceptance, public
  readback, message ID, and matching durable `route_back.sent` event;
- a live gateway-to-edge forbidden-DM probe. Because invalid edge frames
  correctly receive no fabricated signed receipt, this receipt binds the edge
  service/socket identity, observes connection close without a response, and
  proves the journal's logical digest and row count are unchanged;
- a gateway turn-completion receipt proving all model-authored steps completed
  in their observed order and a final model response exists. It is bound to
  the exact authenticated API run/message IDs and requires the real terminal
  event `run.completed`, `completed=true`, `failed=false`, `partial=false`,
  `interrupted=false`, and the healthy core exit reason
  `text_response(finish_reason=stop)`. It never substitutes a synthetic
  `model_completed` reason.

The verifier checks structured state only. It never searches task or message
text, routes on keywords, chooses reasoning effort, judges whether prose is a
good answer, or replaces GPT/Hermes decisions.

## Operational boundary

This verifier is safe to exercise locally with generated keys and fixtures,
but those tests prove only the verifier contract. A production decision needs
the separate live driver and canary-only credentials described above, plus the
sealed runtime lifecycle receipt which binds service identities and ordered
shutdown. Missing live execution remains a blocker; it must not be papered over
by changing provenance fields in a fixture.
