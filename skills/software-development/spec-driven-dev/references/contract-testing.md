# Contract Testing: Two-Layer Approach

Two layers, not one-or-the-other, applied AFTER `/speckit-plan` has
settled the tech stack and the API surface is known.

## Layer 1: Shared Schema-First Contract

A single JSON Schema / OpenAPI document is the source of truth every
client and the server test against. Lives at `contracts/<name>.yaml` (or
`.json`) in the project. Interface violations (wrong field name, wrong
type, missing required field) are caught here, against every consumer at
once.

## Layer 2: Consumer-Driven Contract Tests (Pact-Style)

Each client (e.g. a phone app vs. a work-laptop client hitting the same
API) writes its OWN contract test asserting only the specific slice it
actually depends on. Schema-valid and actually-compatible are not the
same thing: a response can match the shape and still break what one
particular consumer needed. Pact (pact-python, pact-js, or the language
equivalent for the client's stack) is the reference implementation of
this pattern -- pick the library matching the client's language rather
than forcing everything through one runtime.

## Build Ordering (client-first)

1. **Generate a mock server from the contract.** `generate_mock_server.py`
   in this skill wraps Stoplight Prism
   (`npx @stoplight/prism-cli mock <contract>`) to do this directly from
   the OpenAPI/JSON Schema file -- no server implementation needed yet.
2. **Build and consumer-test the client against the mock first.** This is
   where the real per-client Pact contract gets written: the consumer
   test runs against the mock provider, producing a contract file the
   real server verifies against later.
3. **Write E2E tests incrementally during client build, against the
   mock** -- one per contract interaction. Same assertions as later;
   only the target (mock vs. real server) swaps.
4. **Implement the server** to satisfy the same shared contract.
5. **Provider verification** -- replay the consumer contracts from step 2
   against the real server. This confirms the server satisfies what
   clients actually need, not just what the schema technically allows.
   Repoint the E2E tests from step 3 at the real server.

## Why Client-First (Not Server-First or True-Parallel)

For a solo developer there's no team-parallelism benefit to building
client and server at once, but there is a real sequencing benefit to
client-first: it forces proving the contract is actually usable before
sinking implementation time into the server, and the visible product
(what people interact with) gets built and validated before the
invisible one. Server-first risks building exactly what the schema says
and only later discovering the client needed something slightly
different -- the exact failure mode consumer-driven contract testing
exists to catch.

## When This Layer Applies

Only relevant once a project has a real client/server or multi-consumer
boundary. A single-process script or CLI tool has no contract-testing
layer to build -- skip straight from `/speckit-implement` to hand-written
plus property-based tests (see `references/ears-syntax.md`).
