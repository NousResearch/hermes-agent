# Muncho public Discord connector boundary

This boundary fronts the existing Hermes `RelayAdapter`; it does not add a
model tool, a prompt rule, a keyword classifier, or a semantic dispatcher.
GPT/Hermes receives the exact admitted message text and decides its meaning.
Deterministic code is limited to peer identity, permissions, schemas, bounds,
deadlines, idempotency, and receipts.

## Runtime shape

- `muncho-discord-connector.service` is the privileged token owner for ordinary
  Discord ingress and ordinary conversational replies. It admits only explicitly allowlisted users in explicitly
  allowlisted public guild text/news channels and public/news threads whose
  parent is allowlisted. Discord DMs, group DMs, private threads, non-public
  targets, and targets the bot cannot use are rejected before gateway/model
  delivery or Discord dispatch.
- `hermes-cloud-gateway.service` has no Discord credential. It uses the existing
  connector-fronted Relay adapter over the one pinned Unix socket. Both sides
  require the peer UID and the exact current systemd `MainPID` on every frame.
- The connector's SQLite journal is first-wins. Inbound events are acknowledged
  only after the gateway consumer accepts the exact event. Outbound sends move
  to `dispatching` before Discord I/O and are never blindly retried from an
  uncertain state. Hermes reports success only after an exact, digest-bound
  receipt includes the Discord message id and a successful live readback.
- Production readiness is published only after a real Discord session, exact
  live public/sendable proofs for every configured channel, and an accepting
  Unix listener are all present. A Discord disconnect clears readiness and
  terminates the service; the gateway is bound to that service and cannot keep
  presenting a healthy Discord path. The production unit restarts on failure.
  The separately generated isolated-canary unit remains bounded to 15 minutes
  and deliberately does not restart.

The existing signed `discord_edge_*` REST boundary remains the only execution
path for Canonical `route_back` mutations. A signed edge request, capability,
or receipt is not an accepted connector metadata field, and the connector never
manufactures `route_back.sent`. The Canonical Writer may record that terminal
event only after it verifies the signed edge receipt. Conversely, ordinary
conversation replies never enter the signed route-back protocol and use a
separate connector idempotency namespace. This keeps operation classes and
journals disjoint even when both privileged services use the same Discord bot
identity: connector = ordinary session traffic; signed edge = Canonical
route-back only.

The package templates are:

- `ops/muncho/systemd/muncho-discord-connector.service.in`
- `ops/muncho/systemd/hermes-cloud-gateway.discord-connector.conf`
- `ops/muncho/systemd/discord-public-connector.json.in`

The service template must be rendered from the immutable 40-character release
SHA into the production `hermes-agent-<first12>` release directory. The JSON
template must be rendered with numeric service identities and
exact Discord IDs. Placeholder-bearing artifacts are never startable.

## Gateway token retirement and privileged leases

Cutover is a stopped, fail-closed transition for the gateway's old direct token
lease. It must not create a period where the gateway/direct Discord adapter and
connector can both use the token. The separately hardened signed route-back edge
may retain its explicit service-owned credential lease; it is not a gateway
adapter and cannot consume ordinary connector frames.

1. Stop `hermes-cloud-gateway.service` and its direct Discord adapter. Verify the
   gateway `MainPID` is zero. Do not retire the separately approved signed
   `discord_edge_*` route-back service lease.
2. Move the token without printing it into
   `/etc/muncho/discord-connector-credentials/bot-token`. The directory is
   connector-owned mode `0700`; the one-link, non-symlink token file is
   connector-owned mode `0400`. If an atomic rename is impossible, copy while
   the gateway is stopped, verify equality internally, delete the old gateway
   copy, and only then continue. A surviving gateway-readable token path blocks
   activation. The signed edge's separately owned mode-`0400` credential is not
   gateway-readable and is audited as a distinct route-back-only lease.
3. Remove `DISCORD_BOT_TOKEN`, `DISCORD_TOKEN`, and any Discord token value from
   the gateway `.env`, unit environment, config, and secret mounts. Configure
   `platforms.discord.enabled: false`. Install the gateway drop-in, which enables
   only the pinned local Relay endpoint and unsets inherited Discord token names.
4. Render and install the root-owned connector config as mode `0440`, group
   `muncho-discord-connector`. Explicitly initialize the connector-owned journal
   once with `python -B -I -m gateway.discord_connector_bootstrap --config ...
   --bootstrap-journal`; normal startup refuses to create it.
5. Start the connector first. Its readiness requires the exact
   unit/user/MainPID, socket owner/group/mode, existing journal, Discord-ready
   state, and live public/sendable target proofs. Then start the gateway; its
   own readiness requires a credential-free reciprocal `hello` exchange over
   that exact connector socket.
6. Verify from `/proc/<gateway-mainpid>/environ`, the gateway unit, config, and
   open file descriptors that the gateway holds no Discord token/path. Verify a
   public-channel event and receipt-backed reply, and verify DM/private-thread
   attempts produce no gateway event and no Discord dispatch.

Shutdown is the reverse dependency order. Stop the gateway first, drain
connector request handlers, stop Discord ingress, then require a fresh journal
snapshot with no pending/delivering inbound events and no
prepared/dispatching/uncertain outbound send before retiring the connector
credential.

Rollback observes the same gateway lease: stop gateway and connector, move the
ordinary-session credential to one approved rollback owner, retire the connector
copy, remove Relay activation, and only then start the direct rollback adapter.
The signed edge lease remains route-back-only throughout.

The numeric string `1279454038731264061` supplied by the owner is only an input
to the rendered allowlist after live Discord type/publicness proof; this source
tree does not guess whether it is a guild, channel, thread, or user ID.
