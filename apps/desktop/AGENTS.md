# Desktop Engineering Guide

How to build Hermes Desktop well. This is a judgment guide, not an inventory —
it teaches the invariants and the reasoning behind them so a change fits the app
even as files move. Read it with the repository `AGENTS.md` (root rules still
apply), [`DESIGN.md`](./DESIGN.md) for the visual and interaction contract, and
`src/AGENTS.md` for the backend contract, slash-palette curation, and Bot Mode.
The reasoning behind the shorter sections below lives in [`ENGINEERING.md`](./ENGINEERING.md).

When a rule here and the code disagree, trust the code and fix whichever is
wrong — but never break an invariant to make a change easier.

## What this app is

Desktop is its own native chat surface. It is not the browser dashboard and it
does not embed the TUI. Three parties, each authoritative for one thing:

- **Electron** owns the machine: process lifecycle, native filesystem/git/
  windows, install/update, and a narrow, typed capability bridge.
- **The renderer** owns the experience: navigation, presentation, and ephemeral
  interaction state.
- **The agent backend** owns the work: sessions, tools, model calls, streaming.

Keep the seams clean. The renderer never reaches for Node or Electron directly;
native power arrives through a deliberate capability, not a general escape hatch.
Agent behavior lives behind the gateway, never reimplemented in React. When a
change blurs a seam, that is the smell — fix the seam, don't widen it.

## Decide state by authority

The first question for any piece of state is *who is allowed to be right about
it*, not where it is convenient to store it. Put state with its authority:

- The **backend** is authoritative for anything another Hermes surface can also
  change. Treat the renderer's copy as a cache of that truth.
- **Electron** is authoritative for machine and runtime facts.
- The **renderer** owns only what is purely about this window's presentation.

From that, everything else follows: shared renderer state lives in small stores
owned by the feature that owns the concern; request-shaped server data that wants
invalidation lives in the query layer; short-lived interaction detail stays in
the component; hot coordination that must not paint stays in a ref. Reach for the
narrowest home that still lets the state be correct. A new global store is a
claim that many distant surfaces need it — earn that claim.

Persisted state must declare its scope in its own key: is this global, or does it
belong to a connection, a profile, a stored session, a project, or a window?
Getting the scope wrong is how one profile's setting bleeds into another.

## Identity is not incidental

Sessions have durable, runtime and lineage-root identities; pick the one the surface needs and
translate at the boundary. The user's message is durable at send (`prompt.submit` writes the session
and user rows first and the turn adopts that row, #111868). A new profile-keyed localStorage family
joins BOTH `migrateTilesForProfile` and `dropTilesForProfile`. A verifiably gone session
(`goneSessionVerdict` → `'draft'`) drops to a fresh draft that adopts its unsent text once, with no
toast, navigation or focus steal.

## Server truth is cached, not owned

Merge, don't clobber; paint optimistically and roll back visibly; a stale async result never
overwrites newer intent (generation counters, request tokens); only the foreground surface publishes
into the shared view; coalesce cosmetic updates but flush terminal transitions; keep reference
identity on no-ops.

## Switching context is a re-home, not a reboot

A connection/mode apply is a SOFT re-home (shell stays, gateway-bound stores wiped explicitly, then
reconnect); a runtime `HERMES_HOME` change is a HARD re-home (reload); a live profile swap merges
lists while background profiles keep streaming. After any swap the active socket, profile and
connection atoms must agree.

## Cross everything as an observable ladder

Every seam (versions, profiles, local/remote/cloud, older backends) resolves through ONE ordered
ladder per policy: precedence as data, a candidate trusted only after it is probed, a failed read
falls to the next rung but a failed authoritative write surfaces or rolls back, a missing capability
is not a transient failure, and retries are bounded and end in a recovery affordance. OAuth gateway
connections mint a fresh ticket on every dial (only a confirmed 401/403 means reauth); a connection
test exercises the leg you will actually use; `persist:` partition names avoid anything Electron
percent-escapes (`electron/oauth-partition.ts`).

## Guest content never opens anything by itself

Artifact iframes and the preview `<webview>` never drive the OS browser on their own
(GHSA-9f4c-93c8-jc8g): `setWindowOpenHandler` denies everything, the webview has no `allowpopups`,
and a guest `target="_blank"` link reaches `hermes:openExternal` only through the
`persist:hermes-preview` guest preload's trusted-click bridge, `http:`/`https:` only. Widening the
partition key, the trusted-click gate or the scheme set reopens the advisory.

## Compatibility without carrying the past forever

Desktop and its runtime update on separate clocks, so a change can meet an older
backend. Keep those users working: preserve the current feature, keep the
fallback narrow and tied to an identified older runtime, and cover it with a
test. A fallback that quietly degrades the feature it's meant to protect is worse
than the crash it replaced.

## Keep the waist narrow, grow at the edges

The [contribution rubric](../../CODING_STANDARDS.md#work-worth-accepting) governs here too. New capability should arrive at
the smallest surface that solves it: extend what exists, add a feature locally,
lean on an existing seam — before you invent a framework. The shell's internal
registries are composition seams, not a public plugin ABI; do not build a
universal extension system, a manifest, or a plugin adapter for a single
consumer. Design a shared contract only once more than one real consumer proves
its shape. "Plugin" means several unrelated things across Hermes — do not assume
one surface's extension model runs in another.

When the new capability is an **agent-callable** one — a tool that acts on this
renderer (open a pane, read the in-app browser, react to a message) — it is a
property of the SESSION's client, not of the backend host. Wire its
availability off the session source the app already sends on `session.create`
(`source: 'desktop'`), never off an env var on the backend process: that
process might be a remote or cloud gateway this app merely connected to. See
`tools/AGENTS.md`, "Surface capability is a property of the SESSION."

## Respect the person using it

Design and engineering meet at intent. The user's attention and context are
sacred:

- Never navigate, move focus, or open a surface because something *happened* in
  the background. Offer; don't hijack.
- The states around loading are distinct experiences — empty, loading,
  reconnecting, degraded/stale, and exhausted-recovery each deserve their own
  honest copy and their own way out.
- Keyboard ownership follows focus. The focused surface wins its keys; one
  cancel gesture does exactly one thing.
- Expensive, stateful surfaces (terminals, live tools) stay alive when hidden.
  Visibility is not lifecycle.

## Make it feel instant

Performance is a feature the user feels, especially in drag, resize, scroll,
typing, streaming, and terminals. The principles are timeless even as the code
changes: keep hot-path state local or narrowly derived; don't subscribe heavy
trees to per-frame updates; coalesce pointer work; avoid reading layout right
after writing style; and don't mount expensive content mid-gesture. Prove speed
against realistic content — a fast empty demo proves nothing about a long
transcript. If motion is masking latency, remove the motion, don't tune it.

## Testing as a habit of proof

Test the behavior that would actually break a user, not a snapshot of today's
data. Favor invariants over frozen values. Exercise the real path for anything
at a seam — resolver precedence and its failure rungs, identity and scope
boundaries, optimistic rollback and stale-response ordering, and both sides of a
local/remote adapter with its profile routing intact. Match how the suite is
actually run rather than inventing a command; when in doubt, read the scripts.

## The taste test before you hand off

- Does every piece of state live with its authority, at the narrowest scope?
- Would a background event ever steal the foreground or the user's focus?
- Does each resolver have one home, a validated ladder, and a bounded, recoverable
  end?
- Do local, remote, and profile routing still agree?
- Does async failure leave a usable UI and a way forward?
- Do hot interactions stay cheap under realistic load?
- Does the change pass the [`DESIGN.md`](./DESIGN.md) checklist and update all
  locales?

If any answer is "not sure," that's the part to go verify.

## Nous free tier: state is pulled, never latched in the renderer

`free_tier.status` / `free_tier.ack_notice` are the only source: the ready screen and the own-key
strip render the same `notice_pending` state, with no localStorage latch, and every entry point
opens the one sign-in dialog. Branch on `free_tier_row` / `free_tier`, never on provider display
names. Long form: `ENGINEERING.md`.
