/**
 * The canonical Bot Chat: one bot, one forever-chat, resolved by exact title.
 *
 * Read the Bot Mode section of the repo root AGENTS.md before touching any of
 * this — the identity contract below is settled and has been regressed
 * repeatedly. The click-path orchestration that drives these lives in
 * plugin.tsx, which owns the workspace and group state a bot open competes
 * with.
 */

import * as sdk from '@hermes/plugin-sdk'
import {
  type AuthorizedRosterEntry,
  CANONICAL_AGENT_CHAT_TITLE,
  type CanonicalAgentChatCallbacks,
  type CanonicalAgentChatTarget,
  host,
  isAuthorizedCanonicalChatTarget,
  isCanonicalAgentChatRow,
  isTitleConflictError,
  resolveCanonicalAgentChat,
} from '@hermes/plugin-sdk'

import { $botMeta, botMetaKey, botOwner, persistBotMetaSnapshot } from './data'
import {
  aliasIdentityFor,
  backendTargetProfile,
  botConnectionRoute,
  botRosterMeta,
  botWorkspaceOwnerKey,
  requestForBot,
} from './routing'
import type { RpcErrorLike } from './routing'
import { getPluginCtx } from './shared'
import type { BotMeta, CanonicalSession, RosterRow } from './types'

// ── canonical bot chat ───────────────────────────────────────────────────────
// Each bot has ONE forever chat, identified by NAME, never by pointer: the
// session titled exactly "Bot Chat" on that bot's profile. The core
// UNIQUE(title) index makes (profile, "Bot Chat") an exact registry, so every
// open consults that registry directly — there is nothing to verify, re-pin,
// grandfather, or recover. Stored-id pins (ui_meta['hermes-bots'].chat) were
// the previous identity and are REMOVED: every lost-chat incident traced to a
// dangled or stolen pointer that later guards then welded in. Legacy
// ui_meta.chat keys are simply ignored.
//
// Concurrency protection (double-clicking a row must not mint two canonical
// chats) is owned by the shared `resolveCanonicalAgentChat` core, keyed by
// (connectionId, targetProfile) — this module no longer keeps its own
// in-flight-creation map.

/** Upper bound for per-profile session.list scans (hide sweep, canonical-chat
 *  adoption, stored-session lookups). */
export const PROFILE_SESSION_LIST_LIMIT = 200

/** The one canonical title. (profile, CANONICAL_CHAT_TITLE) IS the bot's
 *  forever-chat identity — see the header above. Exported for the roster
 *  click path's tile-staleness probe (hermes-agent#90102), which must
 *  recognize canonical-titled tabs without restating the literal. Aliases
 *  the shared identity contract's constant (`@hermes/plugin-sdk`'s
 *  `CANONICAL_AGENT_CHAT_TITLE`) rather than restating the literal — Bot
 *  Mode and the SDK's `host.openCanonicalAgentChat` must never disagree on
 *  the title they resolve by. */
export const CANONICAL_CHAT_TITLE = CANONICAL_AGENT_CHAT_TITLE

/** A `session.list` row as the registry lookup reads it. CanonicalSession
 *  models the roster's `canonical_session` field, which carries no
 *  `message_count` — the listing row does. `readonly` because the count is
 *  read through an aliased guard, which TS only narrows for immutable
 *  properties. */
interface CanonicalChatRow extends CanonicalSession {
  readonly message_count?: number
}

/** Is the chat on screen the given bot's forever-chat?
 *
 *  Identity comes off the roster's `canonical_session`, resolved server-side
 *  by title, and matches EITHER the durable registry row or the
 *  compression-lineage tip — a compacted Bot Chat is on screen under its tip
 *  id while the registry still names it by the root.
 *
 *  Takes the STORED id. The runtime id belongs to a different id space and
 *  matches neither, which is how the `/new` guard that calls this went dead. */
export function isCanonicalChatOnScreen(
  bot: null | RosterRow | undefined,
  storedSessionId: null | string | undefined
): boolean {
  const canonical = bot?.canonical_session

  if (!storedSessionId || !canonical) {
    return false
  }

  return [canonical.id, canonical.resolved_id].filter(Boolean).map(String).includes(String(storedSessionId))
}

/** Builds the complete source-qualified target the shared resolver requires
 *  from a Bot Mode owner. Never collapses to a bare `{connectionId, profile}`
 *  — `targetProfile` (backendTargetProfile) is carried explicitly and can
 *  diverge from the displayed `profile` for an aliased or remote row. */
function botOwnerToTarget(owner: RosterRow | string): { target: CanonicalAgentChatTarget; bot: RosterRow; route: ReturnType<typeof botOwner>['route'] } {
  const { bot, name, route } = botOwner(owner)

  return {
    target: {
      connectionId: route?.connectionId ?? null,
      profile: name,
      targetProfile: backendTargetProfile(route, name),
    },
    bot,
    route,
  }
}

async function openStoredBotChat(
  owner: RosterRow | string,
  storedId: string,
  summary: CanonicalChatRow
): Promise<string> {
  if (!storedId || typeof host.openSession !== 'function') {
    throw new Error('This Hermes Desktop version cannot open stored sessions')
  }

  const { bot, name, route } = botOwner(owner)
  const ownerKey = botWorkspaceOwnerKey(bot)
  const hasAuthoritativeCount = typeof summary?.message_count === 'number' && Number.isFinite(summary.message_count)
  const expectHistory = hasAuthoritativeCount ? summary.message_count > 0 : true

  // Current SDKs export the Bot-specific budget. The fallback preserves
  // compatibility with older hosts and isolated plugin test harnesses.
  const hydrationTimeoutMs =
    typeof sdk !== 'undefined' && Number.isFinite(sdk.BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS)
      ? sdk.BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS
      : 60_000

  // A profile backend that just woke up can lose the hydration-timeout race
  // even though the session is fine (hermes-agent#89617) — clicking Retry
  // succeeds because the backend is warm by then. retryHydrationTimeoutOnce
  // asks the SDK layer to retry that same wait internally, BEFORE it arms the
  // core stranded-session overlay: a plugin-side retry can't do this because
  // only host.openSession sees the resume-exhausted latch that overlay reads.
  //
  // forceResume: an explicit bot switch must never trust a cached transcript.
  // The SDK's surface-health check passes whenever ANY non-empty transcript is
  // painted, including a stale snapshot the session-states cache kept from the
  // previous time this bot was open — which left the pane showing old messages
  // until an app restart (hermes-agent#93604). A resume is cheap and
  // idempotent, so on this explicit user navigation we always request one.
  await host.openSession(storedId, {
    ...(route
      ? {
          route
        }
      : {}),
    profile: name,
    // Same intent a session row click uses. `tab` stacked a fresh tile every
    // time focusOpenSession missed, so bot chats piled up beside each other and
    // beside the untouched "New session" draft, which then kept focus —
    // clicking a bot appeared to do nothing. `in-place` still fronts an
    // already-open tile first, so Bot tabs survive owner lifecycles (#a81854a2,
    // the reason this stopped being `main`); it just loads into main instead of
    // minting a second tab when there is nothing to front.
    intent: 'in-place',
    awaitHydration: true,
    expectHistory,
    forceResume: true,
    hydrationTimeoutMs,
    keepAllProfilesScope: true,
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey,
    retryHydrationTimeoutOnce: true,
    tabTitle: CANONICAL_CHAT_TITLE
  })

  return storedId
}

/** True when a session summary IS the canonical registry row. Delegates to
 *  the shared identity contract's predicate (see `@hermes/plugin-sdk`'s
 *  `isCanonicalAgentChatRow`) rather than re-implementing the same
 *  root_title/title match — Bot Mode and the SDK's
 *  `host.openCanonicalAgentChat` must agree on what counts as canonical. */
function isCanonicalBotChatHistory(history: CanonicalChatRow) {
  return isCanonicalAgentChatRow(history)
}

function botModeGatewayNeedsUpdate(error: unknown) {
  const message = String((error as RpcErrorLike)?.message || error || '')

  return /(?:method not found|no handler for|unknown method|unsupported rpc)/i.test(message)
}

/** Fetches the CURRENT live roster and reports whether `target` is present,
 *  full-descriptor, via the shared `isAuthorizedCanonicalChatTarget` — the
 *  ONE check both assertAuthorizedTarget (below) and waitForRosterAdmission (above) share. Never throws on a roster-fetch failure — callers that need
 *  fail-closed-on-fetch-failure semantics (assertAuthorizedTarget) check the
 *  fetch outcome themselves; a poll loop treats a fetch failure as "not yet
 *  admitted" and retries within its own bound instead of aborting on one
 *  transient hiccup. */
async function isTargetInLiveRoster(target: CanonicalAgentChatTarget): Promise<boolean> {
  if (typeof host.agents !== 'function') {
    return false
  }

  let rosterResult: { agents?: Array<{ connectionId?: null | string; profile: string; targetProfile?: string }> }

  try {
    rosterResult = await host.agents()
  } catch {
    return false
  }

  // Project each backend row's DISPLAYED identity through the established
  // alias index before the strict full-descriptor check — host.agents()
  // reports a remote row's own backend profile, never a Desktop-local
  // alias's configured display name (see routing.ts's alias-identity
  // section). Without this projection, a legitimately routed alias target
  // (whose `profile` is the alias name) would never full-descriptor-match
  // the raw backend row and would incorrectly fail closed.
  const projectedAgents: AuthorizedRosterEntry[] = (rosterResult?.agents ?? []).map(entry => {
    // No fallback to `entry.profile`: a row that omits its own backend
    // targetProfile cannot vouch for any target — pass an explicit blank
    // through so isAuthorizedCanonicalChatTarget's own required-field
    // check (never a caller-side substitution) is what rejects it
    // (Architect corrective, 2026-09-02, sixth pass). The alias-identity
    // lookup below is a SEPARATE, non-authorizing use of targetProfile
    // (resolving a display alias), so it may still read the raw value —
    // only the AUTHORIZATION entry itself must never substitute profile.
    const entryTargetProfile = (entry.targetProfile ?? '').trim()

    const alias = aliasIdentityFor({
      connectionId: entry.connectionId ?? null,
      name: entry.profile,
      remoteSource: Boolean(entry.connectionId),
      sourceScoped: Boolean(entry.connectionId),
      targetProfile: entry.targetProfile ?? entry.profile,
    } as Partial<RosterRow>)

    return {
      connectionId: entry.connectionId ?? null,
      profile: alias ? alias.name : entry.profile,
      targetProfile: entryTargetProfile,
    }
  })

  return isAuthorizedCanonicalChatTarget(target, { agents: projectedAgents })
}

async function assertAuthorizedTarget(target: CanonicalAgentChatTarget): Promise<void> {
  // Every caller — including one joining an already-in-flight creation for
  // the identical target — must read the CURRENT authorized roster before
  // any lookup/create/title/open reaches the shared resolver (Architect
  // corrective, 2026-09-02, fourth pass). A prior caller's roster read,
  // however recent, is not a current admission for THIS caller: it proves
  // nothing about what the live roster says right now, and short-circuiting
  // on it was rejected as a design that trades a real authorization read
  // for an unverifiable inference. Deduplicating concurrent creation of the
  // SAME target remains entirely `resolveCanonicalAgentChat`'s own
  // `inflightCreations` singleflight — a joiner still performs its own full
  // roster check here, then joins the resolver's flight (or, if that flight
  // has already settled by the time this check completes, its own
  // `callbacks.lookup` inside the resolver adopts the now-existing
  // canonical row rather than minting a second one — the same
  // adopt-before-mint protection that already defends every other
  // concurrent-open path in this file).
  if (typeof host.agents !== 'function') {
    throw new Error(`${target.profile} could not be authorized — this Desktop build cannot enumerate the agent roster`)
  }

  let rosterFetchFailed = false
  let admitted: boolean

  try {
    admitted = await isTargetInLiveRoster(target)
  } catch {
    // isTargetInLiveRoster never throws itself, but keep the fail-closed
    // shape explicit in case that changes.
    rosterFetchFailed = true
    admitted = false
  }

  if (rosterFetchFailed) {
    throw new Error(`Could not confirm ${target.profile} against the current authorized roster — not proceeding`)
  }

  if (!admitted) {
    throw new Error(`${target.profile} is not present in the current authorized agent roster — refusing to open`)
  }
}

/** Polls the SAME live `host.agents()` roster `isTargetInLiveRoster` reads,
 *  for a brand-new profile whose roster enumeration has not caught up with a
 *  `profiles.create` this exact window just issued (Architect corrective,
 *  2026-09-02, second pass): the earlier `freshlyCreatedRosterEntry`
 *  caller-supplied bypass was rejected because a client-side object is never
 *  itself authorization. This function asserts nothing on its own — it only
 *  waits for the ACTUAL roster (the one authority) to admit the target, then
 *  returns normally so the caller proceeds through the ordinary
 *  `assertAuthorizedTarget` → resolver path. If the roster still hasn't
 *  caught up after the bound, it throws — the caller must not open a chat
 *  for a target the live roster has never confirmed. */
async function waitForRosterAdmission(
  target: CanonicalAgentChatTarget,
  { attempts = 10, delayMs = 300 }: { attempts?: number; delayMs?: number } = {}
): Promise<void> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (await isTargetInLiveRoster(target)) {
      return
    }

    if (attempt < attempts - 1) {
      await new Promise(resolve => window.setTimeout(resolve, delayMs))
    }
  }

  throw new Error(
    `${target.profile} did not appear in the current authorized agent roster in time — refusing to open a chat for an unconfirmed profile`
  )
}

export function notifyBotOpenFailure(error: unknown, bot: RosterRow, fallbackMessage: string) {
  if (botModeGatewayNeedsUpdate(error)) {
    const gateway = bot.connectionLabel || bot.connectionId || 'this gateway'
    host.notify?.({
      kind: 'error',
      title: 'Update this gateway to use Bot Mode',
      message: `Update ${gateway}, then try again.`
    })

    return
  }

  host.notifyError?.(error, fallbackMessage)
}

/** THE identity lookup: the profile's session titled exactly "Bot Chat",
 *  consulted on the bot's OWN source. The core UNIQUE title index guarantees
 *  at most ONE such row per profile db — Profile → Named Session is an exact
 *  registry, so consult it exactly: `title` asks the gateway for an indexed
 *  WHERE title = ? lookup (window-free; a busy profile can push the
 *  forever-chat past any recency window). include_hidden is required
 *  (canonical chats are always hidden). Remote bots route via requestForBot
 *  on the immutable captured owner — activation is a UI concern and never
 *  authorizes this RPC. */
async function findExistingCanonicalChat(owner: RosterRow | string): Promise<CanonicalChatRow | null> {
  const { bot, name, route } = botOwner(owner)
  // FAIL CLOSED. A failed registry lookup MUST NOT read as "no Bot Chat
  // exists" — that is the one remaining way to fork a bot's forever chat.
  // The failure lives exactly in the post-update window: the desktop
  // restarts every profile backend, the first bot click races the warm-up,
  // the lookup RPC fails transiently, and a swallowed error here sent
  // createCanonicalChat() straight to session.create — minting a fresh
  // "Bot Chat" while the real one (data intact, hidden) still held the
  // canonical title. Users read that as "my bot lost all context after the
  // update". Cross-connection lookups fail MORE often (network), so this
  // matters doubly for remote bots. Both open paths catch and toast "try
  // again", which is the correct outcome: retry, never mint.
  let res: { sessions?: CanonicalChatRow[] }

  try {
    res = await requestForBot<{ sessions?: CanonicalChatRow[] }>(bot, 'session.list', {
      profile: backendTargetProfile(route, name),
      title: CANONICAL_CHAT_TITLE,
      limit: PROFILE_SESSION_LIST_LIMIT,
      include_hidden: true
    })
  } catch (error) {
    // Plugin tests and host bridges can return Error-like values from another
    // JS realm, where `instanceof Error` is false. Preserve the provider/RPC
    // message so update-required classification and diagnostics still work.
    const message = typeof (error as RpcErrorLike)?.message === 'string' ? (error as { message: string }).message : ''
    const detail = message ? ` (${message})` : ''
    throw new Error(`Could not check ${name}'s Bot Chat registry${detail} — not starting a new chat`)
  }

  const rows = res?.sessions ?? []
  const match = rows.find(row => isCanonicalBotChatHistory(row))

  if (match) {
    return match
  }

  // A zero-row result is NOT the same as a thrown error, but it is just as
  // capable of forking the forever chat: a profile backend mid-restart can
  // answer `session.list` successfully with an empty list rather than
  // failing it, and `|| null` used to read that identically to "this bot
  // never had a chat" (#98383). The roster's own `canonical_session` is the
  // last positive confirmation this profile HAD one — when that exists,
  // an empty lookup is unconfirmed absence, not confirmed absence, so fail
  // closed the same way a thrown RPC error already does instead of minting.
  if (bot?.canonical_session?.id) {
    throw new Error(`Could not confirm ${name}'s Bot Chat registry — not starting a new chat`)
  }

  return null
}

interface CreateCanonicalChatOptions {
  kickoff?: boolean
  openingStillCurrent?: (() => boolean) | null
  /** Genuine New Agent creation (Architect corrective, 2026-09-02, second
   *  pass): rather than trusting a caller-supplied roster entry (rejected as
   *  a client-side bypass — a local object is never itself authorization),
   *  this flag makes `createCanonicalChat` poll the SAME live
   *  `host.agents()` roster via `waitForRosterAdmission` until the ACTUAL
   *  roster admits the target, before proceeding through the ordinary
   *  authorization → resolver path. Only `create-dialog.tsx`'s New Agent
   *  flow, immediately after its own successful `profiles.create`, sets
   *  this — it does not weaken authorization for any other caller. */
  newAgentBirth?: boolean
  /** Test-only override of the poll bound `waitForRosterAdmission` uses when
   *  `newAgentBirth` is set. Production callers should never need this —
   *  defaults are tuned for a real roster refresh interval. */
  rosterAdmissionPoll?: { attempts?: number; delayMs?: number }
}

/** The self-introduction a brand-new bot is born with (#91827).
 *
 *  Localized because it is the FIRST line of the forever-chat: shipped in
 *  English it opened every non-English user's relationship with their bot in a
 *  foreign language, and the bot's reply followed the prompt's language, so one
 *  hardcoded string biased the whole conversation.
 *
 *  It is still submitted on the user's side, which is the half of #91827 this
 *  cannot close from here: `prompt.submit` IS the user-turn API, so an
 *  unattributed birth needs the lazy/silent path that issue proposes — a
 *  gateway contract change, not a rename. The intro itself is deliberate and
 *  documented in AGENTS.md ("kicked off with the bot's intro"); what this
 *  narrows is who has to read it in English. */
function kickoffText(): string {
  return getPluginCtx()?.i18n?.t('bot.kickoff') ?? 'Hey, tell me about yourself!'
}

/** Bot Mode's callback set for the shared resolver. Every RPC/navigation
 *  behavior documented on `createCanonicalChat`/`openBotCanonicalChat` below
 *  is preserved here verbatim — this module supplies HOW Bot Mode performs
 *  each step; `resolveCanonicalAgentChat` owns WHEN/WHETHER each step runs
 *  (lookup, create, title, conflict-adopt, fail-closed). */
function botModeCallbacks(owner: RosterRow | string, kickoff: boolean): CanonicalAgentChatCallbacks {
  const { bot, name, route } = botOwner(owner)
  // Set true by titleSession on a successful eager write. openFresh reads
  // it to decide whether the compat persistence prompt (submitIntro when
  // !titled) is required — mirrors the original single-function contract
  // exactly, just split across the two callbacks the resolver now owns.
  let titled = false

  return {
    lookup: async () => findExistingCanonicalChat(owner),

    create: async () => {
      titled = false

      const res = await requestForBot<{ session_id?: string; stored_session_id?: string }>(bot, 'session.create', {
        profile: backendTargetProfile(route, name),
        title: CANONICAL_CHAT_TITLE,
        // Always born hidden from the global sidebar — Bot Mode sessions are
        // plugin-owned. Core applies this via the generic `hidden` flag
        // (deferred as pending_hidden until the row exists); older gateways
        // ignore the unknown param and it stays visible.
        hidden: true,
        // Explicit contract (PR #97008): this session's runtime always
        // follows the member profile's CURRENT config. Resume must NOT
        // restore the stored model/provider pin from an old row — that left
        // bot DMs stuck on a stale/dead provider after a profile switch.
        // Older gateways ignore the unknown param; the server's exact-title
        // backfill then covers the legacy path.
        follow_profile_config: true
      })

      return { runtimeId: res?.session_id ?? null, storedId: res?.stored_session_id ?? null }
    },

    titleSession: async (_target, runtimeId) => {
      await requestForBot(bot, 'session.title', { session_id: runtimeId, title: CANONICAL_CHAT_TITLE })
      titled = true
    },

    openExisting: async (_target, openedId, row, canNavigateNow) => {
      if (canNavigateNow && typeof host.openSession === 'function') {
        await openStoredBotChat(owner, openedId, row as CanonicalChatRow)
      }
    },

    openFresh: async (_target, storedId, canNavigateNow) => {
      // Mount the session view FIRST, then send the kickoff — submitting
      // into an unmounted session left the intro reply invisible until
      // reopen. The workspace fields ride every open unconditionally: they
      // say this row is a bot's chat, true of a freshly minted one no
      // matter who asked for it.
      const openFreshCanonical = () =>
        host.openSession!(storedId, {
          ...(route
            ? {
                route
              }
            : {}),
          profile: name,
          intent: 'main',
          keepAllProfilesScope: route ? true : false,
          workspaceMode: 'bots',
          workspaceOwnerKey: botWorkspaceOwnerKey(bot),
          tabTitle: CANONICAL_CHAT_TITLE
        })

      let opened = false

      if (canNavigateNow && typeof host.openSession === 'function') {
        try {
          await openFreshCanonical()
          opened = true
        } catch {
          // The stored row may not exist until the kickoff persists it.
          // Retry after prompt.submit below instead of leaving the chat
          // off-screen.
        }
      }

      // Intro turn: on genuine New Agent creation (`kickoff`), OR as the
      // COMPAT persistence write when the eager title failed (`!titled`) —
      // an old gateway prunes the zero-message lazy session, so without
      // some first prompt the chat never survives its own creation. A
      // titled row needs neither: the user speaks first.
      const submitIntro = kickoff || !titled

      if (submitIntro) {
        await new Promise(resolve => window.setTimeout(resolve, 400))

        try {
          await requestForBot(bot, 'prompt.submit', {
            session_id: storedId,
            text: kickoffText()
          })

          if (!opened && canNavigateNow && typeof host.openSession === 'function') {
            await openFreshCanonical()
          }
        } catch {
          // The chat already exists under the canonical title — the next
          // click finds it by name instead of making a second Bot Chat.
        }
      } else if (!opened && canNavigateNow && typeof host.openSession === 'function') {
        // No intro turn: still finish mounting the chat when the first open
        // raced the (now titled) row.
        try {
          await openFreshCanonical()
        } catch {
          /* row is titled and persistent — the next click opens it by name */
        }
      }
    },
  }
}

/** Create the bot's ONE forever chat: a real session titled "Bot Chat".
 *  Adopts the existing "Bot Chat" row instead of creating when the profile
 *  already has one — minting while a "Bot Chat" row exists is always wrong
 *  twice over: it forks the forever-chat AND the new row can never take the
 *  (already held) canonical title. Creates on the bot's own source via
 *  requestForBot.
 *
 *  `kickoff` (New Agent creation ONLY): submit the self-introduction prompt
 *  so a brand-new bot greets its owner once. Every other caller — the bot
 *  row's click-path canonical resolution above all — must NOT pass it: a
 *  resolution miss (retitled row, hidden-listing gap, post-update skew)
 *  re-mints the session, and re-firing the intro there burned a model turn
 *  and stamped a user-attributed "Hey, tell me about yourself!" into the
 *  chat on every click (ScottFive report). The kickoff's original session-
 *  persistence job is done by the eager session.title write below on modern
 *  gateways; older gateways that reject the eager write keep a narrow
 *  compat kickoff, else the pruner reaps the empty lazy session and the
 *  chat never survives its own creation.
 *
 *  `openingStillCurrent` (click-path opens): a staleness probe consulted
 *  before every navigation — when the user has already moved on (opened a
 *  group, clicked another bot), the create still completes registry-side
 *  but never steals the workspace (#89834 family).
 *
 *  Delegates the actual resolution (lookup, creation-flight/concurrency
 *  protection, title materialization, title-conflict recovery, fail-closed
 *  behavior) to the shared `resolveCanonicalAgentChat` — the one identity-
 *  resolution flow shared with the SDK's `host.openCanonicalAgentChat`.
 *  This function supplies only Bot Mode's RPC/navigation behavior via
 *  `botModeCallbacks`, and the legacy-gateway untitled-compat opt-in below,
 *  which is Bot Mode's documented fallback (never authorized generically). */
export function createCanonicalChat(
  owner: RosterRow | string,
  { kickoff = false, openingStillCurrent = null, newAgentBirth = false, rosterAdmissionPoll }: CreateCanonicalChatOptions = {}
): Promise<null | string> {
  const { target } = botOwnerToTarget(owner)

  return (newAgentBirth ? waitForRosterAdmission(target, rosterAdmissionPoll) : Promise.resolve())
    .then(() => assertAuthorizedTarget(target))
    .then(() =>
      resolveCanonicalAgentChat(target, botModeCallbacks(owner, kickoff), {
        openingStillCurrent,
        isTitleConflict: isTitleConflictError,
        // Bot Mode's documented compat path applies ONLY to the recognized
        // legacy-gateway case — an eager session.title rejection that means
        // the write is simply UNSUPPORTED by this gateway (matched the same
        // way notifyBotOpenFailure classifies it). Any other non-conflict
        // failure (network, authorization, routing, ownership, unknown RPC
        // errors) now falls through and fails closed, per the Architect
        // corrective: the compat path is not a general swallow.
        allowUntitledCompat: (_target, error) => botModeGatewayNeedsUpdate(error),
      })
    )
    .then(result => result?.registryId ?? null)
}

/** Open the bot's ONE forever chat and return the opened registry id.
 *
 *  The whole resolution is one registry consultation ON THE BOT'S OWN
 *  SOURCE: the profile's session titled "Bot Chat" exists → open it
 *  (lineage tip); it doesn't → create it. No id pointer is read or written
 *  anywhere in this path — remote bots included. The owner route rides
 *  every RPC (requestForBot) and the open (openStoredBotChat), so a remote
 *  bot's chat opens without re-homing Desktop's chrome.
 *
 *  Delegates to the shared `resolveCanonicalAgentChat` via `createCanonicalChat`
 *  (which itself delegates) — one implementation, shared with the SDK. */
export async function openBotCanonicalChat(
  owner: RosterRow | string,
  openingStillCurrent: (() => boolean) | null = null
): Promise<{ openedId: string; registryId: string } | null> {
  const { target } = botOwnerToTarget(owner)

  await assertAuthorizedTarget(target)

  const result = await resolveCanonicalAgentChat(target, botModeCallbacks(owner, false), {
    openingStillCurrent,
    isTitleConflict: isTitleConflictError,
    allowUntitledCompat: (_target, error) => botModeGatewayNeedsUpdate(error),
  })

  return result ? { openedId: result.openedId, registryId: result.registryId } : null
}

export async function prepareBotSource(bot: RosterRow) {
  if (!bot.sourceScoped) {
    return
  }

  // Cross-connection RPCs ride the immutable captured route (requestForBot →
  // host.requestProfile) — Desktop's active connection does not move, and
  // activation is a UI concern that never authorizes the calls. All this
  // gate does is refuse when the desktop predates routed profile requests.
  const route = botConnectionRoute(bot)

  if (route && typeof host.requestProfile !== 'function') {
    throw new Error(
      getPluginCtx()?.i18n?.t('bot.remoteConnectionsUnsupported') ??
        'Update Hermes Desktop to chat with bots on other connections.'
    )
  }

  if (!route && typeof host.ensureAgent === 'function') {
    // Source-annotated row on the ACTIVE connection (no captured route):
    // legacy activation path, unchanged. An absent connectionId is fine —
    // ensureGatewayAgent normalizes it with `(connectionId ?? '').trim() || null`.
    await host.ensureAgent(bot.connectionId, bot.name)
  }
}

export async function ensureBotMetadata(bot: RosterRow): Promise<BotMeta> {
  if (!bot?.sourceScoped) {
    return botRosterMeta(bot, $botMeta.get()) || {}
  }

  const route = botConnectionRoute(bot)
  const backendProfile = backendTargetProfile(route, bot.name)

  const result = await requestForBot<{ profiles?: Array<Pick<RosterRow, 'name' | 'ui_meta'>> }>(
    bot,
    'profiles.list',
    {}
  )

  const row = (result?.profiles || []).find(profile => profile?.name === backendProfile)
  const server = row?.ui_meta?.['hermes-bots']

  if (server && typeof server === 'object') {
    const key = botMetaKey(bot)
    $botMeta.set({
      ...$botMeta.get(),
      [key]: {
        ...($botMeta.get()[key] || {}),
        ...server
      }
    })
    persistBotMetaSnapshot($botMeta.get(), true)
  }

  return botRosterMeta(bot, $botMeta.get()) || {}
}
