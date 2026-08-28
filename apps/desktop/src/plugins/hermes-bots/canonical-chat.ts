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
import { host } from '@hermes/plugin-sdk'

import { $botMeta, botMetaKey, botOwner, persistBotMetaSnapshot } from './data'
import { backendTargetProfile, botConnectionRoute, botRosterMeta, botWorkspaceOwnerKey, requestForBot } from './routing'
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

interface CanonicalCreation {
  /** Null only between the holder literal and the assignment two statements
   *  later; a flight is published to the map already carrying its promise. */
  run: null | Promise<null | string>
}

// In-flight creations, keyed by bot name — double-clicking a row must not
// mint two canonical chats.
const canonicalCreations = new Map<string, CanonicalCreation>()

/** Upper bound for per-profile session.list scans (hide sweep, canonical-chat
 *  adoption, stored-session lookups). */
export const PROFILE_SESSION_LIST_LIMIT = 200

/** The one canonical title. (profile, CANONICAL_CHAT_TITLE) IS the bot's
 *  forever-chat identity — see the header above. */
const CANONICAL_CHAT_TITLE = 'Bot Chat'

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

/** True when a session summary IS the canonical registry row. root_title is
 *  the durable lineage-root title reported by exact-lookup gateways; plain
 *  title covers windowed listings. */
function isCanonicalBotChatHistory(history: CanonicalChatRow) {
  const rootTitle = String(history?.root_title || '').trim()
  const title = String(history?.title || '').trim()

  return rootTitle === CANONICAL_CHAT_TITLE || (!rootTitle && title === CANONICAL_CHAT_TITLE)
}

function botModeGatewayNeedsUpdate(error: unknown) {
  const message = String((error as RpcErrorLike)?.message || error || '')

  return /(?:method not found|no handler for|unknown method|unsupported rpc)/i.test(message)
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

  return rows.find(row => isCanonicalBotChatHistory(row)) || null
}

interface CreateCanonicalChatOptions {
  kickoff?: boolean
  openingStillCurrent?: (() => boolean) | null
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
 *  but never steals the workspace (#89834 family). */
export function createCanonicalChat(
  owner: RosterRow | string,
  { kickoff = false, openingStillCurrent = null }: CreateCanonicalChatOptions = {}
): Promise<null | string> {
  const { bot, name, key, route } = botOwner(owner)
  const inflight = canonicalCreations.get(key)

  if (inflight) {
    if (!openingStillCurrent) {
      return inflight.run!
    }

    return inflight.run!.then(async sid => {
      if (sid && openingStillCurrent() && typeof host.openSession === 'function') {
        await host.openSession(sid, {
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
      }

      return sid
    })
  }

  const flight: CanonicalCreation = {
    run: null
  }

  const canNavigate = () => !openingStillCurrent || openingStillCurrent()

  const run = (async () => {
    const existing = await findExistingCanonicalChat(owner)

    if (existing?.id) {
      if (typeof host.openSession === 'function' && canNavigate()) {
        // The exact-lookup gateway reports the compression-lineage tip as
        // resolved_id; open the tip, the registry row stays the identity.
        await openStoredBotChat(owner, existing.resolved_id || existing.id, existing)
      }

      return existing.id
    }

    const res = await requestForBot<{ session_id?: string; stored_session_id?: string }>(bot, 'session.create', {
      profile: backendTargetProfile(route, name),
      title: CANONICAL_CHAT_TITLE,
      // Always born hidden from the global sidebar — Bot Mode sessions are
      // plugin-owned. Core applies this via the generic `hidden` flag
      // (deferred as pending_hidden until the row exists); older gateways
      // ignore the unknown param and it stays visible.
      hidden: true,
      // Explicit contract (PR #97008): this session's runtime always follows
      // the member profile's CURRENT config. Resume must NOT restore the
      // stored model/provider pin from an old row — that left bot DMs stuck
      // on a stale/dead provider after a profile switch. Older gateways
      // ignore the unknown param; the server's exact-title backfill then
      // covers the legacy path.
      follow_profile_config: true
    })

    const sid = res?.stored_session_id
    const runtime = res?.session_id

    // session.create is intentionally lazy: its stored row does not exist until
    // the first prompt. Mounting `sid` immediately therefore emits a noisy REST
    // 404 ("Session not found"), and the turn-start auto-titler can win the race
    // against the deferred `title: 'Bot Chat'` — under name-identity that is an
    // identity outage: until the row is titled, the registry has no "Bot Chat"
    // entry, so a second click during the intro turn mints a duplicate.
    // session.title materializes the row now and records a user-authority title
    // before either the open or kickoff, closing both the 404 race and the
    // untitled window. Older gateways may not support the eager write; retain
    // the kickoff-and-retry fallback below.
    let titled = false

    if (runtime) {
      try {
        await requestForBot(bot, 'session.title', {
          session_id: runtime,
          title: CANONICAL_CHAT_TITLE
        })
        titled = true
      } catch (error) {
        // ADOPT-BEFORE-MINT: a title-uniqueness rejection is not an old
        // gateway — it means another writer took the canonical title between
        // our registry miss and this write (peer dm minting server-side, a
        // second machine, cross-connection sync). Falling through to the
        // compat path would prompt into OUR stray session and fork the
        // forever chat. Re-consult the registry and adopt the winner; the
        // stray lazy session holds zero messages and is simply abandoned
        // (the gateway prunes it).
        if (/already in use/i.test(String((error as RpcErrorLike)?.message || ''))) {
          const winner = await findExistingCanonicalChat(owner)

          if (winner?.id) {
            // Adopting the winner settles IDENTITY, which is always correct to
            // return. Navigating to it is not: this path can land a full
            // round-trip after the user clicked another bot, and every sibling
            // open here is staleness-probed for exactly that reason.
            if (typeof host.openSession === 'function' && canNavigate()) {
              await openStoredBotChat(owner, winner.resolved_id || winner.id, winner)
            }

            return winner.id
          }
        }
        /* compatibility fallback: prompt.submit will persist the lazy row */
      }
    }

    // Mount the session view FIRST, then send the kickoff — submitting into
    // an unmounted session left the intro reply invisible until reopen.
    let opened = false

    if (sid && typeof host.openSession === 'function' && canNavigate()) {
      try {
        await host.openSession(sid, {
          ...(route
            ? {
                route
              }
            : {}),
          profile: name,
          intent: 'main',
          keepAllProfilesScope: route ? true : false,
          ...(openingStillCurrent
            ? {
                workspaceMode: 'bots',
                workspaceOwnerKey: botWorkspaceOwnerKey(bot),
                tabTitle: CANONICAL_CHAT_TITLE
              }
            : {})
        })
        opened = true
      } catch {
        // The stored row may not exist until the kickoff persists it. Retry
        // after prompt.submit below instead of leaving the chat off-screen.
      }
    }

    if (runtime) {
      // Intro turn: only on genuine New Agent creation (`kickoff`), or as the
      // COMPAT persistence write when the eager title failed — an old gateway
      // prunes the zero-message lazy session, so without some first prompt
      // the chat never survives its own creation. A titled row needs neither:
      // the user speaks first.
      const submitIntro = kickoff || !titled

      if (submitIntro) {
        await new Promise(resolve => window.setTimeout(resolve, 400))

        try {
          await requestForBot(bot, 'prompt.submit', {
            session_id: runtime,
            text: kickoffText()
          })

          if (!opened && sid && typeof host.openSession === 'function' && canNavigate()) {
            await host.openSession(sid, {
              ...(route
                ? {
                    route
                  }
                : {}),
              profile: name,
              intent: 'main',
              keepAllProfilesScope: route ? true : false,
              ...(openingStillCurrent
                ? {
                    workspaceMode: 'bots',
                    workspaceOwnerKey: botWorkspaceOwnerKey(bot),
                    tabTitle: CANONICAL_CHAT_TITLE
                  }
                : {})
            })
          }
        } catch {
          // The chat already exists under the canonical title — the next click
          // finds it by name instead of making a second Bot Chat.
        }
      } else if (!opened && sid && typeof host.openSession === 'function' && canNavigate()) {
        // No intro turn: still finish mounting the chat when the first open
        // raced the (now titled) row.
        try {
          await host.openSession(sid, {
            ...(route
              ? {
                  route
                }
              : {}),
            profile: name,
            intent: 'main',
            keepAllProfilesScope: route ? true : false,
            ...(openingStillCurrent
              ? {
                  workspaceMode: 'bots',
                  workspaceOwnerKey: botWorkspaceOwnerKey(bot),
                  tabTitle: CANONICAL_CHAT_TITLE
                }
              : {})
          })
        } catch {
          /* row is titled and persistent — the next click opens it by name */
        }
      }
    }

    return sid || null
  })().finally(() => {
    if (canonicalCreations.get(key) === flight) {
      canonicalCreations.delete(key)
    }
  })

  flight.run = run
  canonicalCreations.set(key, flight)

  return run
}

/** Open the bot's ONE forever chat and return the opened registry id.
 *
 *  The whole resolution is one registry consultation ON THE BOT'S OWN
 *  SOURCE: the profile's session titled "Bot Chat" exists → open it
 *  (lineage tip); it doesn't → create it. No id pointer is read or written
 *  anywhere in this path — remote bots included. The owner route rides
 *  every RPC (requestForBot) and the open (openStoredBotChat), so a remote
 *  bot's chat opens without re-homing Desktop's chrome. */
export async function openBotCanonicalChat(
  owner: RosterRow | string,
  openingStillCurrent: (() => boolean) | null = null
): Promise<{ openedId: string; registryId: string } | null> {
  const existing = await findExistingCanonicalChat(owner)

  if (existing?.id && typeof host.openSession === 'function') {
    if (openingStillCurrent && !openingStillCurrent()) {
      return null
    }

    const openedId = existing.resolved_id || existing.id
    await openStoredBotChat(owner, openedId, existing)

    // Both identities matter downstream: the durable registry row names the
    // chat; the resolved lineage tip is what actually takes session focus.
    // Callers matching focus against only the registry id mistook every
    // compressed Bot Chat for a stale open (first click bounced to the home).
    return {
      registryId: String(existing.id),
      openedId: String(openedId)
    }
  }

  const created = await createCanonicalChat(owner, {
    openingStillCurrent
  })

  return created
    ? {
        registryId: String(created),
        openedId: String(created)
      }
    : null
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
