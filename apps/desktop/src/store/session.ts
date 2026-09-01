import type { ConnectionState } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { lastVisibleMessageIsUser } from '@/app/chat/thread-loading'
import type { ContextSuggestion } from '@/app/types'
import type { HermesConnection } from '@/global'
import type { ChatMessage } from '@/lib/chat-messages'
import {
  activeConnectionScopeSuffix,
  connectionScopeSuffix,
  rescopeConnectionScopedStores
} from '@/lib/connection-scoped'
import { persistBoolean, persistString, readJson, storedBoolean, storedString, writeJson } from '@/lib/storage'
import { syncCronModelImpactConnection } from '@/store/cron-model-impact-scope'
import type { SessionInfo, UsageStats } from '@/types/hermes'

import type { SessionOwnerRoute, SessionOwnerScope } from './session-request-router'
import { clearUnreadOnOpen } from './session-unread-remote'

type Updater<T> = T | ((current: T) => T)
export type ComposerModelSource = '' | 'default' | 'manual'

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

// The composer's model/effort/fast is sticky UI state, NOT the profile default
// (that lives in Settings → Model). Persisting it in localStorage makes a pick
// follow across Cmd+N and app restarts instead of snapping back to the default.
// Model/provider/source are scoped to the remote (connection, profile) owner so
// a provider authenticated on one profile cannot contaminate another profile's
// session.create. Local/single-backend users retain the historical bare keys.
const COMPOSER_MODEL_KEY = 'hermes.desktop.composer.model'
const COMPOSER_PROVIDER_KEY = 'hermes.desktop.composer.provider'
const COMPOSER_MODEL_SOURCE_KEY = 'hermes.desktop.composer.model-source'
const COMPOSER_EFFORT_KEY = 'hermes.desktop.composer.reasoning-effort'
const COMPOSER_FAST_KEY = 'hermes.desktop.composer.fast'

// Unlike presentation-oriented $connection, this scope is published from the
// gateway activation coordinate before profile-change effects can reseed the
// composer. null means the exact owner is temporarily unknown: values may still
// paint, but must not be written through the previous backend's storage key.
let composerSelectionScope: string | null = ''

function composerScopeForConnection(connection: HermesConnection | null): string | null {
  if (!connection) {
    return null
  }

  // Electron may infer the sole `local` registry id onto the ordinary primary
  // descriptor. That remains the legacy single-backend path: only an explicit
  // registry-scoped route earns a new namespace.
  if (connection.mode !== 'remote' && !connection.registryScoped) {
    return ''
  }

  if (connection.connectionId) {
    return `.registry.${encodeURIComponent(connection.connectionId)}.${encodeURIComponent(connection.profile || 'default')}`
  }

  return connectionScopeSuffix(connection)
}

function composerSelectionKey(base: string): string | null {
  return composerSelectionScope === null ? null : `${base}${composerSelectionScope}`
}

function storedComposerString(base: string): string | null {
  const key = composerSelectionKey(base)

  return key === null ? null : storedString(key)
}

// The last chat the user had open, so a relaunch lands back on it instead of an
// empty new-chat. Stored (not runtime) id — the route is keyed by stored id.
//
// Scoped per profile with an explicit namespace (`.profile.<encoded>`) and
// encodeURIComponent so a profile name carrying `/` or other reserved chars
// cannot collide or leak across keys. Legacy global (unsuffixed) keys are
// discarded on first read to prevent cross-profile bleed — ownership of the old
// global values is unknowable, and guessing the owning profile is exactly the
// cross-profile corruption this storage boundary prevents (#67709).
const LAST_SESSION_KEY = 'hermes.desktop.lastSessionId'
const LAST_ROUTE_KEY = 'hermes.desktop.lastRoute'

function profileNavigationKey(base: string, profile: string): string {
  const key = profile.trim() || 'default'

  // Also carries the CONNECTION scope: the same profile name on a different
  // gateway is a different backend with its own sessions, and windows on
  // different gateways share this localStorage area — restoring one
  // gateway's remembered session under another navigates to a session that
  // backend has never seen (#77318).
  return `${base}.profile.${encodeURIComponent(key)}${activeConnectionScopeSuffix()}`
}

// Discard legacy global keys once per tick. A module-level flag avoids
// redundant synchronous localStorage reads on every get/set call within
// the same synchronous block. The flag resets on cross-window `storage`
// events, which are the only way another window can recontaminate between
// ticks.
let legacyDiscardNeeded = true

if (typeof window !== 'undefined') {
  window.addEventListener('storage', e => {
    if (e.key === LAST_SESSION_KEY || e.key === LAST_ROUTE_KEY) {
      legacyDiscardNeeded = true
    }
  })
}

function discardLegacyRememberedNavigation(): void {
  if (!legacyDiscardNeeded) {
    return
  }

  legacyDiscardNeeded = false

  // Ownership of the old global values is unknowable. Never migrate them into
  // a profile: guessing is exactly the cross-profile corruption this storage
  // boundary prevents.
  if (storedString(LAST_SESSION_KEY) !== null) {
    persistString(LAST_SESSION_KEY, null)
  }

  if (storedString(LAST_ROUTE_KEY) !== null) {
    persistString(LAST_ROUTE_KEY, null)
  }
}

/** @internal Reset the legacy-discard flag for tests. */
export function _resetLegacyDiscardForTests(): void {
  legacyDiscardNeeded = true
}

export function getRememberedSessionId(profile: string): null | string {
  discardLegacyRememberedNavigation()

  return storedString(profileNavigationKey(LAST_SESSION_KEY, profile))
}

export function setRememberedSessionId(id: null | string, profile: string): void {
  discardLegacyRememberedNavigation()
  persistString(profileNavigationKey(LAST_SESSION_KEY, profile), id)
}

export function sessionBelongsToProfile(
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>[],
  storedSessionId: string,
  profile: string
): boolean {
  const key = profile.trim() || 'default'

  return sessions.some(session => {
    const owner = (session.profile ?? '').trim() || 'default'

    return owner === key && sessionMatchesStoredId(session, storedSessionId)
  })
}

/**
 * The profile that owns a session, from SYNC known sources only: the session
 * row (the cross-profile aggregator tags each row) then the owner hint recorded
 * at open time. Returns undefined when neither knows — the caller must resolve
 * it (cross-profile probe) rather than fall back to whatever is active, because
 * "active" is presentation state and never a routing authority. Hidden sessions
 * (Bot Mode's canonical "Bot Chat") never appear in the row list, so the hint is
 * often the only sync source.
 */
export function knownSessionProfile(sessions: readonly SessionInfo[], sessionId: null | string): string | undefined {
  const owner = knownSessionOwner(sessions, sessionId)

  return typeof owner === 'string' ? owner : (owner?.targetProfile ?? owner?.profile)?.trim() || undefined
}

/**
 * The complete known owner of a session: the EXACT route when the row is
 * connection-tagged (an optimistic row from a routed create, a foreign
 * registry row from the unified-list splice, or a tag mergeSessionPage carried
 * across a refresh), else the open-time / create-time owner hint when it
 * agrees with the row, else the bare profile. Session-scoped RPC callers must
 * use this instead of `knownSessionProfile`: two sources can expose the same
 * profile name, so returning only that name silently collapses the route back
 * to the local/profile-only path. The exact rungs are what let a session's
 * owner be reconstructed after the bounded hint map has evicted it or the app
 * relaunched.
 */
export function knownSessionOwner(sessions: readonly SessionInfo[], sessionId: null | string): SessionOwnerScope {
  if (!sessionId) {
    return undefined
  }

  const session = sessions.find(candidate => sessionMatchesStoredId(candidate, sessionId))
  const profile = session?.profile?.trim()
  const connectionId = session?.connection_id?.trim()
  const hint = getSessionOwnerHint(sessionId)

  if (connectionId) {
    return { connectionId, profile: profile || 'default' }
  }

  const hintProfiles = new Set([hint?.profile.trim() || 'default', hint?.targetProfile?.trim() || 'default'])

  if (hint && (!profile || hintProfiles.has(profile || 'default'))) {
    return hint
  }

  if (profile) {
    return profile
  }

  return hint
}

/**
 * The profile a routed session belongs to, for keying the remembered id and
 * other PRESENTATION uses (which profile's sidebar/navigation this session sits
 * under). Falls back to the active gateway profile when the owner is unknown.
 *
 * Do NOT use this to ROUTE a session-scoped RPC: the active-profile fallback is
 * exactly what sends a hidden/unlisted session's RPC to a backend that never
 * owned it. Routing must use `knownSessionOwner` + a cross-profile probe and
 * surface an error instead of falling back. This remains for the navigation
 * keying it was written for.
 */
export function rememberedSessionProfile(
  sessions: readonly SessionInfo[],
  sessionId: null | string,
  activeProfile: null | string
): string {
  return knownSessionProfile(sessions, sessionId) ?? ((activeProfile ?? '').trim() || 'default')
}

// The last non-overlay route (a page like /skills, or a session route), so a
// relaunch lands back where you were instead of a bare new-chat.
//
// Scoped per profile for the same reason the remembered session id is: a single
// global key remembered ONE route across every profile, and a session route
// carries a session id in its path. Restoring under profile B would navigate to
// a session owned by profile A — the remembered-id scoping above is bypassed
// entirely, because the route is preferred over the id on cold start
// (#67603 family). Legacy global values are discarded on first read.

export function getRememberedRoute(profile: string): null | string {
  discardLegacyRememberedNavigation()

  return storedString(profileNavigationKey(LAST_ROUTE_KEY, profile))
}

export function setRememberedRoute(path: null | string, profile: string): void {
  discardLegacyRememberedNavigation()
  persistString(profileNavigationKey(LAST_ROUTE_KEY, profile), path)
}

let configuredDefaultProjectDir = ''

function workspaceCwdKey(connection: HermesConnection | null = $connection.get()): string {
  if (connection?.mode !== 'remote') {
    return WORKSPACE_CWD_KEY
  }

  const base = encodeURIComponent(connection.baseUrl || 'remote')
  const profile = encodeURIComponent(connection.profile || 'default')

  return `${WORKSPACE_CWD_KEY}.remote.${base}.${profile}`
}

export const getRememberedWorkspaceCwd = (): string => storedString(workspaceCwdKey())?.trim() || ''
export type NewChatWorkspaceTarget = null | string | undefined

export const getConfiguredDefaultProjectDir = (): string => configuredDefaultProjectDir

export async function syncConfiguredDefaultProjectDir(shouldPublish: () => boolean = () => true): Promise<string> {
  const settings = window.hermesDesktop?.settings?.getDefaultProjectDir

  if (!settings) {
    if (shouldPublish()) {
      configuredDefaultProjectDir = ''
    }

    return configuredDefaultProjectDir
  }

  const { dir } = await settings()

  if (shouldPublish()) {
    configuredDefaultProjectDir = dir?.trim() || ''
  }

  return configuredDefaultProjectDir
}

/** Align the renderer workspace with the main-process default (home dir when
 *  packaged, optional Settings override). Clears stale install-dir paths that
 *  PR #37586's localStorage stickiness can preserve across the #37536 fix. */
export async function ensureDefaultWorkspaceCwd(shouldPublish: () => boolean = () => true): Promise<void> {
  const sanitize = window.hermesDesktop?.sanitizeWorkspaceCwd

  if (!sanitize || !shouldPublish()) {
    return
  }

  await syncConfiguredDefaultProjectDir(shouldPublish)

  if (!shouldPublish()) {
    return
  }

  const configured = getConfiguredDefaultProjectDir()

  // Transient: each source below is already remembered or comes from config, so
  // persisting would only promote a configured default into the per-backend
  // memory of what the user picked.
  const seedLiveCwd = (cwd: string) => {
    if (shouldPublish() && cwd && !$activeSessionId.get()) {
      setCurrentCwdTransient(cwd)
    }
  }

  const remembered = getRememberedWorkspaceCwd()

  if ($connection.get()?.mode === 'remote') {
    seedLiveCwd(remembered)

    return
  }

  if (configured) {
    const { cwd } = await sanitize(configured)
    seedLiveCwd(cwd)

    return
  }

  if (remembered) {
    const { cwd } = await sanitize(remembered)
    seedLiveCwd(cwd)
  }
}

export function applyConfiguredDefaultProjectDir(dir: null | string | undefined): void {
  configuredDefaultProjectDir = dir?.trim() || ''
}

interface AppAtom<T> {
  get: () => T
  set: (value: T) => void
}

function updateAtom<T>(store: AppAtom<T>, next: Updater<T>) {
  store.set(typeof next === 'function' ? (next as (current: T) => T)(store.get()) : next)
}

/** Durable id for pinning. Auto-compression rotates a conversation's session
 *  id (root -> continuation tip), so pins keyed on the live id evaporate. The
 *  lineage root is stable across every compression, so we pin on that. */
export const sessionPinId = (session: Pick<SessionInfo, '_lineage_root_id' | 'id'>): string =>
  session._lineage_root_id ?? session.id

/** True when a stored/lineage id resolves to this session — it matches either
 *  the live id or the stable lineage root (see sessionPinId). The one place the
 *  "same conversation across compression" test lives. */
export const sessionMatchesStoredId = (
  session: Pick<SessionInfo, '_lineage_root_id' | 'id'>,
  storedSessionId: string
): boolean => session.id === storedSessionId || session._lineage_root_id === storedSessionId

// Alias lookup, memoized per sessions-list reference. `lineageAliases` runs
// per cached session state per status projection per message delta — an
// O(sessions) scan there multiplies out to states × sessions × ~30Hz per busy
// session, which is what made a populated recents list drag every stream. The
// list is replaced wholesale (never mutated), so its reference is the cache key.
type LineageRow = Pick<SessionInfo, '_lineage_root_id' | 'id'>
const lineageIndexBySessions = new WeakMap<readonly LineageRow[], Map<string, string[]>>()

function lineageIndex(sessions: readonly LineageRow[]): Map<string, string[]> {
  const cached = lineageIndexBySessions.get(sessions)

  if (cached) {
    return cached
  }

  const index = new Map<string, string[]>()

  const add = (key: string, value: string) => {
    const bucket = index.get(key)

    if (!bucket) {
      index.set(key, [value])
    } else if (!bucket.includes(value)) {
      bucket.push(value)
    }
  }

  for (const session of sessions) {
    add(session.id, session.id)

    if (session._lineage_root_id) {
      add(session.id, session._lineage_root_id)
      add(session._lineage_root_id, session.id)
      add(session._lineage_root_id, session._lineage_root_id)
    }
  }

  lineageIndexBySessions.set(sessions, index)

  return index
}

/** Every id one conversation answers to: the id we were handed, plus the live
 *  id and lineage root of each session it resolves to.
 *
 *  Status sets are published under a session's CURRENT stored id, but a sidebar
 *  row, a persisted tile, and the route can each hold a different tip of the
 *  same lineage after a compression. Publishing every alias lets those surfaces
 *  keep using a plain membership test instead of each re-deriving lineage —
 *  and getting it wrong, which reads as a running session going idle mid-turn. */
export function lineageAliases(storedId: string, sessions: readonly LineageRow[]): string[] {
  // Every key is in its own bucket by construction, so the bucket IS the
  // alias set. Copied so no caller can mutate the shared index.
  return lineageIndex(sessions).get(storedId)?.slice() ?? [storedId]
}

/** True when two ids name the same conversation across compression tip rotation. */
export function idsShareLineage(
  a: string,
  b: string,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[]
): boolean {
  if (a === b) {
    return true
  }

  return sessions.some(session => sessionMatchesStoredId(session, a) && sessionMatchesStoredId(session, b))
}

/**
 * Whether a composer draft/queue key should move from `fromKey` onto `toKey`.
 *
 * Only same-conversation rekeys are allowed (compression tip → lineage root).
 * A session-switch window where the route already points at B while the store
 * selection still holds A must NOT migrate — that would re-home Session A's
 * queued prompts onto B and auto-drain them into the wrong chat.
 */
export function shouldMigrateComposerScope(
  fromKey: string | null | undefined,
  toKey: string | null | undefined,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[]
): boolean {
  const from = fromKey?.trim()
  const to = toKey?.trim()

  if (!from || !to || from === to) {
    return false
  }

  return idsShareLineage(from, to, sessions)
}

/**
 * Stable composer + `/queue` scope for a selected stored session.
 *
 * Same durability rule as {@link sessionPinId}: prefer the lineage root so
 * auto-compression tip rotation does not remount the composer onto an empty
 * draft/queue key mid-keystroke. Falls back to the live id when the row is
 * not in the in-memory list yet.
 */
export function resolveComposerSessionKey(
  selectedSessionId: string | null | undefined,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[]
): string | null {
  if (!selectedSessionId) {
    return null
  }

  const row = sessions.find(session => sessionMatchesStoredId(session, selectedSessionId))

  return row ? sessionPinId(row) : selectedSessionId
}

/** Merge a fresh server session page into the in-memory list, keeping any
 *  row the server omitted that we still want visible — both still-"working"
 *  sessions and pinned sessions.
 *
 *  Two reasons the server drops a row we must keep:
 *
 *  1. A brand-new session's first user message isn't flushed to the SessionDB
 *     until its turn is persisted, so `listSessions(min_messages=1)` skips
 *     sessions that are mid-first-response. Because every `message.complete`
 *     triggers a full refresh, a hard replace makes concurrent new chats vanish
 *     the instant any one of them finishes.
 *  2. The sidebar lists only the most-recent page (`SIDEBAR_SESSIONS_PAGE_SIZE`)
 *     ordered by activity. A pinned conversation that hasn't been touched in a
 *     while falls off that page, so a hard replace silently evicts it from the
 *     in-memory list — and because the Pinned section resolves pins against
 *     that list, the pin "disappears until you refresh".
 *
 *  `keepIds` carries both the working set and the pinned set. Pins are stored
 *  on the durable lineage-root id (see {@link sessionPinId}), while the loaded
 *  row surfaces under its live compression tip, so we match a survivor by
 *  either its live `id` or its `_lineage_root_id`. Optimistic deletes/archives
 *  drop the row from `previous` (and unpin it), so a removed session can't be
 *  resurrected here. */
const profileKeyOf = (profile: null | string | undefined): string => (profile ?? '').trim() || 'default'

function carriedConnectionId(prev: SessionInfo | undefined, incoming: SessionInfo): string | undefined {
  if (incoming.connection_id?.trim()) {
    return incoming.connection_id
  }

  const carried = prev?.connection_id?.trim()

  if (!carried) {
    return undefined
  }

  return !incoming.profile?.trim() || profileKeyOf(incoming.profile) === profileKeyOf(prev?.profile)
    ? carried
    : undefined
}

export function mergeSessionPage(
  previous: SessionInfo[],
  incoming: SessionInfo[],
  keepIds: Iterable<string>
): SessionInfo[] {
  const keep = keepIds instanceof Set ? keepIds : new Set(keepIds)

  // Rows are identified by (profile, id), never bare id: two profiles can
  // hold sessions with the SAME stored id (restored backups, copied
  // state.dbs, cross-profile imports — #92454). Keyed by bare id, the twins
  // collapse into one sidebar row whose title/preview carry can stitch one
  // profile's content onto the other profile's route — the user clicks a row
  // previewing profile A and the resume dials profile B. Same-profile rows
  // (including untagged ones, which normalize together) keep the exact
  // carry behavior below.
  // (Local normalize: importing @/store/profile here would be circular —
  // same '' → 'default' rule as normalizeProfileKey.)
  const profileKeyOf = (session: SessionInfo) => (session.profile ?? '').trim() || 'default'
  const identity = (session: SessionInfo) => `${profileKeyOf(session)}::${session.id}`

  const lineageIdentity = (session: SessionInfo) =>
    `${profileKeyOf(session)}::${session._lineage_root_id ?? session.id}`

  // Carry a known title onto a row that arrives title-less, so a freshly
  // submitted session (e.g. a branch draft) holds its placeholder instead of
  // flashing its raw message preview in the gap between persist and the async
  // auto-titler. A real clear sets the local title null first, so this never
  // masks one.
  const prevById = new Map(previous.map(session => [identity(session), session]))
  // Tip rotation changes the live id — carry activity/title across the lineage
  // root so a mid-turn refresh can't drop a touchSessionActivity bump.
  const prevByLineage = new Map(previous.map(session => [lineageIdentity(session), session]))

  const merged = incoming.map(session => {
    const prev = prevById.get(identity(session)) ?? prevByLineage.get(lineageIdentity(session))
    // User-send stamps last_active before the DB flushes the user row
    // (last_active = MAX(messages.timestamp)). Keep the fresher of the two.
    const last_active = Math.max(prev?.last_active ?? 0, session.last_active ?? 0)
    const title = session.title?.trim() ? session.title : prev?.title?.trim() ? prev.title : session.title
    // Carry the owning connection onto a row that arrives untagged. The
    // primary aggregate serves a `local` registry source's rows as plain
    // local rows (the unified-list splice tags only NON-local sources), so
    // the first refresh after a routed create used to replace the optimistic
    // row's exact owner (connection_id + profile) with a bare profile — after
    // which only the transient owner hint knew which socket held the runtime.
    // A refresh is new information layered over what we know, not a clobber;
    // the tag is kept only while the row still names the same profile.
    const connection_id = carriedConnectionId(prev, session)

    return last_active === session.last_active && title === session.title && connection_id === session.connection_id
      ? session
      : { ...session, last_active, title, ...(connection_id ? { connection_id } : {}) }
  })

  if (keep.size === 0) {
    return merged
  }

  const incomingIds = new Set(merged.map(identity))

  // Deduplicate by compression lineage: when auto-compression rotates the tip
  // id (old #4 → new #5), the incoming page carries the new tip but the
  // previous list still holds the old one.  Without lineage-level dedup both
  // rows survive as separate sidebar entries (fixes #43483). Lineage keys are
  // profile-qualified for the same reason as identity above — a twin id in
  // another profile is a DIFFERENT session and must survive the dedupe.
  const incomingLineageKeys = new Set(merged.map(lineageIdentity))

  const survivors = previous.filter(
    session =>
      !incomingIds.has(identity(session)) &&
      !incomingLineageKeys.has(lineageIdentity(session)) &&
      (keep.has(session.id) || (session._lineage_root_id != null && keep.has(session._lineage_root_id)))
  )

  if (!survivors.length) {
    return merged
  }

  // Survivors carry their old relative positions from `previous`, which can be
  // stale — the server page is the fresh `order=recent` truth. Sort survivors
  // by the same effective-recency key the backend sorts by (last_active with a
  // started_at fallback) and interleave them into the title-preserving merged
  // rows so a retained session lands where recency puts it instead of the
  // whole set forming a stale block at the top of the sidebar (fixes #47203).
  // Ties keep the survivor first, matching the old prepend behavior.
  const recency = (session: SessionInfo): number => Math.max(session.last_active || 0, session.started_at || 0)

  const sortedSurvivors = [...survivors].sort((a, b) => recency(b) - recency(a))
  const interleaved: SessionInfo[] = []
  let survivorIndex = 0
  let mergedIndex = 0

  while (survivorIndex < sortedSurvivors.length && mergedIndex < merged.length) {
    if (recency(sortedSurvivors[survivorIndex]) >= recency(merged[mergedIndex])) {
      interleaved.push(sortedSurvivors[survivorIndex++])
    } else {
      interleaved.push(merged[mergedIndex++])
    }
  }

  while (survivorIndex < sortedSurvivors.length) {
    interleaved.push(sortedSurvivors[survivorIndex++])
  }

  while (mergedIndex < merged.length) {
    interleaved.push(merged[mergedIndex++])
  }

  return interleaved
}

/** Raise a session in recents on user send (before stream / turn resolve). */
export function touchSessionActivity(
  sessionId: string | null | undefined,
  options?: { at?: number; preview?: string }
): void {
  const id = sessionId?.trim()

  if (!id) {
    return
  }

  const at = options?.at ?? Date.now() / 1000
  const preview = options?.preview?.trim().slice(0, 200) || undefined

  setSessions(prev => {
    let changed = false

    const next = prev.map(session => {
      if (!sessionMatchesStoredId(session, id)) {
        return session
      }

      const last_active = Math.max(session.last_active ?? 0, at)

      if (last_active === session.last_active && (!preview || preview === session.preview)) {
        return session
      }

      changed = true

      return preview ? { ...session, last_active, preview } : { ...session, last_active }
    })

    return changed ? next : prev
  })
}

export const $connection = atom<HermesConnection | null>(null)
export const $gatewayState = atom<ConnectionState>('idle')
export const $sessions = atom<SessionInfo[]>([])
// Cron-job sessions (source === 'cron') are fetched as their own list so the
// scheduler's always-newest sessions never crowd recents out of the page
// budget. Powers the collapsed "Cron jobs" sidebar section.
export const $cronSessions = atom<SessionInfo[]>([])
// Max cron sessions fetched for the sidebar section (single bounded page). When
// the fetch returns exactly this many rows we know more exist, so the section
// badge renders "N+". Lives here so the controller (fetch) and sidebar (badge)
// share one source of truth without a circular import.
export const CRON_SECTION_LIMIT = 50
// Messaging-platform sessions (telegram/discord/...) are fetched as their own
// slice — separate from local recents — so each platform renders a
// self-managed sidebar section and never interleaves with (or buries) local
// chats in the recents page. One combined fetch seeds every platform; a
// platform that exceeds this cap gets its own per-platform "load more".
export const $messagingSessions = atom<SessionInfo[]>([])
export const MESSAGING_SECTION_LIMIT = 100
// Exact per-platform conversation totals, keyed by source id. Empty until a
// per-platform "load more" fetch resolves it (the combined seed fetch only
// knows the aggregate), so sections fall back to their loaded count.
export const $messagingPlatformTotals = atom<Record<string, number>>({})
// True when the combined seed fetch hit MESSAGING_SECTION_LIMIT, so at least
// one platform may have more rows on disk than were loaded.
export const $messagingTruncated = atom<boolean>(false)

/**
 * Every session row the renderer knows, for OWNER lookups. The sidebar splits
 * its fetch into three source-scoped slices ($sessions / $cronSessions /
 * $messagingSessions), and each slice's rows carry the same `profile` (and,
 * when tagged, `connection_id`) stamps — but the owner ladder's row rung only
 * searched recents. A cron or messaging session's approval.respond (or any
 * session-scoped RPC) therefore found no owner, and on a registry-topology
 * install failed closed with SessionOwnerResolutionError even though the row
 * naming its owner was already in memory, one atom over. Concatenation order
 * mirrors lookup priority: recents first (they can carry fresher optimistic
 * connection tags), then the cron and messaging slices.
 */
export function ownerLookupSessionRows(): SessionInfo[] {
  const cron = $cronSessions.get()
  const messaging = $messagingSessions.get()

  // Recents-only stays the common case; keep its array identity (no copy) so
  // per-list memo caches (lineageAliases) keyed on the reference still hit.
  if (!cron.length && !messaging.length) {
    return $sessions.get()
  }

  return [...$sessions.get(), ...cron, ...messaging]
}

// Whether a profile's last session page was CAPPED by the request limit, keyed
// by profile name — i.e. more rows exist on disk than were loaded. Replaces the
// old exact per-profile totals: rendering `loaded/total` in the sidebar cost a
// COUNT(*) per profile DB on every refresh and only ever confused people, while
// "is there another page?" is what pagination actually needs and comes free
// from the row count the query already returned.
export const $sessionProfilesTruncated = atom<Record<string, boolean>>({})

/** Tokens and spend per profile across ALL its sessions, not just the loaded
 *  page — summed in SQL so a profile group's header total doesn't move when the
 *  window does. Keyed by profile name. */
export interface ProfileUsage {
  cost_usd: number
  tokens: number
}

export const $sessionProfilesUsage = atom<Record<string, ProfileUsage>>({})
export const $sessionsLoading = atom(true)
export const $activeSessionId = atom<string | null>(null)
export const $selectedStoredSessionId = atom<string | null>(null)
export interface ActiveSessionStoredIdRotation {
  nextStoredSessionId: string
  previousStoredSessionId: string
  runtimeSessionId: string
}

// One-shot event for when auto-compression rotates the active runtime's stored
// id. Carrying the runtime + previous id is load-bearing: a bare next id cannot
// tell whether the user has already navigated away while React is waiting to
// run the route-following effect, which lets a background session steal the
// foreground route.
export const $activeSessionStoredIdRotation = atom<ActiveSessionStoredIdRotation | null>(null)
export const $messages = atom<ChatMessage[]>([])

// Streaming-stable derivations of $messages. During a token stream the array
// is replaced ~30×/s; components that only care about coarse facts (is the
// thread empty? is the tail a user message?) subscribe to these instead of
// $messages so per-token flushes don't re-render them — nanostores' `computed`
// only notifies when the derived VALUE changes.
export const $messagesEmpty = computed($messages, messages => messages.length === 0)
export const $lastVisibleMessageIsUser = computed($messages, lastVisibleMessageIsUser)

export const $freshDraftReady = atom(false)
export const $busy = atom(false)
export const $awaitingResponse = atom(false)
// Stored-session id whose most recent resume FAILED terminally (the gateway RPC
// rejected AND the REST transcript fallback also failed), leaving the window
// with no runtime and an empty transcript. Drives use-route-resume's self-heal:
// while this matches the routed session the loader would otherwise latch
// forever (messagesEmpty && !activeSessionId), so the hook re-attempts the
// resume on the next render/focus/reconnect instead of stranding the window.
// Null whenever the active route has a healthy (or in-flight) resume.
export const $resumeFailedSessionId = atom<string | null>(null)
export interface SessionResumeRequest {
  ownerRoute?: SessionOwnerRoute
  sequence: number
  sessionId: string
}
let sessionResumeRequestSequence = 0
export const $sessionResumeRequest = atom<SessionResumeRequest | null>(null)
// ── Exact session owner hints ───────────────────────────────────────────────
// The (connectionId, profile[, targetProfile, mode]) route a session was
// created / resumed / opened on, keyed by stored id. Bounded LRU and
// PERSISTED (best-effort, same origin storage as the tiles): the runtime a
// routed create minted lives on one concrete socket, and after the sidebar
// refresh replaced the optimistic row, or after a relaunch, this record is
// how the exact owner is reconstructed for that session's next RPC instead of
// degrading to a bare profile name that dials a different socket. Connection
// ids are stable registry identities (`local`, registry uuids), so a hint
// stays valid across restarts; forgetSessionOwnerHintsForConnection drops
// them when a connection is removed from the registry.
const SESSION_OWNER_HINT_LIMIT = 256
const SESSION_OWNER_HINTS_KEY = 'hermes.desktop.sessionOwnerHints.v1'
const sessionOwnerHints = new Map<string, { id: string; route: SessionOwnerRoute }>()

function sessionOwnerHintKey(sessionId: string, route: Pick<SessionOwnerRoute, 'connectionId' | 'profile'>): string {
  return JSON.stringify([route.connectionId.trim(), route.profile.trim() || 'default', sessionId])
}

function normalizeOwnerRoute(route: SessionOwnerRoute): SessionOwnerRoute {
  return {
    ...route,
    connectionId: route.connectionId.trim(),
    profile: route.profile.trim() || 'default',
    ...(route.targetProfile ? { targetProfile: route.targetProfile.trim() || 'default' } : {})
  }
}

function persistSessionOwnerHints(): void {
  writeJson(
    SESSION_OWNER_HINTS_KEY,
    sessionOwnerHints.size === 0 ? null : [...sessionOwnerHints.values()].map(entry => [entry.id, entry.route])
  )
}

function rememberSessionOwnerHint(sessionId: string, route: SessionOwnerRoute): boolean {
  const id = sessionId.trim()
  const normalized = normalizeOwnerRoute(route)

  if (!id || !normalized.connectionId) {
    return false
  }

  const key = sessionOwnerHintKey(id, normalized)
  sessionOwnerHints.delete(key)
  sessionOwnerHints.set(key, { id, route: normalized })

  while (sessionOwnerHints.size > SESSION_OWNER_HINT_LIMIT) {
    const oldest = sessionOwnerHints.keys().next().value

    if (oldest === undefined) {
      break
    }

    sessionOwnerHints.delete(oldest)
  }

  return true
}

/** Load persisted hints (oldest first, so LRU order survives). Malformed or
 *  foreign-shaped entries are skipped; nothing here can throw. */
export function hydrateSessionOwnerHints(): void {
  const raw = readJson<unknown>(SESSION_OWNER_HINTS_KEY)

  if (!Array.isArray(raw)) {
    return
  }

  for (const entry of raw) {
    if (!Array.isArray(entry) || entry.length !== 2) {
      continue
    }

    const [id, route] = entry as [unknown, unknown]

    if (
      typeof id !== 'string' ||
      !route ||
      typeof route !== 'object' ||
      typeof (route as SessionOwnerRoute).connectionId !== 'string' ||
      typeof (route as SessionOwnerRoute).profile !== 'string'
    ) {
      continue
    }

    const candidate = route as SessionOwnerRoute

    rememberSessionOwnerHint(id, {
      connectionId: candidate.connectionId,
      profile: candidate.profile,
      ...(typeof candidate.targetProfile === 'string' ? { targetProfile: candidate.targetProfile } : {}),
      ...(candidate.mode === 'local' || candidate.mode === 'remote' ? { mode: candidate.mode } : {})
    })
  }
}

hydrateSessionOwnerHints()

export function setSessionOwnerHint(sessionId: string, route: SessionOwnerRoute): void {
  if (rememberSessionOwnerHint(sessionId, route)) {
    persistSessionOwnerHints()
  }
}

/** Drop every hint naming `connectionId` — the registry no longer has it, so
 *  nothing can dial that route again (fail-closed would otherwise pin those
 *  sessions to a dead source forever). */
export function forgetSessionOwnerHintsForConnection(connectionId: string): void {
  const id = connectionId.trim()

  if (!id) {
    return
  }

  let changed = false

  for (const [key, entry] of [...sessionOwnerHints]) {
    if (entry.route.connectionId === id) {
      sessionOwnerHints.delete(key)
      changed = true
    }
  }

  if (changed) {
    persistSessionOwnerHints()
  }
}

/** Drop every persisted route for one session. Untagged rows are owned by the
 * ambient backend that returned them, so a stale explicit hint must not force a
 * later resume onto a different connection. */
export function forgetSessionOwnerHintsForSession(sessionId: string): void {
  const id = sessionId.trim()

  if (!id) {
    return
  }

  let changed = false

  for (const [key, entry] of [...sessionOwnerHints]) {
    if (entry.id === id) {
      sessionOwnerHints.delete(key)
      changed = true
    }
  }

  if (changed) {
    persistSessionOwnerHints()
  }
}

/** Exact route carried by a connection-tagged row. An untagged row deliberately
 * returns undefined: it belongs to the ambient backend that supplied the list,
 * including the legacy primary-SSH path whose rows have no registry id. */
export function sessionOwnerRouteFromRow(
  session?: Pick<SessionInfo, 'connection_id' | 'profile'>
): SessionOwnerRoute | undefined {
  const connectionId = (session?.connection_id ?? '').trim()
  const profile = (session?.profile ?? '').trim()

  if (!connectionId || !profile) {
    return undefined
  }

  return { connectionId, profile, targetProfile: profile }
}

/** @internal Tests: forget every in-memory hint (storage untouched unless asked). */
export function _resetSessionOwnerHintsForTests({ storage = false }: { storage?: boolean } = {}): void {
  sessionOwnerHints.clear()

  if (storage) {
    writeJson(SESSION_OWNER_HINTS_KEY, null)
  }
}

export function getSessionOwnerHints(sessionId: string): SessionOwnerRoute[] {
  const id = sessionId.trim()

  return [...sessionOwnerHints.values()].filter(entry => entry.id === id).map(entry => ({ ...entry.route }))
}

export function getSessionOwnerHint(
  sessionId: string,
  scope?: Pick<SessionOwnerRoute, 'connectionId' | 'profile'>
): SessionOwnerRoute | undefined {
  const id = sessionId.trim()

  if (scope) {
    const entry = sessionOwnerHints.get(sessionOwnerHintKey(id, scope))

    return entry ? { ...entry.route } : undefined
  }

  const matches = [...sessionOwnerHints.values()].filter(entry => entry.id === id)

  return matches.length === 1 ? { ...matches[0].route } : undefined
}

// Stored-session id whose resume has EXHAUSTED its bounded auto-retries (the
// terminal-failure latch above kept failing through all MAX_RESUME_RETRIES
// attempts). Distinct from $resumeFailedSessionId, which is armed *during* the
// backoff window too: this fires only once auto-recovery has given up, so the
// chat view can swap the perpetual loader for an explicit error + manual Retry
// affordance. A fresh resumeSession() (manual Retry, reconnect, reselect)
// clears it and resets the retry counter. Null whenever the active route has a
// healthy, in-flight, or still-auto-retrying resume.
export const $resumeExhaustedSessionId = atom<string | null>(null)
export const $currentModel = atom(storedComposerString(COMPOSER_MODEL_KEY) ?? '')
export const $currentProvider = atom(storedComposerString(COMPOSER_PROVIDER_KEY) ?? '')
export const $currentReasoningEffort = atom(storedString(COMPOSER_EFFORT_KEY) ?? '')
export const $currentServiceTier = atom('')
export const $currentFastMode = atom(storedBoolean(COMPOSER_FAST_KEY, false))
// Effective approval-bypass state mirrored from the gateway (session.info).
// Persistence lives in the backend config (approvals.mode), so this is a plain
// reflection of the truth the gateway reports rather than its own store.
export const $yoloActive = atom(false)
export const $currentCwd = atom(getRememberedWorkspaceCwd())

// Which conversation the live `$currentCwd` is known to describe. Three
// inhabitants, and the difference between the last two is load-bearing:
// a stored-session id (that conversation owns the path), `null` (the fresh-draft
// state, which MATCHES a null selection and therefore reads as OWNED — a draft's
// workspace is immediately usable), and the released marker
// `WORKSPACE_CWD_UNOWNED` below, which matches no selection and so reads as
// owned by nobody. `null` cannot double as the release value precisely because
// it matches: releasing to `null` while a draft is selected would hand the
// leftover path to the draft as its own workspace.
//
// A conversation switch publishes the new stored id immediately, but the new
// workspace only arrives when the resume settles, so for that whole window
// `$currentCwd` still holds the PREVIOUS conversation's folder. Without a way to
// say "this path is not this conversation's yet", workspace-derived surfaces
// treat the leftover path as authoritative and show the old repo's cached Git
// facts under the newly selected chat (#71254).
//
// Ownership, not emptiness, is what makes the switch atomic: clearing the path
// would collapse the workspace panes and drop file-tree state on every switch,
// so the path stays put and is simply marked as not-yet-owned.
export const $workspaceCwdOwner = atom<null | string>(null)

// Terminal execution backend (local | docker | ssh | ...) mirrored from the
// gateway (session.info). Drives attachment upload decisions: container
// backends have their own filesystem, so a dropped host path must be uploaded
// as bytes and staged into a bind-mounted cache dir (#76577).
export const $terminalBackend = atom('')
export const $newChatWorkspaceTarget = atom<NewChatWorkspaceTarget>(undefined)
export const $newChatWorkspaceTargetGeneration = atom(0)
export const $currentBranch = atom('')
export const $currentUsage = atom<UsageStats>({
  calls: 0,
  input: 0,
  output: 0,
  total: 0
})
export const $sessionStartedAt = atom<number | null>(null)
export const $turnStartedAt = atom<number | null>(null)
export const $introPersonality = atom('')
export const $currentPersonality = atom('')
export const $availablePersonalities = atom<string[]>([])
export const $introSeed = atom(0)
export const $contextSuggestions = atom<ContextSuggestion[]>([])
export const $modelPickerOpen = atom(false)
export const $sessionPickerOpen = atom(false)

function rescopeComposerSelection(nextScope: string | null): void {
  if (nextScope === composerSelectionScope) {
    return
  }

  composerSelectionScope = nextScope
  $currentModel.set(storedComposerString(COMPOSER_MODEL_KEY) ?? '')
  $currentProvider.set(storedComposerString(COMPOSER_PROVIDER_KEY) ?? '')
  $currentModelSource.set(getCurrentModelSource())
}

/** Publish an exact registry route before active-profile effects can persist a
 * forced default. A registry id is authority even while its descriptive
 * HermesConnection lookup is unavailable. */
export function setComposerSelectionOwner(connectionId: string, profile: string): void {
  rescopeComposerSelection(
    `.registry.${encodeURIComponent(connectionId)}.${encodeURIComponent(profile.trim() || 'default')}`
  )
}

/** Fail closed while a successful legacy profile activation has no descriptor. */
export function clearComposerSelectionOwner(): void {
  rescopeComposerSelection(null)
}

export const setConnection = (next: Updater<HermesConnection | null>) => {
  updateAtom($connection, next)
  // Repoint connection-scoped persistence (pins, manual session order,
  // remembered navigation) at the new backend's storage scope before any
  // consumer reconciles against it. A null descriptor (reconnect blip)
  // keeps the current scope.
  rescopeConnectionScopedStores($connection.get())
  syncCronModelImpactConnection($connection.get())
  rescopeComposerSelection(composerScopeForConnection($connection.get()))
}

export const setGatewayState = (next: Updater<ConnectionState>) => updateAtom($gatewayState, next)
export const setSessions = (next: Updater<SessionInfo[]>) => updateAtom($sessions, next)
export const setCronSessions = (next: Updater<SessionInfo[]>) => updateAtom($cronSessions, next)
export const setMessagingSessions = (next: Updater<SessionInfo[]>) => updateAtom($messagingSessions, next)
export const setMessagingPlatformTotals = (next: Updater<Record<string, number>>) =>
  updateAtom($messagingPlatformTotals, next)
export const setMessagingTruncated = (next: Updater<boolean>) => updateAtom($messagingTruncated, next)
export const setSessionProfilesTruncated = (next: Updater<Record<string, boolean>>) =>
  updateAtom($sessionProfilesTruncated, next)
export const setSessionProfilesUsage = (next: Updater<Record<string, ProfileUsage>>) =>
  updateAtom($sessionProfilesUsage, next)
export const setSessionsLoading = (next: Updater<boolean>) => updateAtom($sessionsLoading, next)
export const setActiveSessionId = (next: Updater<string | null>) => updateAtom($activeSessionId, next)
export const setActiveSessionStoredIdRotation = (next: Updater<ActiveSessionStoredIdRotation | null>) =>
  updateAtom($activeSessionStoredIdRotation, next)

// A background session finished and the user hasn't opened it since. This atom
// is the transient PAINT layer (what the dots subscribe to); durability lives
// in session-unread.ts, which persists explicit finish markers + per-session
// "seen message_count" watermarks and rebuilds this atom from them on every
// list refresh — so the green dot survives an app restart, and a session that
// finished while the app was CLOSED still comes up unread. The explicit
// Mark-as-unread toggle rides the BACKEND watermark instead
// (SessionDB.set_session_read, session-unread-remote.ts). Written by
// session-states.ts (live busy→idle edge), cleared here on session open.
export const $unreadFinishedSessionIds = atom<string[]>([])

/** Sidebar "mark all as read" — clears every finished-unread dot. Purely
 *  renderer-local, like the atom itself. */
export function markAllSessionsRead() {
  if ($unreadFinishedSessionIds.get().length > 0) {
    $unreadFinishedSessionIds.set([])
  }
}

// Last time the user actually viewed a session. A finished turn should only
// re-arm the unread marker if it settles AFTER this baseline; otherwise an
// already-viewed completion keeps re-lighting the row.
export const $lastReadAtBySessionId = atom<Record<string, number>>({})

/** A new turn started for this session: the read baseline only guarded the
 *  PREVIOUS completion's re-asserts, so drop it — the new turn's finish must
 *  re-light even when it lands in the same millisecond as the last read. */
export const clearReadBaseline = (storedSessionId: string) => {
  const map = $lastReadAtBySessionId.get()

  if (storedSessionId in map) {
    const { [storedSessionId]: _dropped, ...rest } = map
    $lastReadAtBySessionId.set(rest)
  }
}

export const setSelectedStoredSessionId = (next: Updater<string | null>) => {
  updateAtom($selectedStoredSessionId, next)
  // Opening a session clears its unread state — the user is now looking at it.
  // Clear the whole conversation family (branch children + compression lineage
  // root), not just the exact row: the sidebar lights the dot for every alias
  // of a lineage, so reading any row must clear all of them.
  const id = $selectedStoredSessionId.get()

  if (id) {
    markSessionRead(id)
  }

  // ...and the persisted watermark flag, when the row carried one.
  if (id) {
    void clearUnreadOnOpen(id)
  }
}

/** Record that the user has seen a session (and its conversation family) at
 *  this moment. Clears the unread set for the family and stores a last-read
 *  baseline so a later completion that settles BEFORE this view is not
 *  re-lit. Must be callable before any focus short-circuit (openSession top)
 *  so re-clicking an already-visible session still clears its dot. */
export const markSessionRead = (storedSessionId: string | null | undefined) => {
  if (!storedSessionId) {
    return
  }

  const sessions = $sessions.get()
  const familyIds = new Set<string>(lineageAliases(storedSessionId, sessions))

  const lastReadAt = Date.now()
  const nextReadMap = { ...$lastReadAtBySessionId.get() }

  for (const id of familyIds) {
    nextReadMap[id] = lastReadAt
  }

  $lastReadAtBySessionId.set(nextReadMap)
  $unreadFinishedSessionIds.set($unreadFinishedSessionIds.get().filter(id => !familyIds.has(id)))
}

export const setMessages = (next: Updater<ChatMessage[]>) => updateAtom($messages, next)
export const setFreshDraftReady = (next: Updater<boolean>) => updateAtom($freshDraftReady, next)
export const setResumeFailedSessionId = (next: Updater<string | null>) => updateAtom($resumeFailedSessionId, next)

export const requestSessionResume = (sessionId: string, ownerRoute?: SessionOwnerRoute) => {
  const id = sessionId.trim()

  if (!id) {
    return
  }

  if (ownerRoute) {
    setSessionOwnerHint(id, ownerRoute)
  }

  $sessionResumeRequest.set({
    ...(ownerRoute ? { ownerRoute: { ...ownerRoute } } : {}),
    sequence: ++sessionResumeRequestSequence,
    sessionId: id
  })
}

export const setResumeExhaustedSessionId = (next: Updater<string | null>) => updateAtom($resumeExhaustedSessionId, next)
export const setBusy = (next: Updater<boolean>) => updateAtom($busy, next)
export const setAwaitingResponse = (next: Updater<boolean>) => updateAtom($awaitingResponse, next)

export const setCurrentModel = (next: Updater<string>) => {
  updateAtom($currentModel, next)
  const key = composerSelectionKey(COMPOSER_MODEL_KEY)

  if (key !== null) {
    persistString(key, $currentModel.get() || null)
  }
}

export const setCurrentProvider = (next: Updater<string>) => {
  updateAtom($currentProvider, next)
  const key = composerSelectionKey(COMPOSER_PROVIDER_KEY)

  if (key !== null) {
    persistString(key, $currentProvider.get() || null)
  }
}

export const getCurrentModelSource = (): ComposerModelSource => {
  const source = storedComposerString(COMPOSER_MODEL_SOURCE_KEY)

  return source === 'default' || source === 'manual' ? source : ''
}

// Reactive mirror of the persisted source so UI (the composer pill's
// override badge) can subscribe. The getter above stays storage-backed —
// it's read cross-window, where this atom wouldn't see writes.
export const $currentModelSource = atom<ComposerModelSource>(getCurrentModelSource())

export const setCurrentModelSource = (source: ComposerModelSource) => {
  const key = composerSelectionKey(COMPOSER_MODEL_SOURCE_KEY)

  if (key !== null) {
    persistString(key, source || null)
  }

  $currentModelSource.set(source)
}

// Monotonic intent token for async default refreshes. A profile/config request
// may start before the user opens the picker and finish after their click; the
// token lets that older response stand down even when the selected value is
// unchanged (value comparisons alone cannot detect re-selecting the same row).
let composerSelectionGeneration = 0

export const getComposerSelectionGeneration = (): number => composerSelectionGeneration

export const markComposerSelectionManual = (): void => {
  composerSelectionGeneration += 1
  setCurrentModelSource('manual')
}

export const setCurrentReasoningEffort = (next: Updater<string>) => {
  updateAtom($currentReasoningEffort, next)
  persistString(COMPOSER_EFFORT_KEY, $currentReasoningEffort.get() || null)
}

// The profile's `agent.reasoning_effort`, mirrored from config so surfaces that
// need to render or apply "the default" resolve the user's configured level
// instead of assuming DEFAULT_REASONING_EFFORT (lib/reasoning-effort). Empty
// until config loads, and re-seeded on every profile switch by useHermesConfig.
export const $defaultReasoningEffort = atom('')

export const setDefaultReasoningEffort = (next: string) => updateAtom($defaultReasoningEffort, next)

export const setCurrentServiceTier = (next: Updater<string>) => updateAtom($currentServiceTier, next)

export const setCurrentFastMode = (next: Updater<boolean>) => {
  updateAtom($currentFastMode, next)
  persistBoolean(COMPOSER_FAST_KEY, $currentFastMode.get())
}

export const setYoloActive = (next: Updater<boolean>) => updateAtom($yoloActive, next)

/** Move the live workspace AND remember it as this backend's workspace.
 *
 *  Only for a path the user chose — a folder pick, a project/worktree entry, an
 *  explicit workspace target. The remembered value is where a new chat starts on
 *  a remote backend, so writing it from a path the user merely *looked at* makes
 *  every new chat land in the last session's folder (#77496, #80213). To follow
 *  a conversation's cwd, use `setCurrentCwdTransient`.
 */
export const setCurrentCwd = (next: Updater<string>) => {
  updateAtom($currentCwd, next)
  persistString(workspaceCwdKey(), $currentCwd.get().trim() || null)
}

export const setTerminalBackend = (next: Updater<string>) => updateAtom($terminalBackend, next)

/** Move the live workspace without claiming it as the user's chosen one.
 *
 *  For paths that come from a conversation rather than from the user: resume
 *  settling, a warm switch, the agent relocating mid-turn, detaching a draft.
 */
export const setCurrentCwdTransient = (next: Updater<string>) => updateAtom($currentCwd, next)

// Released-ownership marker: the live path belongs to no conversation. `null`
// cannot serve as the release value because it MATCHES a fresh draft (whose
// selected id is also null), which would declare a leftover path to be the
// draft's own workspace — #71254, one selection over. Kept here beside the atom
// and the comparison so a release site cannot reinvent a subtly different value.
const WORKSPACE_CWD_UNOWNED = 'desktop:workspace-cwd-unowned'

/** Mark the live workspace as belonging to `storedSessionId`.
 *
 *  Call this wherever a cwd is established for a conversation (resume settling,
 *  a warm switch, an explicit folder pick). Until it is called for the newly
 *  selected conversation, primary workspace-derived selectors hide the previous
 *  conversation's cached facts rather than publishing them (#71254).
 */
export const setWorkspaceCwdOwner = (storedSessionId: null | string) => updateAtom($workspaceCwdOwner, storedSessionId)

/** Declare that no conversation owns the live workspace path.
 *
 *  For a conversation whose workspace is not known yet: the path on screen is
 *  provably still the previous conversation's, so workspace-derived surfaces must
 *  hide it rather than adopt it. The path itself is deliberately left alone —
 *  clearing it would collapse the workspace/review panes and drop file-tree
 *  state on every switch.
 */
export const releaseWorkspaceCwdOwner = () => updateAtom($workspaceCwdOwner, WORKSPACE_CWD_UNOWNED)

/** Commit `cwd` as the workspace of the conversation the user is looking at.
 *
 *  The single primitive for "this path IS the selected conversation's" — a folder
 *  pick, a project entry, the agent relocating itself. Prefer it over a bare
 *  `setCurrentCwdTransient`, which moves the path while leaving ownership naming
 *  whatever held it before; workspace-derived slices then stay hidden even though
 *  the path is correct (#71254).
 */
export const commitWorkspaceCwdForSelectedSession = (cwd: string) => {
  setCurrentCwdTransient(cwd)
  setWorkspaceCwdOwner($selectedStoredSessionId.get())
}

/** True when `$currentCwd` is known to describe the selected conversation. */
export const workspaceCwdBelongsToSelectedSession = (): boolean =>
  ($workspaceCwdOwner.get() ?? null) === ($selectedStoredSessionId.get() ?? null)

export const setNewChatWorkspaceTarget = (next: NewChatWorkspaceTarget): number => {
  const generation = $newChatWorkspaceTargetGeneration.get() + 1
  $newChatWorkspaceTarget.set(next)
  $newChatWorkspaceTargetGeneration.set(generation)

  return generation
}

export const workspaceCwdForNewSession = (): string => {
  // A bare new chat starts DETACHED — no inherited cwd, so the composer's coding
  // rail (which keys off $currentCwd) shows no branch and the first message runs
  // in the gateway's default rather than silently in the last repo you touched.
  // Only an explicit default-project-dir setting pre-attaches. Entering a
  // project/worktree attaches its cwd directly (startSessionInWorkspace), so the
  // "remember where I was when I'm in a project" case is unaffected.
  //
  // This must behave identically in local and remote mode: the remembered CWD
  // under the remote-keyed workspaceCwdKey() can be from a *different* project
  // than the one the user is currently scoped into, and bare-new-session in
  // the wrong workspace was the #57911 symptom. Resume/restore still reads
  // the remembered cwd via ensureDefaultWorkspaceCwd (where it remains
  // remote-keyed and intentionally sticky).
  return getConfiguredDefaultProjectDir()
}

export const setCurrentBranch = (next: Updater<string>) => updateAtom($currentBranch, next)
export const setCurrentUsage = (next: Updater<UsageStats>) => updateAtom($currentUsage, next)
export const setSessionStartedAt = (next: Updater<number | null>) => updateAtom($sessionStartedAt, next)
export const setTurnStartedAt = (next: Updater<number | null>) => updateAtom($turnStartedAt, next)
export const setIntroPersonality = (next: Updater<string>) => updateAtom($introPersonality, next)
export const setCurrentPersonality = (next: Updater<string>) => updateAtom($currentPersonality, next)
export const setAvailablePersonalities = (next: Updater<string[]>) => updateAtom($availablePersonalities, next)
export const setIntroSeed = (next: Updater<number>) => updateAtom($introSeed, next)
export const setContextSuggestions = (next: Updater<ContextSuggestion[]>) => updateAtom($contextSuggestions, next)
export const setModelPickerOpen = (next: Updater<boolean>) => updateAtom($modelPickerOpen, next)
export const setSessionPickerOpen = (next: Updater<boolean>) => updateAtom($sessionPickerOpen, next)
