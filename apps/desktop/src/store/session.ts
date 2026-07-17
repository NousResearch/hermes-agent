import { atom, computed } from 'nanostores'

import { lastVisibleMessageIsUser } from '@/app/chat/thread-loading'
import type { ContextSuggestion } from '@/app/types'
import type { HermesConnection } from '@/global'
import type { ChatMessage } from '@/lib/chat-messages'
import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'
import type { SessionInfo, UsageStats } from '@/types/hermes'

import {
  broadcastSessionUnreadChanged,
  broadcastSessionUnreadReset,
  broadcastSessionUnreadSnapshot,
  compareSessionUnreadVersions,
  nextSessionUnreadVersion,
  onSessionUnreadSync,
  requestSessionUnreadSnapshot,
  type SessionCompletionToken,
  type SessionUnreadEntry,
  type SessionUnreadVersion
} from './session-sync'

type Updater<T> = T | ((current: T) => T)
export type ComposerModelSource = '' | 'default' | 'manual'

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

// The composer's model/effort/fast is sticky UI state, NOT the profile default
// (that lives in Settings → Model). Persisting it in localStorage makes a pick
// follow across Cmd+N and app restarts instead of snapping back to the default.
// It's deliberately global (not per-profile): a profile switch force-reseeds to
// that profile's default, while within a profile new chats keep your last pick.
const COMPOSER_MODEL_KEY = 'hermes.desktop.composer.model'
const COMPOSER_PROVIDER_KEY = 'hermes.desktop.composer.provider'
const COMPOSER_MODEL_SOURCE_KEY = 'hermes.desktop.composer.model-source'
const COMPOSER_EFFORT_KEY = 'hermes.desktop.composer.reasoning-effort'
const COMPOSER_FAST_KEY = 'hermes.desktop.composer.fast'

// The last chat the user had open, so a relaunch lands back on it instead of an
// empty new-chat. Stored (not runtime) id — the route is keyed by stored id.
const LAST_SESSION_KEY = 'hermes.desktop.lastSessionId'

export const getRememberedSessionId = (): null | string => storedString(LAST_SESSION_KEY)
export const setRememberedSessionId = (id: null | string) => persistString(LAST_SESSION_KEY, id)

// The last non-overlay route (a page like /skills, or a session route), so a
// relaunch lands back where you were instead of a bare new-chat.
const LAST_ROUTE_KEY = 'hermes.desktop.lastRoute'

export const getRememberedRoute = (): null | string => storedString(LAST_ROUTE_KEY)
export const setRememberedRoute = (path: null | string) => persistString(LAST_ROUTE_KEY, path)

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

export async function syncConfiguredDefaultProjectDir(): Promise<string> {
  const settings = window.hermesDesktop?.settings?.getDefaultProjectDir

  if (!settings) {
    configuredDefaultProjectDir = ''

    return ''
  }

  const { dir } = await settings()
  configuredDefaultProjectDir = dir?.trim() || ''

  return configuredDefaultProjectDir
}

/** Align the renderer workspace with the main-process default (home dir when
 *  packaged, optional Settings override). Clears stale install-dir paths that
 *  PR #37586's localStorage stickiness can preserve across the #37536 fix. */
export async function ensureDefaultWorkspaceCwd(): Promise<void> {
  const sanitize = window.hermesDesktop?.sanitizeWorkspaceCwd

  if (!sanitize) {
    return
  }

  await syncConfiguredDefaultProjectDir()
  const configured = getConfiguredDefaultProjectDir()

  const seedLiveCwd = (cwd: string) => {
    if (cwd && !$activeSessionId.get()) {
      setCurrentCwd(cwd)
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

  // Cache only — new chats read this via workspaceCwdForNewSession(). Do not
  // rewrite the live workspace (or localStorage) while a session is active.
  if (configuredDefaultProjectDir && !$activeSessionId.get()) {
    setCurrentCwd(configuredDefaultProjectDir)
  }
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

const SESSION_SCOPE_SEPARATOR = '\u0000'

export const normalizeSessionProfile = (profile: null | string | undefined): string => profile?.trim() || 'default'

export const sessionScopeKey = (profile: null | string | undefined, sessionId: string): string =>
  `${normalizeSessionProfile(profile)}${SESSION_SCOPE_SEPARATOR}${sessionId}`

export const sessionIdFromScopeKey = (key: string): string => key.slice(key.indexOf(SESSION_SCOPE_SEPARATOR) + 1)

export const sessionProfileFromScopeKey = (key: string): string => {
  const separator = key.indexOf(SESSION_SCOPE_SEPARATOR)

  return separator < 0 ? 'default' : normalizeSessionProfile(key.slice(0, separator))
}

let requestedSessionResumeTarget: null | { profile: string; sessionId: string } = null

export function requestSessionResumeProfile(sessionId: string, profile: null | string | undefined): void {
  requestedSessionResumeTarget = { profile: normalizeSessionProfile(profile), sessionId }
}

export function consumeRequestedSessionResumeProfile(sessionId: string): string | undefined {
  const target = requestedSessionResumeTarget
  requestedSessionResumeTarget = null

  return target?.sessionId === sessionId ? target.profile : undefined
}

export function peekRequestedSessionResumeProfile(sessionId: string): string | undefined {
  return requestedSessionResumeTarget?.sessionId === sessionId ? requestedSessionResumeTarget.profile : undefined
}

export function clearRequestedSessionResumeProfile(): void {
  requestedSessionResumeTarget = null
}

const lineageRootByAlias = new Map<string, string>()
const lineageAliasesByRoot = new Map<string, Set<string>>()

function rememberSessionLineage(session: Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>): void {
  const profile = normalizeSessionProfile(session.profile)
  const declaredRoot = session._lineage_root_id

  const root =
    (declaredRoot ? lineageRootByAlias.get(sessionScopeKey(profile, declaredRoot)) : undefined) ??
    lineageRootByAlias.get(sessionScopeKey(profile, session.id)) ??
    declaredRoot ??
    session.id

  const rootKey = sessionScopeKey(profile, root)
  const aliases = lineageAliasesByRoot.get(rootKey) ?? new Set<string>()

  aliases.add(root)
  aliases.add(session.id)

  if (declaredRoot) {
    aliases.add(declaredRoot)
  }

  lineageAliasesByRoot.set(rootKey, aliases)
  aliases.forEach(alias => lineageRootByAlias.set(sessionScopeKey(profile, alias), root))
  migrateUnreadAliases(profile, root, aliases)
}

export function clearSessionLineageAliases(): void {
  lineageRootByAlias.clear()
  lineageAliasesByRoot.clear()
}

export function sessionLineageRootId(sessionId: string, profile?: null | string): string {
  $sessions.get().forEach(rememberSessionLineage)

  return lineageRootByAlias.get(sessionScopeKey(profile, sessionId)) ?? sessionId
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
export function mergeSessionPage(
  previous: SessionInfo[],
  incoming: SessionInfo[],
  keepIds: Iterable<string>
): SessionInfo[] {
  previous.forEach(rememberSessionLineage)
  incoming.forEach(rememberSessionLineage)
  const keep = keepIds instanceof Set ? keepIds : new Set(keepIds)

  // Carry a known title onto a row that arrives title-less, so a freshly
  // submitted session (e.g. a branch draft) holds its placeholder instead of
  // flashing its raw message preview in the gap between persist and the async
  // auto-titler. A real clear sets the local title null first, so this never
  // masks one.
  const prevById = new Map(previous.map(session => [sessionScopeKey(session.profile, session.id), session]))

  const merged = incoming.map(session => {
    if (session.title?.trim()) {
      return session
    }

    const carried = prevById.get(sessionScopeKey(session.profile, session.id))?.title?.trim()

    return carried ? { ...session, title: carried } : session
  })

  if (keep.size === 0) {
    return merged
  }

  const incomingIds = new Set(merged.map(session => sessionScopeKey(session.profile, session.id)))

  // Deduplicate by compression lineage: when auto-compression rotates the tip
  // id (old #4 → new #5), the incoming page carries the new tip but the
  // previous list still holds the old one.  Without lineage-level dedup both
  // rows survive as separate sidebar entries (fixes #43483).
  const incomingLineageKeys = new Set(
    merged.map(session => sessionScopeKey(session.profile, session._lineage_root_id ?? session.id))
  )

  const survivors = previous.filter(
    session =>
      !incomingIds.has(sessionScopeKey(session.profile, session.id)) &&
      !incomingLineageKeys.has(sessionScopeKey(session.profile, session._lineage_root_id ?? session.id)) &&
      (keep.has(sessionScopeKey(session.profile, session.id)) ||
        (session._lineage_root_id != null && keep.has(sessionScopeKey(session.profile, session._lineage_root_id))) ||
        keep.has(session.id) ||
        (session._lineage_root_id != null && keep.has(session._lineage_root_id)))
  )

  return survivors.length ? [...survivors, ...merged] : merged
}

export const $connection = atom<HermesConnection | null>(null)
export const $gatewayState = atom('idle')
export const $sessions = atom<SessionInfo[]>([])

/** All durable ids currently known to represent one compression lineage. */
export function sessionLineageIds(sessionId: string, profile?: null | string): string[] {
  $sessions.get().forEach(rememberSessionLineage)
  const profileKey = normalizeSessionProfile(profile)
  const root = lineageRootByAlias.get(sessionScopeKey(profileKey, sessionId)) ?? sessionId
  const aliases = lineageAliasesByRoot.get(sessionScopeKey(profileKey, root)) ?? new Set([root])

  return [...new Set([sessionId, root, ...aliases])]
}

export const $sessionsTotal = atom<number>(0)
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
// Listable conversation count per profile (children excluded), keyed by profile
// name. Lets the sidebar scope its "Load more" footer to the active profile so a
// huge default profile doesn't keep "Load more" visible while browsing a small
// one. Empty for single-profile users (fall back to $sessionsTotal).
export const $sessionProfileTotals = atom<Record<string, number>>({})
export const $sessionsLoading = atom(true)
export const $workingSessionIds = atom<string[]>([])
export const UNKNOWN_SESSION_PROFILE_SCOPE = '\u0001unknown'
export const $workingSessionProfiles = atom<Record<string, string[]>>({})
let workingSessionMutationRevision = 0
const workingSessionScopeRevisions = new Map<string, number>()
const workingSessionUnscopedRevisions = new Map<string, number>()
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
// Stored-session id whose resume has EXHAUSTED its bounded auto-retries (the
// terminal-failure latch above kept failing through all MAX_RESUME_RETRIES
// attempts). Distinct from $resumeFailedSessionId, which is armed *during* the
// backoff window too: this fires only once auto-recovery has given up, so the
// chat view can swap the perpetual loader for an explicit error + manual Retry
// affordance. A fresh resumeSession() (manual Retry, reconnect, reselect)
// clears it and resets the retry counter. Null whenever the active route has a
// healthy, in-flight, or still-auto-retrying resume.
export const $resumeExhaustedSessionId = atom<string | null>(null)
export const $currentModel = atom(storedString(COMPOSER_MODEL_KEY) ?? '')
export const $currentProvider = atom(storedString(COMPOSER_PROVIDER_KEY) ?? '')
export const $currentReasoningEffort = atom(storedString(COMPOSER_EFFORT_KEY) ?? '')
export const $currentServiceTier = atom('')
export const $currentFastMode = atom(storedBoolean(COMPOSER_FAST_KEY, false))
// Effective approval-bypass state mirrored from the gateway (session.info).
// Persistence lives in the backend config (approvals.mode), so this is a plain
// reflection of the truth the gateway reports rather than its own store.
export const $yoloActive = atom(false)
export const $currentCwd = atom(getRememberedWorkspaceCwd())
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

export const setConnection = (next: Updater<HermesConnection | null>) => updateAtom($connection, next)
export const setGatewayState = (next: Updater<string>) => updateAtom($gatewayState, next)

export const setSessions = (next: Updater<SessionInfo[]>) => {
  const resolved =
    typeof next === 'function' ? (next as (current: SessionInfo[]) => SessionInfo[])($sessions.get()) : next

  resolved.forEach(rememberSessionLineage)
  $sessions.set(resolved)
}

export const setSessionsTotal = (next: Updater<number>) => updateAtom($sessionsTotal, next)
export const setCronSessions = (next: Updater<SessionInfo[]>) => updateAtom($cronSessions, next)
export const setMessagingSessions = (next: Updater<SessionInfo[]>) => updateAtom($messagingSessions, next)
export const setMessagingPlatformTotals = (next: Updater<Record<string, number>>) =>
  updateAtom($messagingPlatformTotals, next)
export const setMessagingTruncated = (next: Updater<boolean>) => updateAtom($messagingTruncated, next)
export const setSessionProfileTotals = (next: Updater<Record<string, number>>) =>
  updateAtom($sessionProfileTotals, next)
export const setSessionsLoading = (next: Updater<boolean>) => updateAtom($sessionsLoading, next)

export const setWorkingSessionIds = (next: Updater<string[]>) => {
  const resolved =
    typeof next === 'function' ? (next as (current: string[]) => string[])($workingSessionIds.get()) : next

  const retained = new Set(resolved)
  const currentProfiles = $workingSessionProfiles.get()

  const nextProfiles = Object.fromEntries(
    Object.entries(currentProfiles).filter(([sessionId]) => retained.has(sessionId))
  )

  $workingSessionProfiles.set(nextProfiles)
  $workingSessionIds.set(resolved)
}
export const setActiveSessionId = (next: Updater<string | null>) => updateAtom($activeSessionId, next)
export const setActiveSessionStoredIdRotation = (next: Updater<ActiveSessionStoredIdRotation | null>) =>
  updateAtom($activeSessionStoredIdRotation, next)

// Transient: a background session finished and the user hasn't opened it since.
// Written by terminal message producers and cleared once the transcript is
// visibly acknowledged.
export const $unreadFinishedSessionIds = atom<string[]>([])
let publishingUnreadEntries = false
const unreadEntries = new Map<string, SessionUnreadEntry>()

export const setUnreadFinishedSessionIds = (next: Updater<string[]>) => {
  const resolved =
    typeof next === 'function' ? (next as (current: string[]) => string[])($unreadFinishedSessionIds.get()) : next

  const retained = new Set(resolved)

  for (const key of unreadEntries.keys()) {
    if (!retained.has(sessionIdFromScopeKey(key))) {
      unreadEntries.delete(key)
    }
  }

  $unreadFinishedSessionIds.set(resolved)
}

$unreadFinishedSessionIds.listen(ids => {
  if (publishingUnreadEntries) {
    return
  }

  const retained = new Set(ids)

  for (const key of unreadEntries.keys()) {
    if (!retained.has(sessionIdFromScopeKey(key))) {
      unreadEntries.delete(key)
    }
  }
})
let unreadResetVersion: null | SessionUnreadVersion = null
let fallbackCompletionGeneration = Date.now() * 1_000

const INITIAL_UNREAD_EPOCH: SessionUnreadVersion = { revision: 0, source: '' }

function currentUnreadEpoch(): SessionUnreadVersion {
  return unreadResetVersion ?? INITIAL_UNREAD_EPOCH
}

export function createSessionCompletionToken(
  id: string,
  generation: number = ++fallbackCompletionGeneration
): SessionCompletionToken {
  return { epoch: { ...currentUnreadEpoch() }, generation, id }
}

export const setSelectedStoredSessionId = (next: Updater<string | null>) => updateAtom($selectedStoredSessionId, next)

// Stored session ids with a blocking prompt (clarify / approval / sudo /
// secret) waiting on the user.
// Separate from $workingSessionIds: a session can be "working" (turn running)
// AND need input. The sidebar row reads this for a persistent indicator that,
// unlike a toast, survives window blur / alt-tab.
export const $attentionSessionIds = atom<string[]>([])
const attentionSessionScopes = new Set<string>()

export const setAttentionSessionIds = (next: Updater<string[]>) => {
  const resolved =
    typeof next === 'function' ? (next as (current: string[]) => string[])($attentionSessionIds.get()) : next

  const retained = new Set(resolved)

  for (const scope of attentionSessionScopes) {
    if (!retained.has(sessionIdFromScopeKey(scope))) {
      attentionSessionScopes.delete(scope)
    }
  }

  $attentionSessionIds.set(resolved)
}

$attentionSessionIds.listen(ids => {
  const retained = new Set(ids)

  for (const scope of attentionSessionScopes) {
    if (!retained.has(sessionIdFromScopeKey(scope))) {
      attentionSessionScopes.delete(scope)
    }
  }
})

export const getAttentionSessionScopeKeys = (): string[] => {
  const scoped = [...attentionSessionScopes]

  for (const sessionId of $attentionSessionIds.get()) {
    if (scoped.some(scope => sessionIdFromScopeKey(scope) === sessionId)) {
      continue
    }

    for (const session of $sessions.get()) {
      if (sessionMatchesStoredId(session, sessionId)) {
        scoped.push(sessionScopeKey(session.profile, sessionId))
      }
    }
  }

  return [...new Set(scoped)]
}

export function sessionNeedsInput(sessionId: string, profile?: null | string): boolean {
  const root = sessionLineageRootId(sessionId, profile)
  const key = sessionScopeKey(profile, root)

  if (attentionSessionScopes.has(key)) {
    return true
  }

  return (
    $attentionSessionIds.get().includes(root) &&
    ![...attentionSessionScopes].some(scope => sessionIdFromScopeKey(scope) === root)
  )
}

export function setSessionAttention(
  sessionId: string | null | undefined,
  needsInput: boolean,
  profile?: null | string
) {
  if (!sessionId) {
    return
  }

  const root = sessionLineageRootId(sessionId, profile)
  const key = sessionScopeKey(profile, root)

  if (needsInput) {
    attentionSessionScopes.add(key)
  } else if (profile === undefined) {
    for (const scope of attentionSessionScopes) {
      if (sessionIdFromScopeKey(scope) === root) {
        attentionSessionScopes.delete(scope)
      }
    }
  } else {
    attentionSessionScopes.delete(key)
  }

  const activeIds = [...new Set([...attentionSessionScopes].map(sessionIdFromScopeKey))]
  setAttentionSessionIds(activeIds)
}

function compareSessionCompletionTokens(left: SessionCompletionToken, right: SessionCompletionToken): number {
  const epoch = compareSessionUnreadVersions(left.epoch, right.epoch)

  if (epoch !== 0) {
    return epoch
  }

  if (left.generation !== right.generation) {
    return left.generation - right.generation
  }

  return left.id.localeCompare(right.id)
}

function publishSessionUnreadEntries(): void {
  const unread = [
    ...new Set([...unreadEntries.values()].filter(entry => !entry.acknowledged).map(entry => entry.sessionId))
  ]

  // Publish even when raw ids are value-identical: profile ownership can have
  // changed underneath a cloned session id and scoped consumers must rerun.
  publishingUnreadEntries = true

  try {
    $unreadFinishedSessionIds.set(unread)
  } finally {
    publishingUnreadEntries = false
  }
}

function applySessionUnreadReset(version: SessionUnreadVersion): boolean {
  if (unreadResetVersion && compareSessionUnreadVersions(version, unreadResetVersion) <= 0) {
    return false
  }

  unreadResetVersion = version

  for (const [key, entry] of unreadEntries) {
    if (compareSessionUnreadVersions(entry.completion.epoch, version) < 0) {
      unreadEntries.delete(key)
    }
  }

  publishSessionUnreadEntries()

  return true
}

function mergeExactSessionCompletion(previous: SessionUnreadEntry, incoming: SessionUnreadEntry): SessionUnreadEntry {
  const completion =
    compareSessionCompletionTokens(incoming.completion, previous.completion) > 0
      ? incoming.completion
      : previous.completion

  return {
    acknowledged: previous.acknowledged || incoming.acknowledged,
    completion,
    profile: incoming.profile,
    sessionId: incoming.sessionId
  }
}

function applySessionUnreadEntry(incoming: SessionUnreadEntry): boolean {
  const profile = normalizeSessionProfile(incoming.profile)
  const sessionId = sessionLineageRootId(incoming.sessionId, profile)
  const key = sessionScopeKey(profile, sessionId)
  const entry = { ...incoming, profile, sessionId }
  const epoch = compareSessionUnreadVersions(entry.completion.epoch, currentUnreadEpoch())

  if (epoch < 0) {
    return false
  }

  if (epoch > 0) {
    applySessionUnreadReset(entry.completion.epoch)
  }

  const previous = unreadEntries.get(key)
  let next: SessionUnreadEntry = entry

  if (previous) {
    if (previous.completion.id === entry.completion.id) {
      next = mergeExactSessionCompletion(previous, entry)
    } else if (compareSessionCompletionTokens(entry.completion, previous.completion) <= 0) {
      return false
    }
  }

  if (
    previous &&
    previous.acknowledged === next.acknowledged &&
    previous.completion.id === next.completion.id &&
    compareSessionCompletionTokens(previous.completion, next.completion) === 0
  ) {
    return false
  }

  unreadEntries.set(key, next)
  publishSessionUnreadEntries()

  return true
}

function migrateUnreadAliases(profile: string, root: string, aliases: ReadonlySet<string>): void {
  const rootKey = sessionScopeKey(profile, root)
  let merged = unreadEntries.get(rootKey)
  let changed = false

  for (const alias of aliases) {
    if (alias === root) {
      continue
    }

    const aliasKey = sessionScopeKey(profile, alias)
    const entry = unreadEntries.get(aliasKey)

    if (!entry) {
      continue
    }

    unreadEntries.delete(aliasKey)
    changed = true
    const rooted = { ...entry, sessionId: root }

    if (!merged || compareSessionCompletionTokens(rooted.completion, merged.completion) > 0) {
      merged = rooted
    } else if (merged.completion.id === rooted.completion.id) {
      merged = mergeExactSessionCompletion(merged, rooted)
    }
  }

  if (merged) {
    unreadEntries.set(rootKey, merged)
  }

  if (changed) {
    publishSessionUnreadEntries()
  }
}

onSessionUnreadSync({
  onChange: applySessionUnreadEntry,
  onLegacyChange: (sessionId, unread, version) => {
    if (unread) {
      applySessionUnreadEntry({
        acknowledged: false,
        completion: {
          epoch: { ...version },
          generation: version.revision,
          id: `legacy:${version.source}:${version.revision}`
        },
        profile: 'default',
        sessionId
      })
    }
    // Legacy `unread:false` carries no completion identity. Applying it to the
    // current ledger can acknowledge C2 after that old window only saw C1.
  },
  onReset: applySessionUnreadReset,
  onSnapshot: (entries, reset) => {
    if (reset) {
      applySessionUnreadReset(reset)
    }

    entries.forEach(applySessionUnreadEntry)
  },
  onSnapshotRequest: source => {
    broadcastSessionUnreadSnapshot(source, [...unreadEntries.values()], unreadResetVersion)
  }
})
requestSessionUnreadSnapshot()

export function setSessionUnread(sessionId: string | null | undefined, unread: boolean, profile?: null | string) {
  if (!sessionId) {
    return
  }

  const canonical = sessionLineageRootId(sessionId, profile)

  if (unread) {
    if (sessionHasUnread(canonical, profile)) {
      return
    }

    recordSessionCompletion(
      canonical,
      createSessionCompletionToken(`legacy-local:${++fallbackCompletionGeneration}`),
      true,
      profile
    )

    return
  }

  acknowledgeSessionCompletion(canonical, undefined, profile)
}

export function recordSessionCompletion(
  sessionId: string | null | undefined,
  completion: SessionCompletionToken,
  unread: boolean,
  profile?: null | string
): void {
  if (!sessionId) {
    return
  }

  const entry: SessionUnreadEntry = {
    acknowledged: !unread,
    completion,
    profile: normalizeSessionProfile(profile),
    sessionId: sessionLineageRootId(sessionId, profile)
  }

  if (applySessionUnreadEntry(entry)) {
    broadcastSessionUnreadChanged(entry)
  }
}

export function acknowledgeSessionCompletion(
  sessionId: string | null | undefined,
  completion?: null | SessionCompletionToken,
  profile?: null | string
): void {
  if (!sessionId) {
    return
  }

  const normalizedProfile = normalizeSessionProfile(profile)
  const canonical = sessionLineageRootId(sessionId, normalizedProfile)
  const current = unreadEntries.get(sessionScopeKey(normalizedProfile, canonical))
  const target = completion ?? current?.completion

  if (!target) {
    return
  }

  const entry: SessionUnreadEntry = {
    acknowledged: true,
    completion: target,
    profile: normalizedProfile,
    sessionId: canonical
  }

  if (applySessionUnreadEntry(entry)) {
    broadcastSessionUnreadChanged(entry)
  }
}

export function getSessionCompletionToken(
  sessionId: string | null | undefined,
  profile?: null | string
): null | SessionCompletionToken {
  if (!sessionId) {
    return null
  }

  const normalizedProfile = normalizeSessionProfile(profile)
  const root = sessionLineageRootId(sessionId, normalizedProfile)

  return unreadEntries.get(sessionScopeKey(normalizedProfile, root))?.completion ?? null
}

export const getUnreadSessionScopeKeys = (): string[] => {
  const scoped = [...unreadEntries.entries()].filter(([, entry]) => !entry.acknowledged).map(([key]) => key)

  for (const sessionId of $unreadFinishedSessionIds.get()) {
    if (scoped.some(scope => sessionIdFromScopeKey(scope) === sessionId)) {
      continue
    }

    for (const session of $sessions.get()) {
      if (sessionMatchesStoredId(session, sessionId)) {
        scoped.push(sessionScopeKey(session.profile, sessionId))
      }
    }
  }

  return [...new Set(scoped)]
}

export function sessionHasUnread(sessionId: string, profile?: null | string): boolean {
  const normalizedProfile = normalizeSessionProfile(profile)
  const root = sessionLineageRootId(sessionId, normalizedProfile)
  const exact = unreadEntries.get(sessionScopeKey(normalizedProfile, root))

  if (exact) {
    return exact.acknowledged === false
  }

  return (
    $unreadFinishedSessionIds.get().includes(root) &&
    ![...unreadEntries.values()].some(entry => entry.sessionId === root)
  )
}

export interface SessionRenderedCompletion {
  completion: SessionCompletionToken
  profile: string
}

export function getSessionRenderedCompletion(
  sessionId: string,
  profile?: null | string
): null | SessionRenderedCompletion {
  const normalizedProfile = normalizeSessionProfile(profile)
  const root = sessionLineageRootId(sessionId, normalizedProfile)
  const entry = unreadEntries.get(sessionScopeKey(normalizedProfile, root))

  if (!entry || entry.acknowledged) {
    return null
  }

  return {
    completion: entry.completion,
    profile: normalizedProfile
  }
}

export function clearAllSessionUnread(): void {
  const version = nextSessionUnreadVersion()

  applySessionUnreadReset(version)
  broadcastSessionUnreadReset(version)
}

export interface SessionWorkingSnapshotRevision {
  sequence: number
}

function recordSessionWorkingMutation(
  sessionId: string,
  profile: null | string | undefined,
  previousScopes: readonly string[] = []
): void {
  workingSessionMutationRevision += 1

  if (profile === undefined) {
    workingSessionUnscopedRevisions.set(sessionId, workingSessionMutationRevision)

    for (const scope of previousScopes) {
      workingSessionScopeRevisions.set(sessionScopeKey(scope, sessionId), workingSessionMutationRevision)
    }

    return
  }

  workingSessionScopeRevisions.set(sessionScopeKey(profile, sessionId), workingSessionMutationRevision)
}

/** Capture the working ledger revision immediately before requesting an
 * authoritative gateway snapshot. A live event that lands while the request is
 * in flight changes this token, so the older response cannot overwrite it. */
export function sessionWorkingSnapshotRevision(): SessionWorkingSnapshotRevision {
  return { sequence: workingSessionMutationRevision }
}

function sessionWorkingChangedAfter(
  sessionId: string,
  profile: null | string | undefined,
  revision: SessionWorkingSnapshotRevision | undefined
): boolean {
  if (!revision) {
    return false
  }

  return (
    (workingSessionUnscopedRevisions.get(sessionId) ?? 0) > revision.sequence ||
    (workingSessionScopeRevisions.get(sessionScopeKey(profile, sessionId)) ?? 0) > revision.sequence
  )
}

/** Merge one gateway profile from session.active_list without clearing local
 * state. A missing terminal event is healed by the watchdog; pruning here would
 * disarm that recovery while leaving the runtime cache busy. Per-session
 * revisions keep a delayed active snapshot behind newer lifecycle events. */
export function reconcileProfileWorkingSessions(
  profile: null | string | undefined,
  workingSessionIds: readonly string[],
  expectedRevision?: SessionWorkingSnapshotRevision
): boolean {
  const normalized = normalizeSessionProfile(profile)
  const activeIds = [...new Set(workingSessionIds.map(id => id.trim()).filter(Boolean))]

  for (const sessionId of activeIds) {
    if (!sessionWorkingChangedAfter(sessionId, normalized, expectedRevision)) {
      setSessionWorking(sessionId, true, normalized)
    }
  }

  return true
}

export function setSessionWorking(sessionId: string | null | undefined, working: boolean, profile?: null | string) {
  if (!sessionId) {
    return
  }

  const currentProfiles = $workingSessionProfiles.get()
  const previousScopes = currentProfiles[sessionId] ?? []
  const scope = profile === undefined ? UNKNOWN_SESSION_PROFILE_SCOPE : profile?.trim() || 'default'

  const nextScopes = working
    ? [...new Set([...previousScopes, scope])]
    : profile === undefined
      ? []
      : previousScopes.filter(candidate => candidate !== scope)

  $workingSessionProfiles.set(
    nextScopes.length
      ? { ...currentProfiles, [sessionId]: nextScopes }
      : Object.fromEntries(Object.entries(currentProfiles).filter(([id]) => id !== sessionId))
  )
  setWorkingSessionIds(current => {
    const present = current.includes(sessionId)

    // A lifecycle event can be meaningful even when membership is unchanged
    // (e.g. a new turn reasserts true while an old turn's idle snapshot is in
    // flight). Record every assertion so stale snapshots cannot win that race.
    recordSessionWorkingMutation(sessionId, profile, previousScopes)

    if (nextScopes.length > 0) {
      return present ? current : [...current, sessionId]
    }

    return present ? current.filter(id => id !== sessionId) : current
  })
}

export const setMessages = (next: Updater<ChatMessage[]>) => updateAtom($messages, next)
export const setFreshDraftReady = (next: Updater<boolean>) => updateAtom($freshDraftReady, next)
export const setResumeFailedSessionId = (next: Updater<string | null>) => updateAtom($resumeFailedSessionId, next)
export const setResumeExhaustedSessionId = (next: Updater<string | null>) => updateAtom($resumeExhaustedSessionId, next)
export const setBusy = (next: Updater<boolean>) => updateAtom($busy, next)
export const setAwaitingResponse = (next: Updater<boolean>) => updateAtom($awaitingResponse, next)

export const setCurrentModel = (next: Updater<string>) => {
  updateAtom($currentModel, next)
  persistString(COMPOSER_MODEL_KEY, $currentModel.get() || null)
}

export const setCurrentProvider = (next: Updater<string>) => {
  updateAtom($currentProvider, next)
  persistString(COMPOSER_PROVIDER_KEY, $currentProvider.get() || null)
}

export const getCurrentModelSource = (): ComposerModelSource => {
  const source = storedString(COMPOSER_MODEL_SOURCE_KEY)

  return source === 'default' || source === 'manual' ? source : ''
}

// Reactive mirror of the persisted source so UI (the composer pill's
// override badge) can subscribe. The getter above stays storage-backed —
// it's read cross-window, where this atom wouldn't see writes.
export const $currentModelSource = atom<ComposerModelSource>(getCurrentModelSource())

export const setCurrentModelSource = (source: ComposerModelSource) => {
  persistString(COMPOSER_MODEL_SOURCE_KEY, source || null)
  $currentModelSource.set(source)
}

export const setCurrentReasoningEffort = (next: Updater<string>) => {
  updateAtom($currentReasoningEffort, next)
  persistString(COMPOSER_EFFORT_KEY, $currentReasoningEffort.get() || null)
}

export const setCurrentServiceTier = (next: Updater<string>) => updateAtom($currentServiceTier, next)

export const setCurrentFastMode = (next: Updater<boolean>) => {
  updateAtom($currentFastMode, next)
  persistBoolean(COMPOSER_FAST_KEY, $currentFastMode.get())
}

export const setYoloActive = (next: Updater<boolean>) => updateAtom($yoloActive, next)

export const setCurrentCwd = (next: Updater<string>) => {
  updateAtom($currentCwd, next)
  persistString(workspaceCwdKey(), $currentCwd.get().trim() || null)
}

export const setCurrentCwdTransient = (next: Updater<string>) => updateAtom($currentCwd, next)

export const setNewChatWorkspaceTarget = (next: NewChatWorkspaceTarget): number => {
  const generation = $newChatWorkspaceTargetGeneration.get() + 1
  $newChatWorkspaceTarget.set(next)
  $newChatWorkspaceTargetGeneration.set(generation)

  return generation
}

export const workspaceCwdForNewSession = (): string => {
  if ($connection.get()?.mode === 'remote') {
    return getRememberedWorkspaceCwd()
  }

  // A bare new chat starts DETACHED — no inherited cwd, so the composer's coding
  // rail (which keys off $currentCwd) shows no branch and the first message runs
  // in the gateway's default rather than silently in the last repo you touched.
  // Only an explicit default-project-dir setting pre-attaches. Entering a
  // project/worktree attaches its cwd directly (startSessionInWorkspace), so the
  // "remember where I was when I'm in a project" case is unaffected.
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
