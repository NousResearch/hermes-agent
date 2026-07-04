import { atom, computed } from 'nanostores'

import { getProfiles, setApiRequestProfile, STARTUP_REQUEST_TIMEOUT_MS } from '@/hermes'
import {
  arraysEqual,
  persistBoolean,
  persistStringArray,
  persistStringRecord,
  storedBoolean,
  storedStringArray,
  storedStringRecord
} from '@/lib/storage'
import { $gateway, ensureGatewayForProfile } from '@/store/gateway'
import { $attentionSessionIds, $sessions, $workingSessionIds, setConnection } from '@/store/session'
import { resetStarmapGraph } from '@/store/starmap'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

// Canonical key for a profile: trimmed, empty → "default". Used everywhere we
// compare a session's owning profile against the live gateway's profile.
export function normalizeProfileKey(name: string | null | undefined): string {
  const value = (name ?? '').trim()

  return value || 'default'
}

// The profile the running local backend is actually scoped to (mirrors
// /api/profiles/active `current`). "default" is the root ~/.hermes. This is the
// display source of truth for the statusbar pill; the desktop's *stored*
// preference (which may be unset) lives in the Electron main process.
export const $activeProfile = atom<string>('default')

// Cached profile list for the picker. Refreshed lazily; the dropdown also
// re-fetches on open so a profile created elsewhere shows up.
export const $profiles = atom<ProfileInfo[]>([])

export function setActiveProfile(name: string): void {
  $activeProfile.set(name || 'default')
}

export async function refreshProfiles(): Promise<ProfileInfo[]> {
  const { profiles } = await getProfiles()
  $profiles.set(profiles)

  return profiles
}

// ── Rail order ─────────────────────────────────────────────────────────────
// User-defined order for the named (non-default) profile squares in the rail.
// Names absent from the list fall back to alphabetical, appended at the tail —
// so a freshly created profile lands at the end until the user drags it.
const PROFILE_ORDER_STORAGE_KEY = 'hermes.desktop.profileOrder'

export const $profileOrder = atom<string[]>(storedStringArray(PROFILE_ORDER_STORAGE_KEY))

$profileOrder.subscribe(value => persistStringArray(PROFILE_ORDER_STORAGE_KEY, [...value]))

export function setProfileOrder(names: string[]): void {
  if (!arraysEqual($profileOrder.get(), names)) {
    $profileOrder.set(names)
  }
}

// Sort items by the stored order; unordered names alphabetise at the tail.
export function sortByProfileOrder<T extends { name: string }>(items: T[], order: string[]): T[] {
  const rank = new Map(order.map((name, index) => [name, index]))

  return [...items].sort((a, b) => {
    const ra = rank.get(a.name)
    const rb = rank.get(b.name)

    if (ra != null && rb != null) {
      return ra - rb
    }

    return ra != null ? -1 : rb != null ? 1 : a.name.localeCompare(b.name)
  })
}

// ── Rail colors ────────────────────────────────────────────────────────────
// Optional per-profile color override (long-press a rail square to pick). Absent
// names fall back to the deterministic hue from profileColor(); a local-only
// cosmetic preference, so single-profile users never touch it.
const PROFILE_COLORS_STORAGE_KEY = 'hermes.desktop.profileColors'

export const $profileColors = atom<Record<string, string>>(storedStringRecord(PROFILE_COLORS_STORAGE_KEY))

$profileColors.subscribe(value => persistStringRecord(PROFILE_COLORS_STORAGE_KEY, value))

// Set (or, with null, clear) a profile's color override.
export function setProfileColor(name: string, color: null | string): void {
  const key = normalizeProfileKey(name)
  const next = { ...$profileColors.get() }

  if (color) {
    next[key] = color
  } else {
    delete next[key]
  }

  $profileColors.set(next)
}

interface ActiveProfileResponse {
  active: string
  current: string
}

// Pull the running backend's current profile + the available profile list.
// Best-effort: failures (backend not up yet) leave the prior values intact.
export async function refreshActiveProfile(): Promise<void> {
  try {
    const res = await window.hermesDesktop.api<ActiveProfileResponse>({
      path: '/api/profiles/active',
      timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
    })

    setActiveProfile(res.current || 'default')
  } catch {
    // Backend may not be ready; keep the last known value.
  }

  try {
    await refreshProfiles()
  } catch {
    // Leave the cached list in place.
  }
}

// Persist the choice and relaunch the backend under the new HERMES_HOME. The
// main process reloads the window, so this normally never returns to the caller
// (the renderer is torn down). We optimistically reflect the selection first so
// the pill updates instantly if the reload is delayed.
export async function switchProfile(name: string): Promise<void> {
  if (!name || name === $activeProfile.get()) {
    return
  }

  setActiveProfile(name)
  await window.hermesDesktop.profile.set(name)
}

// ── Swap-minimal gateway routing ──────────────────────────────────────────
// One live gateway at a time. When the user opens/sends a session whose profile
// differs from the gateway's current profile, we lazily reconnect the single
// gateway to that profile's backend (spawned on demand by the Electron pool).
// A single-profile user never triggers a swap, so their path is unchanged.

// The profile the live gateway WebSocket is currently connected to. Initialized
// to the primary (window) backend's profile on boot.
export const $activeGatewayProfile = atom<string>('default')

// Profile for the NEXT new chat (chosen via the new-chat picker). null = primary
// / default, so single-profile users are unaffected.
export const $newChatProfile = atom<string | null>(null)

// Bumped whenever the open session should be dropped for a fresh new-session
// draft: a per-profile "+" (newSessionInProfile below), or deleting the project
// that owns the currently-open session (store/projects). The chat controller
// subscribes and resets to the intro draft, so we never strand the user in an
// orphaned view. A plain profile SWITCH no longer routes through here — it
// fires $profileRestoreRequest instead so the controller can resume that
// profile's remembered session (falling back to a fresh draft).
export const $freshSessionRequest = atom(0)

export function requestFreshSession(): void {
  $freshSessionRequest.set($freshSessionRequest.get() + 1)
}

// ── Per-profile last-session memory ────────────────────────────────────────
// Stored id of the last chat each profile had open, so hopping A → B → A lands
// back in A's conversation instead of a blank draft every time. Same idiom as
// the global lastSessionId relaunch memo (store/session), but keyed by profile.
const PROFILE_LAST_SESSIONS_STORAGE_KEY = 'hermes.desktop.profileLastSessions'

export const $profileLastSessions = atom<Record<string, string>>(storedStringRecord(PROFILE_LAST_SESSIONS_STORAGE_KEY))

$profileLastSessions.subscribe(value => persistStringRecord(PROFILE_LAST_SESSIONS_STORAGE_KEY, value))

// Which profile owns a stored session id, per the cross-profile aggregator's
// tag. Falls back to the live gateway profile (a brand-new session created on
// the active backend isn't in the list yet); returns null when the list hasn't
// loaded and the id is unknown, so we never guess-attribute at cold start.
export function profileKeyForSession(sessionId: string): null | string {
  const sessions = $sessions.get()
  const row = sessions.find(session => session.id === sessionId || session._lineage_root_id === sessionId)

  if (row) {
    return normalizeProfileKey(row.profile)
  }

  return sessions.length > 0 ? normalizeProfileKey($activeGatewayProfile.get()) : null
}

/** Memo the open session as its owning profile's "last session". No-ops when
 *  the owner can't be attributed yet (session list not loaded). */
export function rememberProfileSession(sessionId: string): void {
  const key = profileKeyForSession(sessionId)

  if (!key) {
    return
  }

  const current = $profileLastSessions.get()

  if (current[key] !== sessionId) {
    $profileLastSessions.set({ ...current, [key]: sessionId })
  }
}

export function rememberedProfileSession(profile: string): null | string {
  return $profileLastSessions.get()[normalizeProfileKey(profile)] ?? null
}

/** Clear a profile's memo (its remembered session was deleted/archived). */
export function forgetProfileSession(profile: string): void {
  const key = normalizeProfileKey(profile)
  const current = $profileLastSessions.get()

  if (key in current) {
    const next = { ...current }
    delete next[key]
    $profileLastSessions.set(next)
  }
}

/** Drop a dead session id from EVERY profile memo (delete/archive/exhausted
 *  resume) so no profile tries to restore it later. */
export function forgetSessionMemo(sessionId: string): void {
  const current = $profileLastSessions.get()
  const survivors = Object.entries(current).filter(([, id]) => id !== sessionId)

  if (survivors.length !== Object.keys(current).length) {
    $profileLastSessions.set(Object.fromEntries(survivors))
  }
}

/** True when `sessionId` is present, un-archived, in the loaded session list —
 *  the cheap existence check before the by-id REST probe. */
export function isSessionListed(sessions: SessionInfo[], sessionId: string): boolean {
  return sessions.some(
    session => !session.archived && (session.id === sessionId || session._lineage_root_id === sessionId)
  )
}

// A profile switch asks the controller to restore that profile's last session
// (or the explicit `sessionId`, for rail-badge clicks) instead of always
// dropping to a blank draft. Token-guarded so a rapid second switch supersedes
// a restore probe still in flight.
export interface ProfileSessionRestoreRequest {
  profile: string
  /** Explicit target (rail badge click); null → use the per-profile memo. */
  sessionId: null | string
  token: number
}

export const $profileRestoreRequest = atom<null | ProfileSessionRestoreRequest>(null)

let profileRestoreToken = 0

function requestProfileSessionRestore(profile: string, sessionId: null | string): void {
  $profileRestoreRequest.set({ profile, sessionId, token: ++profileRestoreToken })
}

// ── Per-profile activity (working / needs-input badges) ───────────────────
// Sessions come pre-tagged with their owning profile by the cross-profile
// aggregator; bucket the global working/attention sets by that tag so the
// profile rail can badge each square without any extra fetching.

export interface ProfileActivitySession {
  id: string
  lastActive: number
  needsInput: boolean
  title: null | string
}

export interface ProfileActivity {
  /** Sessions blocked on the user (clarify/approval) — the amber badge. */
  attention: ProfileActivitySession[]
  /** Sessions with a turn running that are NOT waiting on the user. */
  working: ProfileActivitySession[]
}

export function deriveProfileActivity(
  sessions: SessionInfo[],
  workingIds: string[],
  attentionIds: string[]
): Record<string, ProfileActivity> {
  const working = new Set(workingIds)
  const attention = new Set(attentionIds)
  const result: Record<string, ProfileActivity> = {}

  for (const session of sessions) {
    const needsInput = attention.has(session.id)

    if (!needsInput && !working.has(session.id)) {
      continue
    }

    // The default profile keys as "default" everywhere in the rail (activeKey,
    // hotkeys), whatever its directory name — mirror that here.
    const key = session.is_default_profile ? 'default' : normalizeProfileKey(session.profile)
    const entry = (result[key] ??= { attention: [], working: [] })

    const row: ProfileActivitySession = {
      id: session.id,
      lastActive: session.last_active,
      needsInput,
      title: session.title
    }

    // "Needs input" wins over "working" (a clarify-blocked session is
    // technically still running) — same precedence as the sidebar row dot.
    ;(needsInput ? entry.attention : entry.working).push(row)
  }

  for (const entry of Object.values(result)) {
    entry.attention.sort((a, b) => b.lastActive - a.lastActive)
    entry.working.sort((a, b) => b.lastActive - a.lastActive)
  }

  return result
}

export const $profileActivity = computed([$sessions, $workingSessionIds, $attentionSessionIds], deriveProfileActivity)

/** The session a badge click should open: blocked-on-you first, else the most
 *  recently active working one. */
export function neediestSessionId(activity: ProfileActivity | undefined): null | string {
  return activity?.attention[0]?.id ?? activity?.working[0]?.id ?? null
}

// Route profile-scoped REST settings (config/env/skills/tools/model/…) to the
// profile the live gateway is currently on. Profile-scoped react-query keys
// carry the profile segment (see activeProfileQueryKey), so a swap needs NO
// cache invalidation: profile B's queries key differently than A's, A's cached
// data can never render under B, and hopping back to A hits A's warm cache.
// Only the starmap (a nanostore, not react-query) still resets by hand.
let _lastRoutedProfile: string | null = null

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)
  setApiRequestProfile(key)

  if (_lastRoutedProfile !== null && _lastRoutedProfile !== key) {
    resetStarmapGraph()
  }

  _lastRoutedProfile = key
})

/** Query-key segment for the profile the live gateway is on RIGHT NOW — for
 *  imperative cache writes/invalidations. Reactive call sites should key off
 *  `normalizeProfileKey(useStore($activeGatewayProfile))` instead so the query
 *  re-keys when the gateway swaps. */
export function activeProfileQueryKey(): string {
  return normalizeProfileKey($activeGatewayProfile.get())
}

// Target profile while a gateway swap is mid-flight (spawning/reconnecting that
// profile's backend), else null. Drives the chat's "waking up <profile>" loader
// so a lazy spawn doesn't read as a hang. Single-profile users never swap.
export const $gatewaySwapTarget = atom<string | null>(null)

let gatewaySwitch: Promise<void> | null = null

// Keep the renderer's $connection (mode / baseUrl / profile) in lockstep with
// the profile the live gateway is now on. $connection seeds from the PRIMARY
// (window) backend at boot and otherwise only refreshes on a sleep/wake
// reconnect — so activating a *background* profile left $connection describing
// the primary, with the wrong `mode` for everything that branches on
// local-vs-remote. Headline symptom: with a local primary and a remote pool
// profile active, image attachments went out via the path-based `image.attach`
// instead of `image.attach_bytes`, handing the remote gateway a client-only
// path it can't resolve ("image not found: C:\…"), while the /api/fs/* file
// browser and /api/media fetches targeted the wrong machine (#46651).
// Best-effort: a failed descriptor fetch leaves the prior connection intact for
// boot/reconnect to resync.
async function syncConnectionToActiveProfile(profile: string): Promise<void> {
  const getConnection = window.hermesDesktop?.getConnection

  if (!getConnection) {
    return
  }

  try {
    setConnection(await getConnection(profile))
  } catch {
    // Leave the prior connection in place; boot/reconnect resyncs it later.
  }
}

// Make `profile`'s backend the active gateway, lazily opening its socket if it
// isn't live yet. Unlike the old single-socket swap, background profiles keep
// their sockets — so their sessions keep streaming concurrently. A null/empty
// target means "no explicit profile" → keep the current gateway (a plain new
// chat stays put; single-profile users never leave the primary).
export async function ensureGatewayProfile(profile: string | null | undefined): Promise<void> {
  if (profile == null || !String(profile).trim()) {
    // "No explicit profile" = use the current gateway. But if an explicit swap
    // (e.g. the user just picked a profile in the switcher) is still in flight,
    // let it settle first so a new chat doesn't race session.create against a
    // half-open socket and land on the wrong backend.
    if (gatewaySwitch) {
      await gatewaySwitch.catch(() => undefined)
    }

    return
  }

  const target = normalizeProfileKey(profile)

  if (normalizeProfileKey($activeGatewayProfile.get()) === target && $gateway.get()) {
    return
  }

  // Serialize concurrent activations so two rapid session switches don't race
  // the active pointer.
  if (gatewaySwitch) {
    await gatewaySwitch.catch(() => undefined)

    if (normalizeProfileKey($activeGatewayProfile.get()) === target && $gateway.get()) {
      return
    }
  }

  $gatewaySwapTarget.set(target)
  gatewaySwitch = (async () => {
    // ensureGatewayForProfile opens (or reuses) the target's socket and points
    // the active gateway at it — without closing the profile you came from.
    await ensureGatewayForProfile(target)
    $activeGatewayProfile.set(target)
    // The active backend just changed; resync $connection so remote-aware
    // paths (image.attach_bytes vs image.attach, /api/fs/*, /api/media) follow.
    await syncConnectionToActiveProfile(target)
  })()

  try {
    await gatewaySwitch
  } finally {
    gatewaySwitch = null
    $gatewaySwapTarget.set(null)
  }
}

// ── Sidebar profile scope (the "workspace switcher" model) ─────────────────
// Mirrors how Slack/VS Code/Linear do multi-context: you're "in" one profile at
// a time and the sidebar shows only that profile's sessions (clean rows, no
// per-row tags). The lone exception is an explicit "All profiles" mode that
// fans every profile's sessions into one grouped, browsable list.

export const ALL_PROFILES = '__all__'

const SHOW_ALL_PROFILES_STORAGE_KEY = 'hermes.desktop.showAllProfiles'

// Opt-in unified view. When false, scope follows the live gateway profile, so
// single-profile users (who never see the switcher) are completely unaffected.
export const $showAllProfiles = atom<boolean>(storedBoolean(SHOW_ALL_PROFILES_STORAGE_KEY, false))

$showAllProfiles.subscribe(value => persistBoolean(SHOW_ALL_PROFILES_STORAGE_KEY, value))

// The profile context the sidebar is currently showing: a concrete profile key,
// or ALL_PROFILES for the unified grouped view. Concrete scope is tied to the
// gateway so opening/selecting a profile (which swaps the gateway) moves the
// whole sidebar with it — a real context switch, not a separate filter to keep
// in sync.
export const $profileScope = computed([$showAllProfiles, $activeGatewayProfile], (showAll, gateway) =>
  showAll ? ALL_PROFILES : normalizeProfileKey(gateway)
)

// Switch the active context to `name`: leave "All profiles" mode, point new
// chats at it, and swap the single live gateway onto its backend (which moves
// $activeGatewayProfile → name, so $profileScope follows).
export function selectProfile(name: string): void {
  const target = normalizeProfileKey(name)
  // Switching profiles (or coming back from the all-profiles browse view)
  // restores that profile's last open session — fresh draft only when there's
  // nothing to restore; re-tapping the profile you're already in leaves your
  // session be.
  const switching = $showAllProfiles.get() || target !== normalizeProfileKey($activeGatewayProfile.get())
  $showAllProfiles.set(false)
  $newChatProfile.set(target)

  if (switching) {
    requestProfileSessionRestore(target, null)
  }

  void ensureGatewayProfile(target)
}

// Switch to `name` AND open a specific session — the profile-rail badge click,
// which jumps straight to the session that needs you. Same context switch as
// selectProfile, but the restore targets the given session instead of the memo.
export function selectProfileSession(name: string, sessionId: string): void {
  const target = normalizeProfileKey(name)
  $showAllProfiles.set(false)
  $newChatProfile.set(target)
  requestProfileSessionRestore(target, sessionId)
  void ensureGatewayProfile(target)
}

// Start a fresh session in `name` WITHOUT collapsing the "All profiles" browse
// view. Unlike selectProfile, it leaves $showAllProfiles untouched, so the
// unified sidebar stays put — used by the per-profile "+" in the all-profiles
// session list, where switching scope would throw away the browse state the user
// is in. Points new chats at the profile and opens its backend so the next
// message lands in the right place.
export function newSessionInProfile(name: string): void {
  const target = normalizeProfileKey(name)
  $newChatProfile.set(target)
  requestFreshSession()
  void ensureGatewayProfile(target)
}

export function setShowAllProfiles(value: boolean): void {
  $showAllProfiles.set(value)
}

export function toggleShowAllProfiles(): void {
  $showAllProfiles.set(!$showAllProfiles.get())
}

// ── Hotkey-driven profile switching ────────────────────────────────────────
// Positional + relative navigation for the rail, used by the keybind runtime.
// The ordered list is [default, ...named-in-rail-order]; switching is a no-op
// when the slot is empty so unused ⌘N keys stay harmless.

function orderedProfileKeys(): string[] {
  const profiles = $profiles.get()

  const named = sortByProfileOrder(
    profiles.filter(profile => !profile.is_default),
    $profileOrder.get()
  ).map(profile => normalizeProfileKey(profile.name))

  const hasDefault = profiles.some(profile => profile.is_default)

  return hasDefault ? ['default', ...named] : named
}

// Switch to the default (root ~/.hermes) profile — bound to ⌘1.
export function switchToDefaultProfile(): void {
  const def = $profiles.get().find(profile => profile.is_default)

  selectProfile(def ? def.name : 'default')
}

// Switch to the Nth named (non-default) profile in rail order (1-based).
export function switchProfileToSlot(slot: number): void {
  const named = sortByProfileOrder(
    $profiles.get().filter(profile => !profile.is_default),
    $profileOrder.get()
  )

  const target = named[slot - 1]

  if (target) {
    selectProfile(target.name)
  }
}

// Step to the next/previous profile in the rail, wrapping around.
export function cycleProfile(direction: 1 | -1): void {
  const keys = orderedProfileKeys()

  if (keys.length < 2) {
    return
  }

  const current = $showAllProfiles.get() ? -1 : keys.indexOf(normalizeProfileKey($activeGatewayProfile.get()))
  const start = current < 0 ? (direction === 1 ? -1 : 0) : current
  const next = (start + direction + keys.length) % keys.length

  selectProfile(keys[next])
}

// Bumped to ask the rail to open its "create profile" dialog (the dialog state
// is local to the rail component; this lets a global hotkey trigger it).
export const $profileCreateRequest = atom(0)

export function requestProfileCreate(): void {
  $profileCreateRequest.set($profileCreateRequest.get() + 1)
}

// Keepalive ping for the active pool backend so the main-process idle reaper
// (which can't see the direct renderer↔backend WS) spares it. No-op for the
// primary/default backend, which is never pooled.
export function touchActiveGatewayBackend(): void {
  // Always ping: the main process no-ops for non-pool (primary) backends, so we
  // don't need to know which profile is primary from here.
  const target = normalizeProfileKey($activeGatewayProfile.get())
  void window.hermesDesktop?.touchBackend?.(target).catch(() => undefined)
}
