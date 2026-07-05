import { atom, computed } from 'nanostores'

import { BOOT_AGGREGATE_REQUEST_TIMEOUT_MS, getProfiles, setApiRequestProfile } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import {
  arraysEqual,
  persistBoolean,
  persistString,
  persistStringArray,
  persistStringRecord,
  storedBoolean,
  storedString,
  storedStringArray,
  storedStringRecord
} from '@/lib/storage'
import { $gateway, ensureGatewayForProfile } from '@/store/gateway'
import { setConnection } from '@/store/session'
import { resetStarmapGraph } from '@/store/starmap'
import type { ProfileInfo } from '@/types/hermes'

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
const PROFILES_CACHE_STORAGE_KEY = 'hermes.desktop.profiles.cache.v1'

function profileInfoFromCache(value: unknown): ProfileInfo | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const item = value as Partial<ProfileInfo>

  if (typeof item.name !== 'string' || !item.name.trim() || typeof item.path !== 'string') {
    return null
  }

  return {
    has_env: Boolean(item.has_env),
    is_default: Boolean(item.is_default),
    model: typeof item.model === 'string' ? item.model : null,
    name: item.name,
    path: item.path,
    provider: typeof item.provider === 'string' ? item.provider : null,
    skill_count: typeof item.skill_count === 'number' && Number.isFinite(item.skill_count) ? item.skill_count : 0
  }
}

function storedProfiles(): ProfileInfo[] {
  try {
    const raw = storedString(PROFILES_CACHE_STORAGE_KEY)

    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw)

    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.map(profileInfoFromCache).filter((profile): profile is ProfileInfo => profile !== null)
  } catch {
    return []
  }
}

function persistProfiles(profiles: ProfileInfo[]): void {
  persistString(PROFILES_CACHE_STORAGE_KEY, profiles.length > 0 ? JSON.stringify(profiles) : null)
}

export const $profiles = atom<ProfileInfo[]>(storedProfiles())

function setProfiles(profiles: ProfileInfo[]): void {
  $profiles.set(profiles)
  persistProfiles(profiles)
}

export function setActiveProfile(name: string): void {
  $activeProfile.set(name || 'default')
}

export async function refreshProfiles(): Promise<ProfileInfo[]> {
  const { profiles } = await getProfiles()
  setProfiles(profiles)

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

// ── Rail icons ─────────────────────────────────────────────────────────────
// Optional per-profile emoji shown on the rail square instead of the first
// letter. Local-only cosmetic preference, mirrors $profileColors.
const PROFILE_ICONS_STORAGE_KEY = 'hermes.desktop.profileIcons'

export const $profileIcons = atom<Record<string, string>>(storedStringRecord(PROFILE_ICONS_STORAGE_KEY))

$profileIcons.subscribe(value => persistStringRecord(PROFILE_ICONS_STORAGE_KEY, value))

// Set (or, with null, clear) a profile's emoji/icon override.
export function setProfileIcon(name: string, icon: null | string): void {
  const key = normalizeProfileKey(name)
  const next = { ...$profileIcons.get() }

  if (icon) {
    next[key] = icon
  } else {
    delete next[key]
  }

  $profileIcons.set(next)
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
      timeoutMs: BOOT_AGGREGATE_REQUEST_TIMEOUT_MS
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

// The profile the user is looking at in the sidebar/rail. This is deliberately
// separate from $activeGatewayProfile: the visible context should switch as
// soon as the user clicks a profile, while the gateway may still be waking that
// backend in the background.
export const $selectedProfileScope = atom<string>('default')

let preferredProfileScope: string | null = null

// Profile for the NEXT new chat (chosen via the new-chat picker). null = primary
// / default, so single-profile users are unaffected.
export const $newChatProfile = atom<string | null>(null)

// Bumped whenever the open session should be dropped for a fresh new-session
// draft: a profile switch/create (below), or deleting the project that owns the
// currently-open session (store/projects). The chat controller subscribes and
// resets to the intro draft, so we never strand the user in an orphaned view.
export const $freshSessionRequest = atom(0)

export function requestFreshSession(): void {
  $freshSessionRequest.set($freshSessionRequest.get() + 1)
}

// Route profile-scoped REST settings (config/env/skills/tools/model/…) to the
// profile the live gateway is currently on, and drop cached settings from the
// previous profile so pages refetch against the right backend. Fires once
// immediately (no real change → no invalidation), so single-profile users just
// get "default" (→ the primary backend) with no extra fetches.
let _lastRoutedProfile: string | null = null

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)
  setApiRequestProfile(key)

  if (preferredProfileScope === null || preferredProfileScope === key) {
    $selectedProfileScope.set(key)

    if (preferredProfileScope === key) {
      preferredProfileScope = null
    }
  }

  if (_lastRoutedProfile !== null && _lastRoutedProfile !== key) {
    // Profile-scoped settings + the unified session list are now stale.
    void queryClient.invalidateQueries()
    resetStarmapGraph()
  }

  _lastRoutedProfile = key
})

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
export const $profileScope = computed([$showAllProfiles, $selectedProfileScope], (showAll, selected) =>
  showAll ? ALL_PROFILES : normalizeProfileKey(selected)
)

// Switch the active context to `name`: leave "All profiles" mode, point new
// chats at it, update the visible sidebar scope immediately, and swap the live
// gateway onto its backend in the background.
export function selectProfile(name: string): void {
  const target = normalizeProfileKey(name)
  // Switching profiles (or coming back from the all-profiles browse view) starts
  // fresh; re-tapping the profile you're already in leaves your session be.
  const switching = $showAllProfiles.get() || target !== normalizeProfileKey($selectedProfileScope.get())
  preferredProfileScope = target
  $selectedProfileScope.set(target)
  $showAllProfiles.set(false)
  $newChatProfile.set(target)

  if (switching) {
    requestFreshSession()
  }

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

  const current = $showAllProfiles.get() ? -1 : keys.indexOf(normalizeProfileKey($selectedProfileScope.get()))
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
