import { atom, batch, computed } from 'nanostores'

import type { HermesConnection } from '@/global'
import { getProfiles, hermesApi, setApiRequestProfile, STARTUP_REQUEST_TIMEOUT_MS } from '@/hermes'
import { invalidateProfileScopedQueries } from '@/lib/query-client'
import {
  arraysEqual,
  persistBoolean,
  persistStringArray,
  persistStringRecord,
  storedBoolean,
  storedStringArray,
  storedStringRecord
} from '@/lib/storage'
import { invalidateCronModelImpactScopeState } from '@/store/cron-model-impact-scope'
import { $gateway, ensureGatewayForAgent, ensureGatewayForProfile, openGatewayForProfile } from '@/store/gateway'
import { setConnection } from '@/store/session'
import { resetStarmapGraph } from '@/store/starmap'
import type { ProfileInfo } from '@/types/hermes'

// Canonical key for a profile: trimmed, empty → "default". Used everywhere we
// compare a session's owning profile against the live gateway's profile.
export function normalizeProfileKey(name: string | null | undefined): string {
  const value = (name ?? '').trim()

  return value || 'default'
}

// Presentation-only label: the display_name from profile.yaml when set (e.g. a
// renamed default profile), else the canonical name. Never used for
// comparison or routing — canonical `name` remains the identity everywhere.
export function profileLabel(profile: Pick<ProfileInfo, 'display_name' | 'name'>): string {
  return (profile.display_name ?? '').trim() || profile.name
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

// ── Stale-fetch invalidation across backend switches ───────────────────────
// $profiles mirrors the ACTIVE backend's /api/profiles. A connection/mode
// apply (the soft re-home) or a profile/agent activation changes which backend
// that is while a fetch may still be in flight — and a late response from the
// PREVIOUS backend must not clobber the list the new backend just served.
// That was #85731's disappearing rail: applying a different remote/Cloud
// connection let the old (often dying, profile-less) backend's response land
// last, collapsing $profiles and hiding the rail. Bumping the epoch strands
// every in-flight fetch: the response still resolves for its caller, but it
// no longer writes the shared cache ("guard against the past").
let profileListEpoch = 0

export function invalidateProfileListFetches(): void {
  profileListEpoch += 1
}

export async function refreshProfiles(): Promise<ProfileInfo[]> {
  const epoch = profileListEpoch
  const { profiles } = await getProfiles()

  if (epoch === profileListEpoch) {
    $profiles.set(profiles)
  }

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
  const epoch = profileListEpoch

  try {
    const res = await hermesApi<ActiveProfileResponse>({
      path: '/api/profiles/active',
      timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
    })

    // Same stale-response guard as refreshProfiles: a backend switch mid-fetch
    // means this answer describes the PREVIOUS backend.
    if (epoch === profileListEpoch) {
      setActiveProfile(res.current || 'default')
    }
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
// to the primary (window) backend's profile on boot. The gateway registry
// mirrors its own route into this atom via the onActiveRouteChanged callback
// (wired in use-gateway-boot's configureGatewayRegistry), so registry-internal
// eviction fallbacks (idle reap, connection removal, profile delete) can never
// leave this naming a profile the active socket no longer serves (#89206).
export const $activeGatewayProfile = atom<string>('default')

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

  if (_lastRoutedProfile !== null && _lastRoutedProfile !== key) {
    invalidateCronModelImpactScopeState()
    // Profile-scoped settings + the unified session list are now stale.
    // Narrowed so account/marketplace/onboarding caches don't refetch on
    // every profile switch.
    invalidateProfileScopedQueries()
    resetStarmapGraph()
    // /api/profiles now routes to a different backend: strand any in-flight
    // profile-list fetch so the previous backend's late answer can't clobber
    // the rail (the #85731 class — same guard as the connection-apply wipe).
    invalidateProfileListFetches()
  }

  _lastRoutedProfile = key
})

// Target profile while a gateway swap is mid-flight (spawning/reconnecting that
// profile's backend), else null. Drives the chat's "waking up <profile>" loader
// so a lazy spawn doesn't read as a hang. Single-profile users never swap.
export const $gatewaySwapTarget = atom<string | null>(null)

// ── Hover-intent backend pre-warm ───────────────────────────────────────────
// A cold switch to a profile whose pool backend isn't running pays the full
// spawn (Python boot + port announce + readiness probe — measured ~2.5-3s)
// plus the socket connect before the sidebar can repopulate. The pointer
// entering a profile square in the rail signals the switch a few hundred ms
// before the click lands, so we run the same spawn + connect chain then
// (openGatewayForProfile — without activating). `ensureBackend` in the
// Electron main is idempotent (a pooled profile returns its existing
// connectionPromise), so the real switch joins the in-flight work instead of
// duplicating it — and a pre-warm for an already-open profile is a no-op.
// Throttled per profile so drive-by hovers can't spam spawn attempts; failures
// stay silent here and surface on the real switch, which owns retry/error UX.
const PREWARM_MIN_INTERVAL_MS = 60_000

const prewarmedAt = new Map<string, number>()

export function prewarmProfileBackend(name: string): void {
  const key = normalizeProfileKey(name)

  if (key === normalizeProfileKey($activeGatewayProfile.get())) {
    return
  }

  const now = Date.now()

  if (now - (prewarmedAt.get(key) ?? 0) < PREWARM_MIN_INTERVAL_MS) {
    return
  }

  prewarmedAt.set(key, now)
  openGatewayForProfile(key).catch(() => undefined)
}

let gatewaySwitch: Promise<void> | null = null

// The target profile's connection descriptor (mode / baseUrl / …), resolved
// CONCURRENTLY with the socket work so the switch can publish the profile
// pointer and $connection in one frame. Without this, $connection seeds from
// the PRIMARY backend at boot and only refreshes on sleep/wake — activating a
// *background* profile left it describing the primary, with the wrong `mode`
// for everything that branches on local-vs-remote (#46651: path-based
// `image.attach` against a remote gateway, /api/fs/* and /api/media on the
// wrong machine).
//
// Best-effort BY DESIGN (fail open): a failed lookup resolves null, the prior
// descriptor stays, and boot/reconnect resyncs it later. The earlier
// atomic-publish series (#89483) failed the whole switch closed here instead,
// and its decline path turned routine registry churn into dead profile
// clicks (#89622) — reverted in #89785. Do not reintroduce fail-closed
// switching at this seam.
async function resolveConnectionForProfile(profile: string): Promise<HermesConnection | null> {
  const getConnection = window.hermesDesktop?.getConnection

  if (!getConnection) {
    return null
  }

  try {
    return await getConnection(profile)
  } catch (err) {
    console.warn(`[profile] descriptor lookup for "${profile}" failed; keeping the previous connection`, err)

    return null
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
    // The descriptor resolves concurrently so nothing awaits between the
    // activation and the publication below: the old post-activation
    // syncConnectionToActiveProfile await left a window where $gateway
    // already targeted the new backend while $connection still described the
    // previous one, and remote-aware paths announced the wrong mode (#46651).
    const [connection] = await Promise.all([resolveConnectionForProfile(target), ensureGatewayForProfile(target)])

    // ONE publication frame. batch() defers Nanostores' notifications to the
    // end of the callback, so the profile pointer and the connection
    // descriptor become visible together; a null descriptor (no bridge, or a
    // failed best-effort lookup) keeps the previous one — fail open.
    batch(() => {
      $activeGatewayProfile.set(target)

      if (connection) {
        setConnection(connection)
      }
    })
  })()

  try {
    await gatewaySwitch
  } finally {
    gatewaySwitch = null
    $gatewaySwapTarget.set(null)
  }
}

// Registry-aware sibling of syncConnectionToActiveProfile: a connection-scoped
// agent's descriptor comes from getConnectionFor (its SOURCE connection), not
// getConnection (the local pool). Same best-effort, fail-open contract as
// resolveConnectionForProfile: a failed lookup resolves null and keeps the
// previous descriptor.
async function resolveConnectionForAgent(connectionId: string, profile: string): Promise<HermesConnection | null> {
  const getConnectionFor = window.hermesDesktop?.getConnectionFor

  if (!getConnectionFor) {
    return null
  }

  try {
    return await getConnectionFor({ connectionId, profile })
  } catch (err) {
    console.warn(
      `[profile] descriptor lookup for agent "${connectionId}:${profile}" failed; keeping the previous connection`,
      err
    )

    return null
  }
}

// Activate a connection-scoped agent's gateway — the (connectionId, profile)
// analogue of ensureGatewayProfile, and the door the SDK's ensureAgent goes
// through. Two invariants the raw store call (ensureGatewayForAgent) does not
// provide on its own:
//  - Every activation moves $activeGatewayProfile and resyncs $connection,
//    exactly like the profile path — otherwise activating an ALREADY-OPEN
//    registry agent left both describing the previous backend, routing
//    /api/fs, /api/media and image.attach to the wrong machine (the same
//    class as #46651) and pointing newSessionInProfile at the stale profile.
//  - Activations share the gatewaySwitch mutex with profile switches, so a
//    rapid agent↔profile (or agent↔agent) interleave can't finish out of
//    order and leave the EARLIER setActive() as the last write.
// Only a null connectionId falls through to the legacy profile path. Explicit
// `local` is a registry identity and must use the genuinely-local route.
export async function ensureGatewayAgent(connectionId: null | string, profile: string): Promise<void> {
  const target = normalizeProfileKey(profile)
  const connection = (connectionId ?? '').trim() || null

  if (!connection) {
    return ensureGatewayProfile(target)
  }

  // Serialize against any in-flight profile/agent switch (shared mutex).
  if (gatewaySwitch) {
    await gatewaySwitch.catch(() => undefined)
  }

  $gatewaySwapTarget.set(target)
  gatewaySwitch = (async () => {
    // Descriptor resolves concurrently with the dial, same as the profile
    // path, so no await sits between the activation and the publication.
    const [descriptor, activated] = await Promise.all([
      resolveConnectionForAgent(connection, target),
      ensureGatewayForAgent(connection, target)
    ])

    if (!activated) {
      // The target stopped existing mid-dial (source edited/removed). Keep
      // every atom on the previous backend; the caller's surfaces re-check
      // what's active. Log so a dead agent click is diagnosable (#89622's
      // silence lesson) — but never fail the whole switch closed here.
      console.warn(`[profile] agent gateway activation for "${connection}:${target}" did not land`)

      return
    }

    // ONE publication frame, profile pointer + descriptor together. A null
    // descriptor keeps the previous one — fail open, resynced by
    // boot/reconnect later.
    batch(() => {
      $activeGatewayProfile.set(target)

      if (descriptor) {
        setConnection(descriptor)
      }
    })
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

/** Normalize a sidebar scope to the profile key used by session and cron queries. */
export const sidebarProfileForScope = (profileScope: string): string =>
  profileScope === ALL_PROFILES ? 'all' : normalizeProfileKey(profileScope)

/** Key a platform total by its Desktop profile route so counts cannot leak across profiles. */
export const messagingTotalsKey = (messagingProfile: string, sourceId: string): string =>
  `${messagingProfile}:${sourceId}`

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
  // Switching profiles (or coming back from the all-profiles browse view) starts
  // fresh; re-tapping the profile you're already in leaves your session be.
  const switching = $showAllProfiles.get() || target !== normalizeProfileKey($activeGatewayProfile.get())
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
