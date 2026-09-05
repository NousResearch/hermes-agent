import { atom } from 'nanostores'

// Event-driven "backend data changed" signals — the workspace-events twin for
// gateway-side state. The change watcher in tui_gateway broadcasts
// `pet.changed` / `cron.changed` / `sessions.changed` when its on-disk
// signatures move (see server._broadcast_watched_changes), the gateway-event
// lifecycle handler routes them here, and the surfaces that used to poll (cron
// sidebar/page, messaging lists, pet sprite, live session statuses) subscribe
// to these ticks and refresh — so they move exactly when the backend acts and
// stay idle otherwise.
//
// `$changeEventsAvailable` is the compat gate: gateway.ready advertises
// `change_events: true` on backends that broadcast. Consumers keep their old
// polls as SLOW backstops when it's true and at the legacy cadence when it's
// false (an older backend), so nothing goes dark mid-upgrade.

export const $changeEventsAvailable = atom(false)

export const $cronChangeTick = atom(0)
export const $sessionsChangeTick = atom(0)
export const $platformsChangeTick = atom(0)
export const $pairingChangeTick = atom(0)

/** `pet.info.meta`-shaped payload carried on `pet.changed` — lets the pet skip
 *  the heavy sprite refetch when the broadcast already says enabled=false. */
export interface PetChangeMeta {
  enabled: boolean
  slug?: string
  displayName?: string
  scale?: number
  spritesheetRevision?: string
}

export const $petChange = atom<{ meta?: PetChangeMeta; tick: number }>({ tick: 0 })

export function setChangeEventsAvailable(available: boolean): void {
  $changeEventsAvailable.set(available)
}

export function notifyPetChanged(meta?: PetChangeMeta): void {
  $petChange.set({ meta, tick: $petChange.get().tick + 1 })
}

export function notifyCronChanged(): void {
  $cronChangeTick.set($cronChangeTick.get() + 1)
}

export function notifySessionsChanged(): void {
  $sessionsChangeTick.set($sessionsChangeTick.get() + 1)
}

export function notifyPlatformsChanged(): void {
  $platformsChangeTick.set($platformsChangeTick.get() + 1)
}

export function notifyPairingChanged(): void {
  $pairingChangeTick.set($pairingChangeTick.get() + 1)
}

/** Reset on gateway wipe/reconnect — a new backend re-advertises capability on
 *  its own gateway.ready, and stale ticks must not fire refreshes into stores
 *  the wipe just cleared. */
export function resetLiveSync(): void {
  $changeEventsAvailable.set(false)
}
