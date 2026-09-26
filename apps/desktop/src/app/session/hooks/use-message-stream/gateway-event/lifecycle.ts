import type { GatewayEvent } from '@hermes/shared'
import type { HermesSkin } from '@hermes/shared/skin'

import {
  notifyCronChanged,
  notifyPairingChanged,
  notifyPetChanged,
  notifyPlatformsChanged,
  notifyProjectsChanged,
  notifySessionsChanged,
  notifySetupReady,
  type PetChangeMeta,
  setChangeEventsAvailable
} from '@/store/live-sync'
import { markRuntimeGone } from '@/store/runtime-gone'
import { $activeSessionId } from '@/store/session'
import {
  $sessionStates,
  dropSessionState,
  publishSessionState,
  unbindTileRuntime
} from '@/store/session-states'
// Leaf import (not the `@/themes` barrel) to avoid pulling the ThemeProvider
// module graph into the gateway event hot path.
import { ingestBackendSkin } from '@/themes/backend-sync'

import type { GatewayEventContext } from './types'

/** gateway.ready / setup.ready / skin.changed / change-watcher broadcasts / session.reclaimed. */
export function handleLifecycleEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, fromActiveSource } = ctx

  if (event.type === 'gateway.ready') {
    const ready = (event as GatewayEvent<'gateway.ready'>).payload
    // Seed the active skin into the desktop theme registry without applying,
    // so a fresh connect never overrides the user's persisted desktop theme.
    ingestBackendSkin(ready?.skin, { apply: false })
    // Backends with the change watcher broadcast pet/cron/sessions change
    // events; consumers demote their legacy polls to slow backstops.
    setChangeEventsAvailable(Boolean(ready?.change_events))

    return true
  }

  if (event.type === 'setup.ready') {
    // The boot bootstrap (hermes_cli/free_tier_bootstrap.py) resolved the
    // free-tier identity and the inference route, and broadcast once. The
    // payload is only a hint — the status snapshot re-reads `setup.status` /
    // `setup.runtime_check` / `free_tier.status` through its own scoped
    // requester so the chip, strip and onboarding react now rather than on
    // the next ambient tick. Only the active source's boot matters here.
    if (fromActiveSource()) {
      notifySetupReady()
    }

    return true
  }

  if (event.type === 'skin.changed') {
    // A runtime skin switch (Hermes activating an authored skin, or `/skin`
    // on another surface). Only the active source+profile's change repaints.
    if (fromActiveSource()) {
      ingestBackendSkin(payload as HermesSkin | undefined, { apply: true })
    }

    return true
  }

  if (
    event.type === 'pet.changed' ||
    event.type === 'cron.changed' ||
    event.type === 'sessions.changed' ||
    event.type === 'projects.changed' ||
    event.type === 'platforms.changed' ||
    event.type === 'pairing.changed'
  ) {
    // Change-watcher broadcasts (server._broadcast_watched_changes): the
    // backend's on-disk signature moved. Route to the live-sync ticks the
    // former pollers now subscribe to. Only the active source+profile's
    // changes apply — background profile sockets (and other connections'
    // gateways) watch their own homes.
    if (fromActiveSource()) {
      if (event.type === 'pet.changed') {
        notifyPetChanged(payload as PetChangeMeta | undefined)
      } else if (event.type === 'cron.changed') {
        notifyCronChanged()
      } else if (event.type === 'projects.changed') {
        notifyProjectsChanged()
      } else if (event.type === 'platforms.changed') {
        notifyPlatformsChanged()
      } else if (event.type === 'pairing.changed') {
        notifyPairingChanged()
      } else {
        notifySessionsChanged()
      }
    }

    return true
  }

  if (event.type === 'session.reclaimed') {
    // The backend reclaimed a live session we may still be holding (idle
    // TTL, LRU cap, or the WS-orphan reap). Without this the runtime id
    // stays cached until something fails against it, which reads as the
    // session vanishing rather than being reclaimed. Drop the cached state
    // now — the stored row is untouched, so the sidebar keeps the
    // conversation and reopening it resumes from the DB.
    const reclaimPayload = payload as { session_id?: string; stored_session_id?: string } | undefined
    const reclaimedRuntimeId = String(reclaimPayload?.session_id ?? '')

    if (reclaimedRuntimeId) {
      // Heal while the cached stored-id mapping is still intact. The active view
      // renders directly from this state slice, so deleting it here makes an
      // already-painted chat flash empty until the explicit durable resume lands.
      const isActiveRuntime = $activeSessionId.get() === reclaimedRuntimeId

      markRuntimeGone(reclaimedRuntimeId, reclaimPayload?.stored_session_id)

      if (isActiveRuntime) {
        // The runtime is dead, so its activity/input claims cannot stay authoritative
        // while the durable resume is in flight. Keep only the visible transcript
        // and cheap metadata; late events for this id remain harmless until the
        // active atom is replaced by the resumed runtime.
        const current = $sessionStates.get()[reclaimedRuntimeId]

        if (current) {
          publishSessionState(reclaimedRuntimeId, {
            ...current,
            awaitingResponse: false,
            busy: false,
            needsInput: false,
            streamId: null,
            turnStartedAt: null,
            turnLive: false
          })
        }
      } else {
        dropSessionState(reclaimedRuntimeId)
      }

      // A tile bound to the reclaimed runtime would otherwise keep pointing at
      // a dead binding. Unbind it so the effect refires against the intact stored
      // session — and purge the wiring cache's entry, or resumeTile's warm path
      // would hand the dead runtime straight back instead of cold-resuming a live one.
      unbindTileRuntime(reclaimedRuntimeId)
      deps.sessionStateByRuntimeIdRef.current.delete(reclaimedRuntimeId)
    }

    // The row's ended_at moved, so refresh the lists that render it.
    notifySessionsChanged()

    return true
  }

  return false
}
