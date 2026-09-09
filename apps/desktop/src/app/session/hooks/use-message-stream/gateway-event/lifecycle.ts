import type { GatewayEvent } from '@hermes/shared'
import type { HermesSkin } from '@hermes/shared/skin'

import { clearClarifyRequest } from '@/store/clarify'
import { invalidateContextBreakdown } from '@/app/shell/hooks/use-context-breakdown'
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
import { clearAllPrompts, clearApprovalRequest } from '@/store/prompts'
import { markRuntimeGone } from '@/store/runtime-gone'
import { dropSessionState, unbindTileRuntime } from '@/store/session-states'
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

  if (event.type === 'approval.cancelled') {
    // The backend dropped pending approvals for a session being interrupted or
    // torn down (#106678) — the deny-resolve is otherwise silent, so a parked
    // prompt card would keep offering Approve/Reject against an approval that
    // no longer exists (the backend answers resolved: 0 and the click looks
    // dead). Clear the parked prompts; the turn's BLOCKED tool result is the
    // in-transcript signal, same as the timeout path.
    const cancelled = (event as GatewayEvent<'approval.cancelled'>).payload
    const runtimeId = String(cancelled?.session_id ?? '')
    const requestIds = (cancelled?.request_ids ?? []).map(id => String(id)).filter(Boolean)

    if (runtimeId && requestIds.length > 0) {
      // A request-id mismatch is a no-op in clear(), so a cancelled id can
      // never wipe a newer prompt re-armed by a live turn on the same session.
      for (const requestId of requestIds) {
        clearApprovalRequest(runtimeId, requestId)
      }
    } else if (runtimeId) {
      // No correlation ids on the wire — drop the session's prompt wholesale.
      clearAllPrompts(runtimeId)
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
    const reclaimedRuntimeId = String((payload as { session_id?: string } | undefined)?.session_id ?? '')

    // The compression/reclaim lifecycle invalidates the keyed context
    // breakdown so the statusbar gauge refetches instead of serving the
    // pre-compression figure (#94001). The breakdown cache keys on the
    // STORED id; the reclaim payload carries it alongside the runtime id.
    const reclaimedStoredId = String(
      (payload as { stored_session_id?: string } | undefined)?.stored_session_id ?? ''
    )

    if (reclaimedRuntimeId) {
      // Heal while the cached stored-id mapping is still intact, then drop.
      markRuntimeGone(reclaimedRuntimeId)
      invalidateContextBreakdown(reclaimedStoredId || reclaimedRuntimeId)
      dropSessionState(reclaimedRuntimeId)
      // A prompt keyed to the dead runtime must not outlive it. The runtime id
      // rotates on every resume (cold/lazy/eager all mint a fresh sid), so the
      // new runtime's turn-end clears can never remove an entry keyed to THIS
      // one — a stale approval would re-mount the floating "needs approval"
      // bar whenever the reclaimed conversation is reopened (#86577).
      clearAllPrompts(reclaimedRuntimeId)
      clearClarifyRequest(undefined, reclaimedRuntimeId)
      // A tile bound to the reclaimed runtime would otherwise render an
      // empty transcript forever: its view reads $sessionStates[runtime]
      // (just dropped) and its resume effect is gated on !runtimeId, so a
      // bound tile never re-resumes (#82620). Unbind it so the effect
      // refires against the intact stored session — and purge the wiring
      // cache's entry, or resumeTile's warm path would hand the dead
      // runtime straight back instead of cold-resuming a live one.
      unbindTileRuntime(reclaimedRuntimeId)
      deps.sessionStateByRuntimeIdRef.current.delete(reclaimedRuntimeId)
    }

    // The row's ended_at moved, so refresh the lists that render it.
    notifySessionsChanged()

    return true
  }

  return false
}
