import { useEffect } from 'react'

import { createClientSessionState } from '@/lib/chat-runtime'
import { refreshActiveProfile } from '@/store/profile'
import { $activeSessionId, $currentCwd, setCurrentCwd } from '@/store/session'
import {
  $sessionStates,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS,
  setSessionStalled
} from '@/store/session-states'

import type { GatewayRequester } from '../types'

// Cron sessions are written by a background scheduler tick, messaging turns by
// the background gateway (Telegram, WeChat, Discord, …) — neither signals the
// desktop websocket, so poll the bounded lists while the app is visible.
const CRON_POLL_INTERVAL_MS = 30_000
const MESSAGING_POLL_INTERVAL_MS = 10_000
const ACTIVE_MESSAGING_SESSION_POLL_INTERVAL_MS = 5_000
// Match the TUI's live-session refresh cadence. Auto-compression can rotate a
// stored session id while its turn keeps running; until the next snapshot the
// sidebar row points at the new id while the renderer still knows the old one.
// A 15s cadence made that healthy transition look finished long enough to be
// alarming (and clicking the row appeared to "fix" it by touching the live
// session). This snapshot is small and already polled at 1.5s by the TUI.
const LIVE_SESSION_STATUS_POLL_INTERVAL_MS = 1_500

interface LiveSessionStatusItem {
  id?: string
  last_active?: number
  session_key?: string
  status?: 'idle' | 'starting' | 'waiting' | 'working'
}

interface LiveSessionStatusResponse {
  sessions?: LiveSessionStatusItem[]
}

// Runtime ids this poll has seen live, per gateway profile. A profile only
// ever reaps what its OWN snapshot previously reported: background profiles are
// served by different gateways and never appear in this profile's active_list,
// so an unscoped reap would dark out every other profile's running rows.
const liveRuntimeIdsByProfile = new Map<string, Set<string>>()

/** Restore sidebar liveness after a renderer/backend reconnect. Stream events
 * normally own these states, but events emitted while Desktop was disconnected
 * cannot be replayed. `session.active_list` is the authoritative in-memory
 * snapshot and does not resume, focus, or otherwise mutate a chat.
 *
 * The snapshot is authoritative about ABSENCE too. A turn that ends while the
 * websocket is degraded — a remote gateway over a flaky link, a reconnect, a
 * profile swap — drops out of `_sessions` without Desktop ever seeing the
 * `running: false` edge, so the row keeps spinning and the busy→idle transition
 * that paints the green "your turn" dot never fires. Reaping runtimes that
 * vanish between polls restores both. */
export function rehydrateLiveSessionStatuses(
  response: LiveSessionStatusResponse,
  nowMs = Date.now(),
  profileKey = 'default'
): void {
  const seen = new Set<string>()

  for (const session of response.sessions ?? []) {
    const runtimeSessionId = session.id?.trim()
    const storedSessionId = session.session_key?.trim()
    const needsInput = session.status === 'waiting'
    const working = session.status === 'working' || needsInput

    if (!runtimeSessionId || !storedSessionId) {
      continue
    }

    seen.add(runtimeSessionId)

    const existing = $sessionStates.get()[runtimeSessionId]

    // Avoid re-arming the watchdog on every poll. Publish only when the
    // authoritative live snapshot differs from the renderer mirror; normal
    // gateway events continue to own subsequent transitions.
    if (
      !existing ||
      existing.storedSessionId !== storedSessionId ||
      existing.busy !== working ||
      existing.needsInput !== needsInput
    ) {
      publishSessionState(runtimeSessionId, {
        ...(existing ?? createClientSessionState(storedSessionId)),
        busy: working,
        needsInput,
        storedSessionId
      })
    }

    if (!working) {
      setSessionStalled(storedSessionId, false)

      continue
    }

    const lastActiveMs = Number(session.last_active) * 1000

    const isQuiet =
      session.status === 'working' &&
      Number.isFinite(lastActiveMs) &&
      lastActiveMs > 0 &&
      nowMs - lastActiveMs >= SESSION_WATCHDOG_TIMEOUT_MS

    setSessionStalled(storedSessionId, isQuiet)
  }

  // A runtime this profile's snapshot reported live LAST poll but not this one
  // has ended: the gateway reaps a session out of `_sessions` when its turn
  // completes and its transport goes away. Settle it through the normal publish
  // path so the busy→idle transition fires — that edge is what clears the
  // spinner AND marks the row unread ("your turn"). Only ids this profile
  // previously saw are eligible, so another profile's live rows are untouched.
  const previouslyLive = liveRuntimeIdsByProfile.get(profileKey)

  if (previouslyLive) {
    for (const runtimeSessionId of previouslyLive) {
      if (seen.has(runtimeSessionId)) {
        continue
      }

      const existing = $sessionStates.get()[runtimeSessionId]

      if (existing?.busy || existing?.needsInput) {
        publishSessionState(runtimeSessionId, {
          ...existing,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          streamId: null,
          turnStartedAt: null
        })
      }
    }
  }

  liveRuntimeIdsByProfile.set(profileKey, seen)
}

/** Forget every profile's live-runtime bookkeeping. A gateway wipe already
 *  drops the session states these ids point at, so a carried-over set would
 *  only reap runtimes that no longer exist. */
export function resetLiveRuntimeTracking(): void {
  liveRuntimeIdsByProfile.clear()
}

interface BackgroundSyncParams {
  activeGatewayProfile: string
  activeIsMessaging: boolean
  activeSessionId: null | string
  freshDraftReady: boolean
  gatewayState: string
  refreshActiveMessagingTranscript: () => Promise<unknown> | unknown
  refreshCronJobs: () => Promise<unknown> | unknown
  refreshCurrentModel: (force?: boolean) => Promise<unknown> | unknown
  refreshHermesConfig: () => Promise<unknown> | unknown
  refreshMessagingSessions: () => Promise<unknown> | unknown
  refreshSessions: () => Promise<unknown> | unknown
  requestGateway: GatewayRequester
}

/** Poll a callback while the tab is visible, on `intervalMs`; re-checks on tab
 *  re-focus. Returns nothing — meant to live inside an effect. */
function visiblePoll(intervalMs: number, tick: () => void): () => void {
  const run = () => {
    if (document.visibilityState === 'visible') {
      tick()
    }
  }

  const intervalId = window.setInterval(run, intervalMs)
  document.addEventListener('visibilitychange', run)

  return () => {
    window.clearInterval(intervalId)
    document.removeEventListener('visibilitychange', run)
  }
}

/**
 * Keeps app data live while the gateway is open: an on-connect reseed (model /
 * profile / sessions + relative-cwd resolution), the cron / messaging /
 * open-transcript visibility polls, and the fresh-draft model/config reseed.
 * All the "the desktop websocket won't tell us, so poll" logic in one place.
 */
export function useBackgroundSync({
  activeGatewayProfile,
  activeIsMessaging,
  activeSessionId,
  freshDraftReady,
  gatewayState,
  refreshActiveMessagingTranscript,
  refreshCronJobs,
  refreshCurrentModel,
  refreshHermesConfig,
  refreshMessagingSessions,
  refreshSessions,
  requestGateway
}: BackgroundSyncParams): void {
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    void refreshCurrentModel()
    void refreshActiveProfile()
    void refreshSessions()

    // A RELATIVE workspace cwd (config `terminal.cwd: .`) renders as "." in the
    // file tree header — resolve it to the backend's absolute path once.
    // Session runtime info still overrides later, and never while a session is
    // active.
    const cwd = $currentCwd.get().trim()

    if (!$activeSessionId.get() && cwd && !/^(\/|[A-Za-z]:[\\/])/.test(cwd)) {
      void requestGateway<{ cwd?: string }>('config.get', { key: 'project', cwd })
        .then(info => {
          if (info.cwd && !$activeSessionId.get()) {
            setCurrentCwd(info.cwd)
          }
        })
        .catch(() => undefined)
    }
  }, [gatewayState, refreshCurrentModel, refreshSessions, requestGateway])

  // A reconnect loses renderer-only working/attention atoms while the backend
  // keeps the actual turns alive. Re-seed from the gateway's in-memory session
  // registry immediately, then cheaply poll while visible so a profile switch
  // or missed reconnect edge cannot leave running rows dark until clicked.
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    let cancelled = false
    let inFlight = false

    const refreshLiveStatuses = async () => {
      if (inFlight) {
        return
      }

      inFlight = true

      try {
        const response = await requestGateway<LiveSessionStatusResponse>('session.active_list', {})

        if (!cancelled) {
          rehydrateLiveSessionStatuses(response, Date.now(), activeGatewayProfile)
        }
      } catch {
        // Older gateways may not expose session.active_list. Live stream events
        // still work as before; leave the current sidebar state untouched.
      } finally {
        inFlight = false
      }
    }

    const dispose = visiblePoll(LIVE_SESSION_STATUS_POLL_INTERVAL_MS, () => void refreshLiveStatuses())

    void refreshLiveStatuses()

    return () => {
      cancelled = true
      dispose()
    }
  }, [activeGatewayProfile, gatewayState, requestGateway])

  // Keep the cron-jobs section live without a user action (scheduler ticks in
  // the background); re-check on tab re-focus too.
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    return visiblePoll(CRON_POLL_INTERVAL_MS, () => void refreshCronJobs())
  }, [gatewayState, refreshCronJobs])

  // Keep the messaging-platform session lists live (inbound turns are written
  // by the gateway, not the desktop websocket).
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    return visiblePoll(MESSAGING_POLL_INTERVAL_MS, () => void refreshMessagingSessions())
  }, [gatewayState, refreshMessagingSessions])

  // Only the open messaging transcript needs its own poll — local chats are
  // live over the websocket already.
  useEffect(() => {
    if (gatewayState !== 'open' || !activeIsMessaging) {
      return
    }

    const dispose = visiblePoll(
      ACTIVE_MESSAGING_SESSION_POLL_INTERVAL_MS,
      () => void refreshActiveMessagingTranscript()
    )

    void refreshActiveMessagingTranscript()

    return dispose
  }, [activeIsMessaging, gatewayState, refreshActiveMessagingTranscript])

  // A fresh new-session draft (gateway open, no active session) re-pulls the
  // model + config so the composer pill reflects the profile default.
  useEffect(() => {
    if (gatewayState === 'open' && !activeSessionId && freshDraftReady) {
      void refreshCurrentModel()
      void refreshHermesConfig()
    }
  }, [activeSessionId, freshDraftReady, gatewayState, refreshCurrentModel, refreshHermesConfig])
}
