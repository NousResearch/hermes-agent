import { useEffect } from 'react'

import { refreshActiveProfile } from '@/store/profile'
import { $activeSessionId, $currentCwd, setCurrentCwd } from '@/store/session'

import type { GatewayRequester } from '../types'

// Cron sessions are written by a background scheduler tick, messaging turns by
// the background gateway (Telegram, WeChat, Discord, …) — neither signals the
// desktop websocket, so poll the bounded lists while the app is visible.
const CRON_POLL_INTERVAL_MS = 30_000
const MESSAGING_POLL_INTERVAL_MS = 10_000
const ACTIVE_MESSAGING_SESSION_POLL_INTERVAL_MS = 5_000

interface BackgroundSyncParams {
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
