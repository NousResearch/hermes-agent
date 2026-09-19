import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { $gateway } from '@/store/gateway'
import { type InboxAutomationAction, type InboxRequest, refreshInbox, runInboxAutomationAction } from '@/store/inbox'
import { $activeGatewayProfile } from '@/store/profile'

interface AutomationControlsProps {
  kind: 'goal' | 'heartbeat' | 'loop'
  liveSessionId: string
  onChanged?: () => void
  sessionKey: string
  /** ``null`` until request details resolve, then whether the session has a live runtime. */
  sessionLive: boolean | null
  status: string
}

/**
 * Pause / Resume for the session's goal, loop or heartbeat — the same two actions the
 * composer's status cards offer, reachable from the panel. Edit and destructive actions
 * (clear / stop) stay in the chat where their confirm flows live.
 *
 * Works for a stored session too: pause/resume are persisted-state writes, so the gateway
 * runs them against the session key when the session is not running (no live runtime needed).
 */
export function AutomationControls({ kind, liveSessionId, onChanged, sessionKey, sessionLive, status }: AutomationControlsProps) {
  const [busy, setBusy] = useState<InboxAutomationAction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pinnedGateway = useRef($gateway.get())
  const pinnedProfile = useRef($activeGatewayProfile.get() ?? '')

  const canPause = status === 'active'
  const canResume = status === 'paused'

  if (!canPause && !canResume) {
    return null
  }

  const verb: 'pause' | 'resume' = canPause ? 'pause' : 'resume'
  const action = `${kind}.${verb}` as InboxAutomationAction

  const run = async () => {
    const currentGateway = $gateway.get()
    const currentProfile = $activeGatewayProfile.get() ?? ''

    if (!currentGateway || pinnedGateway.current !== currentGateway || pinnedProfile.current !== currentProfile) {
      setError('Profile changed — re-open to act')

      return
    }

    setBusy(action)
    setError(null)

    try {
      const boundRequest: InboxRequest = (method, params) => currentGateway.request(method, params ?? {})

      await runInboxAutomationAction({ action, liveSessionId, profile: currentProfile, request: boundRequest, sessionKey })

      // The control's own state lives in the inbox LIST item, not the request details: pull the
      // fresh snapshot so the button flips as soon as the write is confirmed. Without this the
      // flip waits for the background poll (up to 15s), which reads as a failed click.
      try {
        await refreshInbox(currentProfile, boundRequest)
      } catch {
        /* the action landed; the background poll will refresh the list */
      }

      onChanged?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="mt-1 flex flex-col gap-1">
      <div className="flex items-center gap-1.5">
        <Button disabled={busy !== null} onClick={() => void run()} size="xs" variant="secondary">
          {busy ? 'Working…' : canPause ? `Pause ${kind}` : `Resume ${kind}`}
        </Button>
        {sessionLive === false && (
          <span className="text-[0.6rem] text-muted-foreground/60">session isn't running — applies to stored state</span>
        )}
      </div>
      {error && (
        <p className="text-[0.62rem] text-destructive" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
