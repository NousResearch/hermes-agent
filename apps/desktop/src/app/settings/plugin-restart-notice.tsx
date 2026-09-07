import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import { RefreshCw } from '@/lib/icons'
import { withTimeout } from '@/lib/with-timeout'
import type { GatewayRequest } from '@/store/agent-plugins'
import { reconnectGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

import { Pill } from './primitives'

interface PluginRestartNoticeProps {
  connectionId: string | null
  profile: string
  gatewayLabel: string
  required: boolean | null | undefined
  request: GatewayRequest
  onRecheck: () => Promise<boolean>
}

export function PluginRestartNotice({
  connectionId,
  profile,
  gatewayLabel,
  required,
  request,
  onRecheck
}: PluginRestartNoticeProps) {
  const { t } = useI18n()
  const p = t.settings.plugins.agent
  const [confirming, setConfirming] = useState(false)
  const target = connectionId ? { connectionId, profile } : null

  const capability = useQuery({
    queryKey: ['plugin-restart-capability', connectionId, profile],
    queryFn: () =>
      target && window.hermesDesktop?.backendRestartStatus
        ? withTimeout(window.hermesDesktop.backendRestartStatus(target), 10_000, p.restartUnavailable)
        : Promise.resolve({ supported: false }),
    enabled: required === true,
    retry: false
  })

  const activity = useQuery({
    queryKey: ['plugin-restart-activity', connectionId, profile],
    queryFn: () =>
      withTimeout(request<{ sessions: { status: string }[] }>('session.active_list', {}), 10_000, p.activityUnknown),
    enabled: confirming,
    retry: false
  })

  const count = activity.data?.sessions?.filter(session => session.status !== 'idle').length
  const canRestart = capability.data?.supported === true && Boolean(window.hermesDesktop?.restartBackendFor)

  return (
    <>
      {required == null && (
        <p className="my-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {p.restartUnknown}
        </p>
      )}
      {required === true && (
        <div className="my-3 text-[length:var(--conversation-caption-font-size)]" role="status">
          <div className="flex flex-wrap items-center gap-3">
            <Pill tone="warn">{p.restartRequired}</Pill>
            {canRestart && (
              <Button onClick={() => setConfirming(true)} size="sm" variant="textStrong">
                <RefreshCw />
                {p.restartBackend}
              </Button>
            )}
            <Button
              onClick={() => void onRecheck().catch(error => notifyError(error, p.restartUnverified))}
              size="sm"
              variant="text"
            >
              {p.checkAgain}
            </Button>
          </div>
          <p className="mt-1 text-(--ui-text-secondary)">{p.restartNotice}</p>
          {!capability.isPending && !canRestart && (
            <p className="mt-1 text-(--ui-text-tertiary)">{p.restartInstructions(profile)}</p>
          )}
        </div>
      )}
      <ConfirmDialog
        busyLabel={p.restarting}
        confirmLabel={p.restartConfirm}
        description={
          <>
            {p.restartDescription(profile, gatewayLabel)}
            {count !== undefined && count > 0 && (
              <span className="mt-2 block text-destructive">{p.activeRuns(count)}</span>
            )}
            {activity.isError && <span className="mt-2 block">{p.activityUnknown}</span>}
          </>
        }
        doneLabel={p.restarted}
        onClose={() => setConfirming(false)}
        onConfirm={async () => {
          if (!target || !window.hermesDesktop?.restartBackendFor) {
            throw new Error(p.restartUnavailable)
          }

          const connection = await withTimeout(
            window.hermesDesktop.restartBackendFor(target),
            90_000,
            p.restartUnverified
          )

          await reconnectGatewayForAgent(target.connectionId, target.profile, connection)

          if (!(await onRecheck())) {
            throw new Error(p.restartUnverified)
          }
        }}
        open={confirming}
        title={p.restartTitle(profile)}
      />
    </>
  )
}
