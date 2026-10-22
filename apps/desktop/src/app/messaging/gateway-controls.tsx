import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import type { Translations } from '@/i18n/types'
import { AlertTriangle, RefreshCw } from '@/lib/icons'

interface GatewayControlsProps {
  copy: Translations['messaging']
  busy: boolean
  stopped: boolean
  restartNeeded: boolean
  onStart: () => Promise<void>
  onRestart: () => Promise<void>
}

export function GatewayControls({ copy, busy, stopped, restartNeeded, onStart, onRestart }: GatewayControlsProps) {
  if (!restartNeeded && !stopped) {
    return null
  }

  const action = stopped ? onStart : onRestart

  const label = busy
    ? stopped
      ? copy.startingMessagingGateway
      : copy.restarting
    : stopped
      ? copy.startMessagingGateway
      : copy.restartNow

  return (
    <Alert variant="warning">
      <AlertTriangle />
      <AlertDescription className="flex flex-wrap items-center justify-between gap-2">
        <span>{restartNeeded ? copy.restartNeeded : copy.gatewayStopped}</span>
        <Button disabled={busy} onClick={() => void action()} size="sm" variant="secondary">
          <RefreshCw className={busy ? 'animate-spin' : undefined} />
          {label}
        </Button>
      </AlertDescription>
    </Alert>
  )
}
