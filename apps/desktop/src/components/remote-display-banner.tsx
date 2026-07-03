import { useEffect, useState } from 'react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { Info } from '@/lib/icons'

export function RemoteDisplayBanner() {
  const { t } = useI18n()
  const [reason, setReason] = useState<string | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    void window.hermesDesktop?.getRemoteDisplayReason?.().then(result => setReason(result))
  }, [])

  if (!reason || dismissed) {
    return null
  }

  return (
    <div className="pointer-events-none fixed left-1/2 top-[calc(var(--titlebar-height,34px)+0.75rem)] z-[200] w-[min(32rem,calc(100%-2rem))] -translate-x-1/2">
      <Alert className="pointer-events-auto grid-cols-[auto_minmax(0,1fr)_auto] border-(--stroke-nous) bg-popover/95 pr-2.5 shadow-nous backdrop-blur-md">
        <Info className="text-muted-foreground" />
        <AlertDescription className="col-start-2">
          <p className="m-0">{t.remoteDisplayBanner.message(reason)}</p>
        </AlertDescription>
        <Button
          aria-label={t.remoteDisplayBanner.dismiss}
          className="col-start-3 -mr-1 text-muted-foreground"
          onClick={() => setDismissed(true)}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Codicon name="close" size="0.875rem" />
        </Button>
      </Alert>
    </div>
  )
}
