import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { $desktopMetricsGate, recordFeatureUse } from '@/store/desktop-metrics'
import { $onboardingSurfaceClear } from '@/store/onboarding-presence'
import { $gatewayState } from '@/store/session'
import { $agentSidebarChoice, setSimplifiedAgentMode } from '@/store/simplified-agent-mode'

export function SimplifiedAgentOffer() {
  const { t } = useI18n()
  const choice = useStore($agentSidebarChoice)
  const clear = useStore($onboardingSurfaceClear)
  const gateway = useStore($gatewayState)
  const copy = t.simplifiedAgents
  const metricsGate = useStore($desktopMetricsGate)
  const open = choice === null && clear && gateway === 'open'

  useEffect(() => {
    if (open && metricsGate === 'on') {
      recordFeatureUse('simplified_sidebar_offer')
    }
  }, [open, metricsGate])

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {
          setSimplifiedAgentMode(false, 'dismiss')
        }
      }}
      open={open}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{copy.offerTitle}</DialogTitle>
          <DialogDescription>{copy.offerDescription}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button onClick={() => setSimplifiedAgentMode(false, 'offer')} variant="secondary">
            {copy.no}
          </Button>
          <Button onClick={() => setSimplifiedAgentMode(true, 'offer')}>{copy.yes}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
