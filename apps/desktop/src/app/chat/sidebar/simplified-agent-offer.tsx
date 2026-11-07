import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { $tourActive } from '@/lib/tour/tour-active'
import { $desktopMetricsGate, recordFeatureUse } from '@/store/desktop-metrics'
import { $desktopOnboarding } from '@/store/onboarding'
import { $onboardingSurfaceClear } from '@/store/onboarding-presence'
import { $gatewayState } from '@/store/session'
import { $agentSidebarChoice, setSimplifiedAgentMode } from '@/store/simplified-agent-mode'

export function SimplifiedAgentOffer() {
  const { t } = useI18n()
  const choice = useStore($agentSidebarChoice)
  const clear = useStore($onboardingSurfaceClear)
  const gateway = useStore($gatewayState)
  const copy = t.simplifiedAgents
  const onboarding = useStore($desktopOnboarding)
  const tourActive = useStore($tourActive)
  const metricsGate = useStore($desktopMetricsGate)
  const open =
    choice === null && clear && gateway === 'open' && !onboarding.manual && !onboarding.freeTierReady && !tourActive

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
      <DialogContent className="max-w-md" showCloseButton={false}>
        <div
          aria-hidden
          className="grid size-10 place-items-center rounded-lg bg-(--ui-bg-tertiary) text-(--ui-text-primary)"
        >
          <Codicon name="layout-sidebar-left" size={24} />
        </div>
        <DialogHeader>
          <DialogTitle>{copy.offerTitle}</DialogTitle>
          <DialogDescription>{copy.offerDescription}</DialogDescription>
        </DialogHeader>
        <div className="space-y-2 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)/40 p-3 text-xs">
          <p className="text-(--ui-text-secondary)">{copy.offerRevert}</p>
          <p className="font-medium leading-relaxed">
            {[
              t.commandCenter.settings,
              t.settings.sections.appearance,
              t.settings.subpages.appearanceWindowLayout,
              copy.title
            ].join(' → ')}
          </p>
        </div>
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
