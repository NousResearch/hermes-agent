import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { firstRunIntro } from '@/components/onboarding/first-run/kickoff'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $onboardingChatOpener, startOnboarding } from '@/store/onboarding-run'
import { startOnboarding as startSetupGuide } from '@/store/setup-guide'
import { usePortalShell } from '@/store/shell-variant'

import { ListRow } from './primitives'

/** Advanced → Run onboarding again. */
export function OnboardingSettings() {
  const { t } = useI18n()
  const copy = t.firstRun.rerun
  const redesign = usePortalShell()
  const { requestGateway } = useGatewayRequest()
  const [starting, setStarting] = useState(false)

  const rerun = async () => {
    setStarting(true)

    try {
      const open = $onboardingChatOpener.get()

      const started = redesign
        ? await startSetupGuide(requestGateway)
        : open !== null && (await startOnboarding(requestGateway, open, { intro: firstRunIntro(), rerun: true }))

      if (!started) {
        notifyError(null, copy.failed)
      }
    } catch (error) {
      notifyError(error, copy.failed)
    } finally {
      setStarting(false)
    }
  }

  return (
    <ListRow
      action={
        <Button disabled={starting} onClick={() => void rerun()} size="sm" variant="outline">
          {copy.action}
        </Button>
      }
      description={redesign ? t.firstRun.guide.intro : copy.description}
      title={copy.title}
    />
  )
}
