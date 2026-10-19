import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { firstRunIntro } from '@/components/onboarding/first-run/kickoff'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $onboardingChatOpener, startOnboarding } from '@/store/onboarding-run'

import { ListRow } from './primitives'

/** Advanced → Run onboarding again. */
export function OnboardingSettings() {
  const copy = useI18n().t.firstRun.rerun
  const { requestGateway } = useGatewayRequest()
  const [starting, setStarting] = useState(false)

  const rerun = async () => {
    setStarting(true)

    // The whole first-run experience again: the boot overlay, the onboarding-sized window, the demo
    // layout and the intro, in a new chat opened the way the launch opens it.
    const open = $onboardingChatOpener.get()

    const started =
      open !== null && (await startOnboarding(requestGateway, open, { intro: firstRunIntro(), rerun: true }))

    setStarting(false)

    if (!started) {
      notifyError(null, copy.failed)
    }
  }

  return (
    <ListRow
      action={
        <Button disabled={starting} onClick={() => void rerun()} size="sm" variant="outline">
          {copy.action}
        </Button>
      }
      description={copy.description}
      title={copy.title}
    />
  )
}
