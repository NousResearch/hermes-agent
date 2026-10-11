import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Wrench } from '@/lib/icons'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { notifyError } from '@/store/notifications'
import { resetOnboarding } from '@/store/onboarding-gate'

import { ListRow, SettingsSection } from './primitives'

/** Advanced → Developer. Only builds with the guided first run have anything here. */
export function DeveloperSettings() {
  const { t } = useI18n()
  const c = t.settings.config
  const [resetting, setResetting] = useState(false)
  const { requestGateway } = useGatewayRequest()

  if (!isOnboardingEnabled()) {
    return null
  }

  const reset = async () => {
    setResetting(true)

    try {
      await resetOnboarding(requestGateway)
      // A fresh window runs the first run from zero, starting screen included.
      window.location.reload()
    } catch (error) {
      notifyError(error, c.resetOnboardingFailed)
      setResetting(false)
    }
  }

  return (
    <SettingsSection icon={Wrench} title={c.developerTitle}>
      <ListRow
        action={
          <Button disabled={resetting} onClick={() => void reset()} size="sm" variant="outline">
            {c.resetOnboardingAction}
          </Button>
        }
        description={c.resetOnboardingDesc}
        title={c.resetOnboardingTitle}
      />
    </SettingsSection>
  )
}
