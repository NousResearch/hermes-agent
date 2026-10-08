import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { $questionnaireAvailable, runSetupAgain } from '@/onboarding/due'
import { notifyError } from '@/store/notifications'

import { ListRow } from './primitives'

/** Settings → Run setup again: opens the first-run questionnaire now, no reload (D1). Its answers go to the default profile (D17). */
export function RunSetupAgainSetting() {
  const { t } = useI18n()
  const copy = t.questionnaire.settings
  const available = useStore($questionnaireAvailable)
  const [opening, setOpening] = useState(false)
  const { requestGateway } = useGatewayRequest()

  if (!available) {
    return null
  }

  const open = async () => {
    setOpening(true)

    try {
      await runSetupAgain(requestGateway)
    } catch (error) {
      notifyError(error, copy.failed)
    } finally {
      setOpening(false)
    }
  }

  return (
    <ListRow
      action={
        <Button disabled={opening} onClick={() => void open()} size="sm" variant="outline">
          {copy.action}
        </Button>
      }
      description={copy.description}
      title={copy.title}
    />
  )
}
