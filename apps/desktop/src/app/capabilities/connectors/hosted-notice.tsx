import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'

import type { HostedPhase } from './types'

export function HostedNotice({ hasGuest, phase }: { hasGuest: boolean; phase: HostedPhase }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.page

  if (phase === 'signedOut') {
    return (
      <p className="flex shrink-0 items-center gap-1 text-[0.7rem] text-(--ui-text-tertiary)">
        {copy.signInLine}
        <Button onClick={() => openFreeTierSignIn()} size="xs" variant="text">
          {copy.signIn}
        </Button>
      </p>
    )
  }

  if (phase === 'unavailable') {
    return <p className="shrink-0 text-[0.7rem] text-(--ui-text-tertiary)">{copy.managedUnavailable}</p>
  }

  return hasGuest ? <p className="shrink-0 text-[0.7rem] text-(--ui-text-tertiary)">{copy.freeTierNote}</p> : null
}
