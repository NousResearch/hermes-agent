import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $freeTierSignInOpen } from '@/store/free-tier'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'

/** The redesign has no statusbar, so its rail carries the free tier's standing Sign in (the
 *  statusbar chip's job) in every interface mode. Same state, same one sign-in dialog. */
export function PortalFreeTierSignIn() {
  const { t } = useI18n()
  const open = useStore($freeTierSignInOpen)

  if (!open) {
    return null
  }

  return (
    <Tip label={t.freeTier.signIn}>
      <Button aria-label={t.freeTier.signIn} onClick={() => openFreeTierSignIn()} size="icon" variant="ghost">
        <Codicon name="account" />
      </Button>
    </Tip>
  )
}
