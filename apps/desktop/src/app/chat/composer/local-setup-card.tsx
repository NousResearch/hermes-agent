import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import {
  $localSetupEligibility,
  $localSetupOffer,
  dismissLocalSetupOffer,
  readLocalSetupEligibility,
  requestLocalSetupMenu
} from '@/store/local-setup-offer'
import { isHudWindow } from '@/store/windows'

import { useComposerScope } from './scope'

interface LocalSetupCardProps {
  busy: boolean
  guidedChat: boolean
}

/**
 * "This could run on your computer" strip above the input, after the first
 * finished turn once onboarding is over (store/local-setup-offer.ts). The offer
 * is about the machine, not the chat, so it sits in the main window's primary
 * composer (not tiles, not the HUD) and survives a relaunch until x or Show me.
 * Only while that session is idle: an automatic follow-up turn hides it, the
 * next idle brings it back. Offers, never hijacks: "Show me" opens this
 * composer's model menu on click, nothing moves by itself.
 */
export function LocalSetupCard({ busy, guidedChat }: LocalSetupCardProps) {
  const primary = useSessionView().kind === 'primary'
  const { target } = useComposerScope()
  const offer = useStore($localSetupOffer)
  const eligibility = useStore($localSetupEligibility)
  const fit = eligibility?.fit
  const { t } = useI18n()
  const copy = t.composer.localSetup

  // No answer yet (relaunch, or the backend just changed) or a failed one: ask again.
  useEffect(() => {
    if (offer.state === 'shown' && (!eligibility || eligibility.transient)) {
      void readLocalSetupEligibility()
    }
  }, [offer.state, eligibility])

  if (offer.state !== 'shown' || !fit || !primary || busy || guidedChat || isHudWindow()) {
    return null
  }

  return (
    <div
      className="flex items-center justify-between gap-2 rounded-lg border border-[color-mix(in_srgb,var(--dt-composer-ring)_32%,transparent)] bg-accent/18 px-2 py-1.5"
      data-slot="composer-local-setup"
      role="status"
    >
      <div className="flex min-w-0 items-center gap-2">
        <Codicon className="shrink-0 text-(--ui-accent)" name="chip" size="0.85rem" />
        <div className="min-w-0 text-[0.72rem] leading-snug">
          <div className="font-medium text-foreground">{copy.title}</div>
          <div className="truncate text-muted-foreground/88">{copy.text(fit.model.display_name)}</div>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <Button onClick={() => requestLocalSetupMenu(target)} size="xs" type="button" variant="secondary">
          {copy.action}
        </Button>
        <Button
          aria-label={t.common.close}
          onClick={dismissLocalSetupOffer}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Codicon name="close" size="0.7rem" />
        </Button>
      </div>
    </div>
  )
}
