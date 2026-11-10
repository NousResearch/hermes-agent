import { useStore } from '@nanostores/react'

import { useSessionView } from '@/app/chat/session-view'
import { composerFloatingPill } from '@/components/chat/composer-dock'
import { useOnboardingChatActive } from '@/components/onboarding/first-run/assembly'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $gatewaySwapTarget, $profileColors } from '@/store/profile'
import { $profileSwitchPending } from '@/store/profile-switch-state'

/**
 * While a profile switch is in flight the composer stays usable: a send waits
 * for the swap (ensureGatewayProfile) and lands on the target. This pill says
 * so, on the primary surface only — tiles keep their own session's backend.
 * Same suppression as the switch indicator during first-run onboarding.
 */
export function SendTargetHint() {
  const { t } = useI18n()
  const primary = useSessionView().kind === 'primary'
  const swapTarget = useStore($gatewaySwapTarget)
  // A registry-source pick waits behind the agent dial before the swap target
  // is published; the send still waits for it and lands on the pick.
  const pendingPick = useStore($profileSwitchPending)
  const target = swapTarget ?? pendingPick
  const colors = useStore($profileColors)
  const onboarding = useOnboardingChatActive()

  if (!primary || !target || onboarding) {
    return null
  }

  return (
    // Informational, not a control — but it keeps pointer events: the pop-out
    // drag region sits behind the strip (see micro-actions.tsx).
    <span
      className={cn(composerFloatingPill, 'cursor-default hover:bg-(--composer-fill) hover:text-(--ui-text-secondary)')}
      data-slot="composer-send-target"
    >
      <ProfileGlyph
        aria-hidden="true"
        color={resolveProfileColor(target, colors)}
        isDefault={target === 'default'}
        name={target}
      />
      {t.composer.willSendTo(target)}
    </span>
  )
}
