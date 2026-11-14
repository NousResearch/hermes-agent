import { useStore } from '@nanostores/react'

import { useSessionView } from '@/app/chat/session-view'
import { composerFloatingPill } from '@/components/chat/composer-dock'
import { useOnboardingChatActive } from '@/components/onboarding/first-run/assembly'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import {
  $activeGatewayProfile,
  $gatewaySwapTarget,
  $newChatProfile,
  $newChatRoute,
  $profileColors,
  normalizeProfileKey
} from '@/store/profile'
import { $profileSwitchPending } from '@/store/profile-switch-state'
import { knownOwnerForSession } from '@/store/session-states'

/** Activity comes from the switch; its assurance comes from Send's owner. */
export function useSwitchSendTarget(): null | string {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const swapTarget = useStore($gatewaySwapTarget)
  const pendingPick = useStore($profileSwitchPending)
  const draftRoute = useStore($newChatRoute)
  const draftProfile = useStore($newChatProfile)
  const activeProfile = useStore($activeGatewayProfile)

  if (view.kind !== 'primary' || (!swapTarget && !pendingPick)) {
    return null
  }

  if (runtimeId || storedId) {
    const owner = knownOwnerForSession(runtimeId) ?? knownOwnerForSession(storedId)

    // An unresolved existing session must never borrow the new-chat pin.
    return typeof owner === 'string' ? owner : (owner?.targetProfile || owner?.profile) ?? null
  }

  return draftRoute?.targetProfile || draftRoute?.profile || draftProfile || normalizeProfileKey(activeProfile)
}

/**
 * While a profile switch is in flight, name the destination Send captures,
 * not the older activation that may still hold the gateway mutex. Tiles keep
 * their own backend and first-run onboarding suppresses switch chrome.
 */
export function SendTargetHint() {
  const { t } = useI18n()
  const target = useSwitchSendTarget()
  const colors = useStore($profileColors)
  const onboarding = useOnboardingChatActive()

  if (!target || onboarding) {
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
