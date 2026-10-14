import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { skipIntro } from '@/components/onboarding/first-run/intro'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { skipOnboarding } from '@/store/onboarding-run'
import { onboardingChatStore } from '@/store/onboarding-session'

/** Skip setup, shown in the composer of the onboarding chat only. Once the backend has ended the run,
 *  like main's: the intro ends on the previous layout at normal size and the tutorial tips stop. */
export function OnboardingSkip({ sessionId }: { sessionId: null | string }) {
  const onboardingChat = useStore(useMemo(() => onboardingChatStore(sessionId), [sessionId]))

  // The chat's current runtime id: a resume after a reconnect gives it a new one.
  return sessionId && onboardingChat ? <SkipButton runtimeId={sessionId} /> : null
}

// Split out so the gateway hook mounts only in the onboarding chat, not in every composer.
function SkipButton({ runtimeId }: { runtimeId: string }) {
  const { t } = useI18n()
  const { requestGateway } = useGatewayRequest()

  return (
    <Tip label={t.firstRun.skipTip}>
      <button
        className="ml-auto text-[11px] text-(--ui-text-quaternary) transition-colors hover:text-(--ui-text-secondary)"
        onClick={() =>
          void skipOnboarding(requestGateway, runtimeId).then(skipIntro, error =>
            notifyError(error, t.firstRun.skipFailed)
          )
        }
        type="button"
      >
        {t.firstRun.skip}
      </button>
    </Tip>
  )
}
