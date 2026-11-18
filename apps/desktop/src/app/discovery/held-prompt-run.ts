import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $heldPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $onboardingPhase, $setupGuidePhase } from '@/store/onboarding-phase'
import { $gatewayState } from '@/store/session'
import { usePortalShell } from '@/store/shell-variant'
import { $signUpRequired } from '@/store/visitor-access'

import { requestComposerInsert } from '../chat/composer/focus'

/**
 * The message a visitor sent before signing up runs first once they are signed in (build plan 2A): as
 * soon as this window can run anything (signed in, connected, setup off screen), it is taken and sent
 * as a new chat, which opens. Taken before it is sent, so it runs once even across reloads and tabs; a
 * send that fails leaves it in the composer rather than losing it.
 */
export function useHeldPromptRun(submitToNewChat: (text: string) => Promise<unknown>): void {
  const portal = usePortalShell()
  const signUpRequired = useStore($signUpRequired)
  const connected = useStore($gatewayState) === 'open'
  const onboarding = useStore($onboardingPhase)
  const setupGuide = useStore($setupGuidePhase)
  const settingUp = onboarding === 'shown' || setupGuide === 'shown'
  const held = useStore($heldPrompt)

  useEffect(() => {
    if (!portal || signUpRequired || !connected || settingUp || !held) {
      return
    }

    const text = takeHeldPrompt()

    if (text) {
      void submitToNewChat(text).catch(() => requestComposerInsert(text, { target: 'main' }))
    }
  }, [connected, held, portal, settingUp, signUpRequired, submitToNewChat])
}
