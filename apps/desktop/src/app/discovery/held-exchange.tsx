import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { $heldPrompt } from '@/store/held-prompt'
import { usePortalShell } from '@/store/shell-variant'
import { $signInFailed, $signUpRequired, startSignUp } from '@/store/visitor-access'

import { useFirstTimeCopy } from './first-time-copy'

/**
 * A visitor's message held until sign up, and Hermes' answer to it (Figma 1C 462:3935): "I can do
 * that. Sign up and I'll start on it right away." Nothing was sent anywhere; this is not a transcript
 * and holds no chat messages, only the held text (store/held-prompt.ts) laid out like the opening of
 * one. After a sign up that could not sign them in, the answer says so and offers to try again.
 */
interface HeldExchangeProps {
  /** The window's own chat, not a tile. */
  primary: boolean
  /** The chat shown is a stored one, not a new chat. */
  routed: boolean
}

/** Whether a held message stands in this chat: the redesign's new chat, for a visitor who must sign
 *  up and has sent something. It replaces the empty chat's landing. */
function useHeldExchangeShown({ primary, routed }: HeldExchangeProps): boolean {
  const portal = usePortalShell()
  const signUpRequired = useStore($signUpRequired)
  const held = useStore($heldPrompt) !== null

  return portal && primary && !routed && signUpRequired && held
}

/** The new chat's intro and landing stand down while a held message takes their place. */
export function useIntroUnlessHeld(showIntro: boolean, where: HeldExchangeProps): boolean {
  const held = useHeldExchangeShown(where)

  return showIntro && !held
}

export function HeldExchange(where: HeldExchangeProps) {
  const shown = useHeldExchangeShown(where)
  const text = useStore($heldPrompt)
  const failed = useStore($signInFailed)
  const copy = useFirstTimeCopy().held

  if (!shown || !text) {
    return null
  }

  return (
    <div className="discovery-held" data-slot="held-exchange">
      <div className="discovery-held-column">
        <article aria-label={copy.you} className="discovery-held-you">
          {text}
        </article>
        <article aria-label={copy.hermes} aria-live="polite" className="discovery-held-reply">
          <h2 className="discovery-held-title">{failed ? copy.failedTitle : copy.title}</h2>
          <p className="discovery-held-body">{failed ? copy.failedBody : copy.body}</p>
          <Button className="discovery-held-action" onClick={() => startSignUp('chat')}>
            {failed ? copy.tryAgain : copy.signUp}
          </Button>
        </article>
      </div>
    </div>
  )
}
