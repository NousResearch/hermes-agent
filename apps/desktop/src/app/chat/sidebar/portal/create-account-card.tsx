import './create-account-card.css'

import { useStore } from '@nanostores/react'

import { useFirstTimeCopy } from '@/app/discovery/first-time-copy'
import { Codicon } from '@/components/ui/codicon'
import { $signUpRequired, startSignUp } from '@/store/visitor-access'

/**
 * "Create an Account" at the foot of the chat sidebar (Figma 18695:18439), for as long as sign up is
 * required: the whole card is the way to sign up. It takes the foot before anything else that offers
 * itself there (the usage-stats card waits for it).
 */
export function CreateAccountCard() {
  const copy = useFirstTimeCopy().account
  const signUpRequired = useStore($signUpRequired)

  if (!signUpRequired) {
    return null
  }

  return (
    <button
      className="portal-account-offer"
      data-slot="create-account-offer"
      onClick={() => startSignUp('sidebar')}
      type="button"
    >
      <span className="portal-account-offer-words">
        <span className="portal-account-offer-title">{copy.title}</span>
        <span className="portal-account-offer-body">{copy.body}</span>
      </span>
      <Codicon aria-hidden className="portal-account-offer-chevron" name="chevron-right" />
    </button>
  )
}
