import { startSignUp } from '@/store/visitor-access'

import { useFirstTimeCopy } from './first-time-copy'

/** "Sign up to start" at the top end of the composer while sign up is required (Figma 18695:18460),
 *  where the model row sits once there is an account to pick a model for. */
export function ComposerSignUpLink() {
  const copy = useFirstTimeCopy().landing

  return (
    <button className="discovery-composer-sign-up" onClick={() => startSignUp('chat')} type="button">
      {copy.signUp}
    </button>
  )
}
