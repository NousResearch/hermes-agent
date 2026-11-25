import { useStore } from '@nanostores/react'

import { $checkInDue } from '@/store/check-in'
import { $landingMoment } from '@/store/starter-ideas'
import { $signUpRequired } from '@/store/visitor-access'

import type { ComposerTarget } from '../chat/composer/focus'

import { CheckIn } from './check-in'
import { useFirstTimeCopy } from './first-time-copy'

interface LandingProps {
  /** The composer the greeting's check-in is sent through, as if typed there. */
  target: ComposerTarget
}

/**
 * What a newcomer's empty chat says above the composer ($landingMoment): "Welcome to Hermes" before a
 * first task (18695:18388, the eyebrow naming the hosted Portal where sign up is required), "Welcome
 * back." after one (C4, C5), "Good to see you again." after a month away (C8). The check-in takes its
 * place while it is due (C7). The app knows no one's name, so it is never "Welcome back, [name]"
 * (18732:32622).
 */
export function LandingGreeting({ target }: LandingProps) {
  const moment = useStore($landingMoment)
  const checkIn = useStore($checkInDue)
  const signUpRequired = useStore($signUpRequired)
  const copy = useFirstTimeCopy()

  if (!moment) {
    return null
  }

  if (checkIn && moment !== 'first') {
    return (
      <div className="discovery-landing-greeting">
        <CheckIn target={target} />
      </div>
    )
  }

  const heading = {
    first: { title: copy.landing.title, line: copy.landing.subtitleRetired },
    back: { title: copy.landing.back, line: copy.landing.subtitleRetired },
    away: { title: copy.landing.awayTitle, line: copy.landing.awayLine }
  }[moment]

  return (
    <div className="discovery-landing-greeting" data-moment={moment}>
      {moment === 'first' && (
        <p className="discovery-welcome-eyebrow">{signUpRequired ? copy.landing.eyebrowPortal : copy.splash.eyebrow}</p>
      )}
      <div className="discovery-landing-heading">
        <h1 className="discovery-welcome-title">{heading.title}</h1>
        <p className="discovery-landing-subtitle">{heading.line}</p>
      </div>
    </div>
  )
}
