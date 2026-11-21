import { useStore } from '@nanostores/react'
import { useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FileText } from '@/lib/icons'
import { $checkInDue } from '@/store/check-in'
import { $sessions } from '@/store/session'
import {
  $landingMoment,
  $starterIdeas,
  hideStarterIdeas,
  noteStarterIdeaPicked,
  showStarterIdeas,
  STARTER_IDEAS,
  type StarterIdea
} from '@/store/starter-ideas'
import { $signUpRequired } from '@/store/visitor-access'

import {
  type ComposerTarget,
  requestComposerFocus,
  requestComposerInsert,
  requestComposerSubmit
} from '../chat/composer/focus'
import { sessionRoute } from '../routes'

import { CheckIn } from './check-in'
import { useFirstTimeCopy } from './first-time-copy'

interface LandingProps {
  /** The composer the landing's ideas and suggestions are sent through, as if typed there. */
  target: ComposerTarget
}

/**
 * What a newcomer's empty chat says above the composer ($landingMoment): "Welcome to Hermes" before a
 * first task (18695:18388, the eyebrow naming the hosted Portal where sign up is required), "Welcome
 * back." after one (C4; its line asks for a task once the ideas are put away, C5), "Good to see you
 * again." after a month away (C8). The check-in takes its place while it is due (C7). The app knows
 * no one's name, so it is never "Welcome back, [name]" (18732:32622).
 */
export function LandingGreeting({ target }: LandingProps) {
  const moment = useStore($landingMoment)
  const ideas = useStore($starterIdeas)
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
    first: { title: copy.landing.title, line: copy.landing.subtitle },
    back: { title: copy.landing.back, line: ideas === 'ideas' ? copy.landing.subtitle : copy.landing.subtitleRetired },
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

/** The newest chat the person was in, for "Continue a recent chat" (C8). */
function useNewestChatId(): null | string {
  const sessions = useStore($sessions)

  return (
    sessions
      .filter(row => !row.hidden && !row.archived && !row.scheduler_owned && !row.cron_execution_id)
      .reduce<null | (typeof sessions)[number]>(
        (newest, row) => (!newest || row.last_active > newest.last_active ? row : newest),
        null
      )?.id ?? null
  )
}

/** Each idea's glyph (18695:18476–18486): what the task is about. */
function IdeaGlyph({ idea }: { idea: StarterIdea }) {
  if (idea === 'summarize_document') {
    return <FileText aria-hidden className="discovery-idea-glyph" />
  }

  const name = { morning_briefing: 'clock', research_options: 'search', clean_inbox: 'inbox' }[idea]

  return <Codicon aria-hidden className="discovery-idea-glyph" name={name} />
}

interface StarterIdeasProps {
  /** The composer an idea is sent through, as if typed there. */
  target: ComposerTarget
}

/**
 * A newcomer's four starter ideas under the composer, with Hide ideas under them; once they are gone,
 * Show ideas (18677:28385, 18677:28493; rules in store/starter-ideas.ts). An idea is sent like a typed
 * message, through the composer, so it meets everything a typed one does (a visitor who must sign up
 * gets it held). Hiding or bringing the ideas back keeps focus on the control that replaces the one
 * pressed.
 */
export function StarterIdeas({ target }: StarterIdeasProps) {
  const state = useStore($starterIdeas)
  const moment = useStore($landingMoment)
  const copy = useFirstTimeCopy().landing
  const regionRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const newestChat = useNewestChatId()
  // Back after a month away the ideas wait behind Show ideas, beside the way back to the work (C8).
  const [awayIdeas, setAwayIdeas] = useState(false)

  // The control pressed is replaced (Hide by Show ideas, Show ideas by the ideas): once it has painted,
  // focus goes to the first control in its place instead of falling to the page.
  const keepFocusAcrossSwap = () =>
    requestAnimationFrame(() => regionRef.current?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true }))

  if (state === 'none') {
    return null
  }

  if (moment === 'away' && !awayIdeas) {
    return (
      <div className="discovery-ideas" data-slot="starter-ideas" ref={regionRef}>
        <div className="discovery-landing-actions">
          {newestChat && (
            <Button className="discovery-landing-action" onClick={() => navigate(sessionRoute(newestChat))}>
              {copy.continueChat}
            </Button>
          )}
          <Button
            className="discovery-landing-action"
            onClick={() => {
              showStarterIdeas()
              setAwayIdeas(true)
              keepFocusAcrossSwap()
            }}
            variant="secondary"
          >
            {copy.showIdeas}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="discovery-ideas" data-slot="starter-ideas" ref={regionRef}>
      {state === 'ideas' ? (
        <>
          <ul aria-label={copy.ideasLabel} className="discovery-ideas-list">
            {STARTER_IDEAS.map(idea => (
              <li key={idea}>
                <Button
                  className="discovery-idea"
                  onClick={() => {
                    noteStarterIdeaPicked(idea)

                    // No composer to send through right now: the idea waits in it instead.
                    if (!requestComposerSubmit(copy.ideas[idea], { target })) {
                      requestComposerInsert(copy.ideas[idea], { target })
                      requestComposerFocus(target)
                    }
                  }}
                  variant="secondary"
                >
                  <IdeaGlyph idea={idea} />
                  {copy.ideas[idea]}
                </Button>
              </li>
            ))}
          </ul>
          <Button
            className="discovery-ideas-toggle"
            onClick={() => {
              hideStarterIdeas()
              keepFocusAcrossSwap()
            }}
            size="inline"
            variant="text"
          >
            {copy.hide}
          </Button>
        </>
      ) : (
        <Button
          className="discovery-ideas-toggle"
          onClick={() => {
            showStarterIdeas()
            keepFocusAcrossSwap()
          }}
          size="inline"
          variant="text"
        >
          {copy.showIdeas}
        </Button>
      )}
    </div>
  )
}
