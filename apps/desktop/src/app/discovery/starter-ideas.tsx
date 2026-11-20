import { useStore } from '@nanostores/react'
import { useRef } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FileText } from '@/lib/icons'
import {
  $newcomerGreeting,
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

import { useFirstTimeCopy } from './first-time-copy'

/** "Welcome to Hermes" over a newcomer's empty chat (Figma 18695:18388), until their first tasks are
 *  done: the eyebrow names the hosted Portal where sign up is required, Hermes Agent elsewhere. */
export function LandingGreeting() {
  const greet = useStore($newcomerGreeting)
  const signUpRequired = useStore($signUpRequired)
  const copy = useFirstTimeCopy()

  if (!greet) {
    return null
  }

  return (
    <div className="discovery-landing-greeting">
      <p className="discovery-welcome-eyebrow">{signUpRequired ? copy.landing.eyebrowPortal : copy.splash.eyebrow}</p>
      <div className="discovery-landing-heading">
        <h1 className="discovery-welcome-title">{copy.landing.title}</h1>
        <p className="discovery-landing-subtitle">{copy.landing.subtitle}</p>
      </div>
    </div>
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
  const copy = useFirstTimeCopy().landing
  const regionRef = useRef<HTMLDivElement>(null)

  // The control pressed is replaced (Hide by Show ideas, Show ideas by the ideas): once it has painted,
  // focus goes to the first control in its place instead of falling to the page.
  const keepFocusAcrossSwap = () =>
    requestAnimationFrame(() => regionRef.current?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true }))

  if (state === 'none') {
    return null
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
