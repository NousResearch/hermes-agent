import { useStore } from '@nanostores/react'
import { useRef } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  $newcomerGreeting,
  $starterIdeas,
  hideStarterIdeas,
  noteStarterIdeaPicked,
  showStarterIdeas,
  STARTER_IDEAS
} from '@/store/starter-ideas'

import {
  type ComposerTarget,
  requestComposerFocus,
  requestComposerInsert,
  requestComposerSubmit
} from '../chat/composer/focus'

import { useFirstTimeCopy } from './first-time-copy'

/** "Hi, I'm Hermes." over a newcomer's empty chat (1A' 479:4114), until their first tasks are done. */
export function LandingGreeting() {
  const greet = useStore($newcomerGreeting)
  const copy = useFirstTimeCopy().landing

  return greet ? <h1 className="discovery-landing-greeting">{copy.greeting}</h1> : null
}

interface StarterIdeasProps {
  /** The composer an idea is sent through, as if typed there. */
  target: ComposerTarget
}

/**
 * A newcomer's four starter ideas under the composer, with Hide; once they are gone, Show ideas
 * (479:4114, 486:4227, 486:5035; rules in store/starter-ideas.ts). An idea is sent like a typed
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
                variant="outline"
              >
                {copy.ideas[idea]}
              </Button>
            </li>
          ))}
          <li>
            <Button
              aria-label={copy.hideLabel}
              className="discovery-ideas-toggle"
              onClick={() => {
                hideStarterIdeas()
                keepFocusAcrossSwap()
              }}
              size="inline"
              variant="text"
            >
              {copy.hide}
              <Codicon name="close" />
            </Button>
          </li>
        </ul>
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
