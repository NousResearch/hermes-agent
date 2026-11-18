import { useStore } from '@nanostores/react'
import { useCallback } from 'react'

import { Button } from '@/components/ui/button'
import {
  closeFirstTimeSplash,
  type FirstTimeSplashClose,
  type FirstTimeSplashRow,
  noteFirstTimeSplashRow
} from '@/store/first-time-splash'
import { $signUpRequired, startSignUp } from '@/store/visitor-access'

import { requestComposerFocus } from '../chat/composer/focus'

import { useFirstTimeCopy } from './first-time-copy'
import { MakeItYours } from './make-it-yours'
import { useRenamedNames } from './renamed-names'
import { type SplashRow, SplashSurface } from './splash-surface'

/** The destinations the splash explains, in its order (Figma 1A). No Machines or Notifications: a
 *  newcomer has nothing there yet. */
const PLACES = ['chats', 'routines', 'agents', 'library'] as const

/** Closed by its own controls, focus goes where the person was headed: back to the bell that
 *  reopened it, else into the composer to get to work. A nav click already put focus where it went;
 *  sign up leaves the page. */
function leave(how: Exclude<FirstTimeSplashClose, 'nav'>, reopened: boolean) {
  closeFirstTimeSplash(how)

  if (how === 'signup') {
    startSignUp('splash')

    return
  }

  // Next frame: the chat under the splash is inert until the splash has unmounted.
  requestAnimationFrame(() => {
    if (reopened) {
      document.querySelector<HTMLElement>('[data-nav-spot="notifications"]')?.focus({ preventScroll: true })
    } else {
      requestComposerFocus('active')
    }
  })
}

interface FirstTimeSplashProps {
  /** Opened from the tray: it takes focus at once. */
  reopened: boolean
}

/** "Welcome to Hermes Agent" (Figma 1A 477:4004): what each place in the nav is for, one line each,
 *  with an example and the place lit while a row is hovered or focused; light or dark and a colour;
 *  and the way in (sign up where it is required). */
export function FirstTimeSplash({ reopened }: FirstTimeSplashProps) {
  const copy = useFirstTimeCopy().splash
  const names = useRenamedNames()
  const signUpRequired = useStore($signUpRequired)
  const onEscape = useCallback(() => leave('escape', reopened), [reopened])

  const rows: SplashRow<FirstTimeSplashRow>[] = [
    ...PLACES.map(place => ({
      id: place,
      name: `${names[place]}: ${copy.lines[place].line}`,
      label: (
        <>
          <span className="discovery-first-name">{names[place]}</span>
          <span className="discovery-first-line">{copy.lines[place].line}</span>
        </>
      ),
      detail: copy.lines[place].example,
      spot: place
    })),
    {
      id: 'theme',
      name: copy.theme.label,
      label: <span className="discovery-first-name">{copy.theme.label}</span>,
      control: <MakeItYours />
    }
  ]

  return (
    <SplashSurface
      close={{ label: copy.close, onClose: () => leave('close', reopened) }}
      eyebrow={copy.eyebrow}
      footer={
        <>
          <h2 className="discovery-splash-ready">{copy.ready}</h2>
          <p className="discovery-splash-footnote">{copy.footnote}</p>
          {signUpRequired ? (
            <Button onClick={() => leave('signup', reopened)} size="lg">
              {copy.signUp}
            </Button>
          ) : (
            <Button onClick={() => leave('close', reopened)} size="lg">
              {copy.start}
            </Button>
          )}
        </>
      }
      kind="first-time"
      listLabel={copy.listLabel}
      onEscape={onEscape}
      onRowActivated={noteFirstTimeSplashRow}
      rows={rows}
      startActive={false}
      subtitle={copy.subtitle}
      takeFocus={reopened}
      title={copy.title}
    />
  )
}
