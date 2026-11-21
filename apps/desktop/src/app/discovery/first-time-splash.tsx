import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FileText } from '@/lib/icons'
import { useRootAttribute } from '@/lib/root-attribute'
import {
  $firstTimeSplashPlace,
  $firstTimeSplashSeen,
  closeFirstTimeSplash,
  FIRST_TIME_PLACES,
  type FirstTimeSplashClose,
  type FirstTimeSplashPlace,
  noteFirstTimeSplashRow
} from '@/store/first-time-splash'
import { $signUpRequired, startSignUp } from '@/store/visitor-access'

import { requestComposerFocus } from '../chat/composer/focus'

import { useFirstTimeCopy } from './first-time-copy'
import { MakeItYours } from './make-it-yours'
import { useRenamedNames } from './renamed-names'
import { useSplashFrame } from './splash-surface'

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

/** Each place's glyph: the rail's own (Pixelarticons in the redesign), Library's the file Figma draws. */
function PlaceGlyph({ place }: { place: FirstTimeSplashPlace }) {
  if (place === 'library') {
    return <FileText aria-hidden className="discovery-welcome-glyph" />
  }

  const name = { chats: 'comment-discussion', agents: 'robot', routines: 'clock' }[place]

  return <Codicon aria-hidden className="discovery-welcome-glyph" name={name} />
}

interface StepProps {
  titleId: string
  subtitleId: string
}

interface PlacesProps extends StepProps {
  onNext: () => void
  nextRef: RefObject<HTMLButtonElement | null>
}

/**
 * Step 1 (Figma 18695:17147): what each place in the nav is for, one row each in one card. The row a
 * pointer, focus or tap last reached is the place: it lights in the rail and Chats brings the chat
 * sidebar back, and while the pointer or focus is still on the rows it says its example in place of
 * its line (18695:17294). Every row reached carries the seen check for the rest of the showing
 * (18695:17833). Each row's name stays its place and line for assistive tech; the example and
 * the check are its description.
 */
function Places({ nextRef, onNext, subtitleId, titleId }: PlacesProps) {
  const copy = useFirstTimeCopy().splash
  const names = useRenamedNames()
  const place = useStore($firstTimeSplashPlace)
  const seen = useStore($firstTimeSplashSeen)
  const id = useId()
  // The rows under the pointer or holding focus: the rail lights the place only then (discovery.css).
  const [pointerOver, setPointerOver] = useState(false)
  const [focusWithin, setFocusWithin] = useState(false)

  useRootAttribute('data-discovery-place', pointerOver || focusWithin ? place : null)

  return (
    <>
      <p className="discovery-welcome-eyebrow">{copy.eyebrow}</p>
      <div className="discovery-welcome-heading">
        <h1 className="discovery-welcome-title" id={titleId}>
          {copy.title}
        </h1>
        <p className="discovery-welcome-subtitle" id={subtitleId}>
          {copy.subtitle}
        </p>
      </div>
      <ul
        aria-label={copy.listLabel}
        className="discovery-welcome-card discovery-welcome-places"
        onBlur={event => setFocusWithin(event.currentTarget.contains(event.relatedTarget))}
        onFocus={() => setFocusWithin(true)}
        onPointerEnter={() => setPointerOver(true)}
        onPointerLeave={() => setPointerOver(false)}
      >
        {FIRST_TIME_PLACES.map(row => {
          const active = row === place
          const isSeen = seen.has(row)
          const note = () => noteFirstTimeSplashRow(row)

          return (
            <li data-active={active || undefined} key={row}>
              <button
                aria-describedby={`${id}-${row}-example${isSeen ? ` ${id}-seen` : ''}`}
                aria-label={`${names[row]}: ${copy.lines[row].line}`}
                className="discovery-welcome-place"
                onClick={note}
                onFocus={note}
                onPointerEnter={event => {
                  if (event.pointerType === 'mouse') {
                    note()
                  }
                }}
                type="button"
              >
                <PlaceGlyph place={row} />
                <span aria-hidden className="discovery-welcome-place-text">
                  <span className="discovery-welcome-place-name">{names[row]}</span>
                  {/* Both in one cell: the row is as tall as the longer, so none moves when one speaks. */}
                  <span className="discovery-welcome-place-lines">
                    <span className="discovery-welcome-place-line">{copy.lines[row].line}</span>
                    <span className="discovery-welcome-place-line" data-example="">
                      {copy.lines[row].example}
                    </span>
                  </span>
                </span>
                {isSeen && (
                  <span aria-hidden className="discovery-welcome-seen" data-slot="seen-check">
                    <Codicon name="pass" />
                  </span>
                )}
                <span className="sr-only" id={`${id}-${row}-example`}>
                  {copy.lines[row].example}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      <span className="sr-only" id={`${id}-seen`}>
        {copy.seen}
      </span>
      <div className="discovery-welcome-lead">
        <h2 className="discovery-welcome-lead-title">{copy.style.title}</h2>
        <p className="discovery-welcome-lead-line">{copy.style.line}</p>
      </div>
      <div className="discovery-welcome-actions">
        <Button className="discovery-welcome-action" onClick={onNext} ref={nextRef}>
          {copy.style.action}
        </Button>
      </div>
    </>
  )
}

interface StyleProps extends StepProps {
  onBack: () => void
  reopened: boolean
  titleRef: RefObject<HTMLHeadingElement | null>
}

/** Step 2 (Figma 18695:18059): colour, light or dark, and the artwork, then the way in. Back returns
 *  to the places for keyboard and touch (Figma draws none: there a nav click closes). */
function Style({ onBack, reopened, subtitleId, titleId, titleRef }: StyleProps) {
  const copy = useFirstTimeCopy().splash
  const signUpRequired = useStore($signUpRequired)

  return (
    <>
      <div className="discovery-welcome-eyebrow-row">
        <p className="discovery-welcome-eyebrow">{copy.theme.eyebrow}</p>
        <Button className="discovery-welcome-back" onClick={onBack} size="inline" variant="text">
          <Codicon className="discovery-welcome-back-glyph" name="arrow-left" />
          {copy.theme.back}
        </Button>
      </div>
      <div className="discovery-welcome-heading">
        <h1 className="discovery-welcome-title" id={titleId} ref={titleRef} tabIndex={-1}>
          {copy.theme.title}
        </h1>
        <p className="discovery-welcome-subtitle" data-tone="compact" id={subtitleId}>
          {copy.theme.subtitle}
        </p>
      </div>
      <MakeItYours />
      <div className="discovery-welcome-lead" data-gap="loose">
        <h2 className="discovery-welcome-lead-title">{copy.ready}</h2>
        <p className="discovery-welcome-lead-line">{copy.footnote}</p>
      </div>
      <div className="discovery-welcome-actions">
        {signUpRequired ? (
          <>
            <Button className="discovery-welcome-action" onClick={() => leave('signup', reopened)}>
              {copy.signUp}
            </Button>
            <Button className="discovery-welcome-action" onClick={() => leave('close', reopened)} variant="secondary">
              {copy.lookAround}
            </Button>
          </>
        ) : (
          <Button className="discovery-welcome-action" onClick={() => leave('close', reopened)}>
            {copy.start}
          </Button>
        )}
      </div>
    </>
  )
}

interface FirstTimeSplashProps {
  /** Opened from the tray: it takes focus at once. */
  reopened: boolean
}

/** "Welcome to Hermes" (Figma Onboarding 18695:17147 → 18695:18059): the places in the nav, then
 *  "Make it yours" and the way in (sign up where it is required). Its own controls, Escape and any
 *  click in the nav close it; it has no ×. */
export function FirstTimeSplash({ reopened }: FirstTimeSplashProps) {
  const [step, setStep] = useState<'places' | 'style'>('places')
  const place = useStore($firstTimeSplashPlace)
  const onEscape = useCallback(() => leave('escape', reopened), [reopened])
  const surfaceRef = useSplashFrame(reopened, onEscape)
  const nextRef = useRef<HTMLButtonElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const stepChanged = useRef(false)
  const id = useId()

  // A step change moves focus with it: onto the second step's title, or back onto the button that
  // led there. The first render keeps the splash's own focus rule (useSplashFrame).
  useEffect(() => {
    if (!stepChanged.current) {
      return
    }

    const target = step === 'style' ? titleRef.current : nextRef.current
    target?.focus({ preventScroll: true })
  }, [step])

  const goTo = (next: 'places' | 'style') => {
    stepChanged.current = true

    if (next === 'style') {
      noteFirstTimeSplashRow('theme')
    }

    setStep(next)
  }

  const ids = { subtitleId: `${id}-subtitle`, titleId: `${id}-title` }

  return (
    <section
      aria-describedby={ids.subtitleId}
      aria-labelledby={ids.titleId}
      className="discovery-splash"
      data-place={step === 'places' ? (place ?? undefined) : undefined}
      data-slot="discovery-splash"
      data-splash="first-time"
      data-step={step}
      ref={surfaceRef}
      tabIndex={-1}
    >
      <div className="discovery-welcome">
        {step === 'places' ? (
          <Places nextRef={nextRef} onNext={() => goTo('style')} {...ids} />
        ) : (
          <Style onBack={() => goTo('places')} reopened={reopened} titleRef={titleRef} {...ids} />
        )}
      </div>
    </section>
  )
}
