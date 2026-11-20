import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useId } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { $heldPrompt } from '@/store/held-prompt'
import { dismissSignInRecovery, startSignUp } from '@/store/visitor-access'

import { requestComposerFocus } from '../chat/composer/focus'

import { useFirstTimeCopy } from './first-time-copy'
import { useSplashFrame } from './splash-surface'

type StatusTone = 'alert' | 'ok' | 'waiting'

interface StatusRowProps {
  glyph: string
  title: string
  line: string
  badge: string
  tone: StatusTone
}

/** One step of signing in: what it is, a line, and where it stands as a badge (18732:33149). A step
 *  still waiting steps back like Figma's (18732:33167). */
function StatusRow({ badge, glyph, line, title, tone }: StatusRowProps) {
  return (
    <li className="discovery-status-row" data-tone={tone}>
      <Codicon aria-hidden className="discovery-welcome-glyph" name={glyph} />
      <span className="discovery-welcome-place-text">
        <span className="discovery-welcome-place-name">{title}</span>
        <span className="discovery-welcome-place-line">{line}</span>
      </span>
      <span className="discovery-status-badge" data-tone={tone}>
        {badge}
      </span>
    </li>
  )
}

/** Every way on starts sign up again: the portal's hand-off takes no hint for "another email" or "a new
 *  link", so the three share it. */
const restart = () => startSignUp('recovery')

/**
 * "Let's try signing in again" (Figma 18732:33066, with a held message 18695:19224): sign up came back
 * without signing the visitor in (`?hermes_sign_in=failed`, read once in store/visitor-access.ts).
 * It covers the chat like the welcome splash, in its column and card, and is the one place that says
 * so; a nav click or Escape puts it away and leaves the held message held.
 */
export function SignInRecovery() {
  const copy = useFirstTimeCopy().recovery
  const held = useStore($heldPrompt)
  const id = useId()

  const onEscape = useCallback(() => {
    dismissSignInRecovery()
    requestAnimationFrame(() => requestComposerFocus('active'))
  }, [])

  const surfaceRef = useSplashFrame(false, onEscape)

  const rows: ReactNode[] = [
    <StatusRow
      badge={copy.unverified}
      glyph="pass"
      key="account"
      line={copy.accountLine}
      title={copy.account}
      tone="alert"
    />,
    held && <StatusRow badge={copy.saved} glyph="comment" key="request" line={held} title={copy.request} tone="ok" />,
    <StatusRow
      badge={copy.waiting}
      glyph="loading"
      key="opening"
      line={copy.openingLine}
      title={copy.opening}
      tone="waiting"
    />
  ]

  return (
    <section
      aria-describedby={`${id}-subtitle`}
      aria-labelledby={`${id}-title`}
      className="discovery-splash"
      data-slot="discovery-splash"
      data-splash="recovery"
      ref={surfaceRef}
      tabIndex={-1}
    >
      <div className="discovery-welcome">
        <p className="discovery-welcome-eyebrow">{copy.eyebrow}</p>
        <div className="discovery-welcome-heading">
          <h1 className="discovery-welcome-title" id={`${id}-title`}>
            {copy.title}
          </h1>
          <p className="discovery-welcome-subtitle" id={`${id}-subtitle`}>
            {held ? copy.subtitleHeld : copy.subtitle}
          </p>
        </div>
        <ul aria-label={copy.listLabel} className="discovery-welcome-card discovery-status-rows">
          {rows}
        </ul>
        <div className="discovery-welcome-actions">
          <Button className="discovery-welcome-action" onClick={restart}>
            {copy.tryAgain}
          </Button>
          <Button className="discovery-welcome-action" onClick={restart} variant="secondary">
            {copy.otherEmail}
          </Button>
        </div>
        <p className="discovery-welcome-lead discovery-status-expired">
          {copy.expired}
          <br />
          <button className="discovery-status-link" onClick={restart} type="button">
            {copy.requestNew}
          </button>
        </p>
      </div>
    </section>
  )
}
