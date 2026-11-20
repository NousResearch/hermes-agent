import './discovery.css'

import { useStore } from '@nanostores/react'
import { useEffect, useId } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { $artifactRegistry } from '@/store/artifacts'
import {
  $developerViewOfferOpen,
  $developerViewPreviewing,
  acceptDeveloperView,
  declineDeveloperView,
  noteDeveloperViewOffered,
  previewDeveloperView
} from '@/store/developer-view-offer'

import { useFirstTimeCopy } from './first-time-copy'

interface DeveloperViewOfferProps {
  /** The chat shown: the offer is made in one whose answers held code. */
  sessionId: null | string
}

/**
 * "Want the developer view?" (Onboarding C6 18677:28595), floated above the composer of a chat about
 * code, for someone in Simple (store/developer-view-offer.ts). "About code" is the app's own signal:
 * an answer in this chat held a code block long enough to become a code artifact
 * (lib/artifact-detect.ts). Hovering or focusing "See the work as it happens" shows the developer view
 * while it lasts (a tap previews on touch); Turn on keeps it, Not now retires the offer.
 */
export function DeveloperViewOffer({ sessionId }: DeveloperViewOfferProps) {
  const open = useStore($developerViewOfferOpen)
  const previewing = useStore($developerViewPreviewing)
  const registry = useStore($artifactRegistry)
  const copy = useFirstTimeCopy().developerView
  const id = useId()
  const aboutCode = Boolean(sessionId && registry[sessionId]?.some(record => record.kind === 'code'))
  const shown = open && aboutCode

  useEffect(() => {
    if (shown) {
      noteDeveloperViewOffered()
    }
  }, [shown])

  // No unmount cleanup: switching the interface mode rebuilds the layout and remounts this card under
  // the pointer, which must not end the preview it just started. Leaving the row ends it.

  if (!shown) {
    return null
  }

  const preview = (on: boolean) => () => previewDeveloperView(on)

  return (
    <section aria-labelledby={`${id}-title`} className="discovery-developer-offer" data-slot="developer-view-offer">
      <p className="discovery-welcome-eyebrow">{copy.eyebrow}</p>
      <div className="discovery-landing-heading">
        <h2 className="discovery-welcome-title" id={`${id}-title`}>
          {copy.title}
        </h2>
        <p className="discovery-landing-subtitle">{copy.body}</p>
      </div>
      <button
        aria-pressed={previewing}
        className="discovery-developer-preview"
        onBlur={preview(false)}
        // A tap previews too (touch has no hover); leaving the row, by pointer or focus, ends it.
        onClick={preview(true)}
        onFocus={preview(true)}
        onPointerEnter={preview(true)}
        onPointerLeave={preview(false)}
        type="button"
      >
        <Codicon aria-hidden className="discovery-welcome-glyph" name="terminal" />
        <span className="discovery-welcome-place-text">
          <span className="discovery-welcome-place-name">{copy.preview}</span>
          <span className="discovery-welcome-place-line">{copy.previewLine}</span>
        </span>
        <Codicon aria-hidden className="discovery-check-in-chevron" name="chevron-right" />
      </button>
      <div className="discovery-check-in-answers">
        <Button className="discovery-landing-action" onClick={acceptDeveloperView}>
          {copy.turnOn}
        </Button>
        <Button className="discovery-landing-action" onClick={declineDeveloperView} variant="secondary">
          {copy.notNow}
        </Button>
      </div>
      <p className="discovery-developer-note">{copy.settingsNote}</p>
    </section>
  )
}
