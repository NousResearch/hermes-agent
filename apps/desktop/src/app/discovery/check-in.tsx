import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FileText } from '@/lib/icons'
import { answerCheckIn, type CheckInTry, dismissCheckIn, noteCheckInTry } from '@/store/check-in'
import { reopenFirstTimeSplash } from '@/store/first-time-splash'

import {
  type ComposerTarget,
  requestComposerFocus,
  requestComposerInsert,
  requestComposerSubmit
} from '../chat/composer/focus'

import { useFirstTimeCopy } from './first-time-copy'

interface CheckInProps {
  /** The composer a suggestion is sent through, as if typed there. */
  target: ComposerTarget
}

/**
 * "How's Hermes going so far?" (Onboarding C7 18677:28705) in place of the landing's heading, once two
 * chats have had a task finish (store/check-in.ts). Its first two suggestions are tasks, sent through
 * the composer like a starter idea (the routine sheet takes no prefilled prompt, so "Keep me updated
 * weekly" asks the agent, which can set the routine up); the third reopens the tour. Going well and
 * Could be better answer it; Dismiss puts it in the notifications tray. Both retire it here.
 */
export function CheckIn({ target }: CheckInProps) {
  const copy = useFirstTimeCopy().checkIn

  const send = (action: CheckInTry, text: string) => {
    noteCheckInTry(action)

    // No composer to send through right now: the suggestion waits in it instead.
    if (!requestComposerSubmit(text, { target })) {
      requestComposerInsert(text, { target })
      requestComposerFocus(target)
    }
  }

  const rows = [
    {
      id: 'try_video',
      glyph: <FileText aria-hidden className="discovery-welcome-glyph" />,
      title: copy.video,
      line: copy.videoLine,
      action: copy.try,
      run: () => send('try_video', copy.video)
    },
    {
      id: 'try_weekly',
      glyph: <Codicon aria-hidden className="discovery-welcome-glyph" name="zap" />,
      title: copy.weekly,
      line: copy.weeklyLine,
      action: copy.try,
      run: () => send('try_weekly', copy.weekly)
    },
    {
      id: 'open_tour',
      glyph: <Codicon aria-hidden className="discovery-welcome-glyph" name="redo" />,
      title: copy.tour,
      line: copy.tourLine,
      action: copy.open,
      run: () => {
        noteCheckInTry('open_tour')
        reopenFirstTimeSplash()
      }
    }
  ]

  return (
    <section aria-label={copy.title} className="discovery-check-in" data-slot="check-in">
      <p className="discovery-welcome-eyebrow">{copy.eyebrow}</p>
      <div className="discovery-landing-heading">
        <h1 className="discovery-welcome-title">{copy.title}</h1>
        <p className="discovery-landing-subtitle">{copy.subtitle}</p>
      </div>
      <ul aria-label={copy.listLabel} className="discovery-welcome-card discovery-check-in-rows">
        {rows.map(row => (
          <li key={row.id}>
            <button className="discovery-check-in-row" onClick={row.run} type="button">
              {row.glyph}
              <span className="discovery-welcome-place-text">
                <span className="discovery-welcome-place-name">{row.title}</span>
                <span className="discovery-welcome-place-line">{row.line}</span>
              </span>
              <span aria-hidden className="discovery-check-in-action">
                {row.action}
                <Codicon className="discovery-check-in-chevron" name="chevron-right" />
              </span>
            </button>
          </li>
        ))}
      </ul>
      <div className="discovery-check-in-answers">
        <Button className="discovery-landing-action" onClick={() => answerCheckIn('going_well')}>
          {copy.goingWell}
        </Button>
        <Button
          className="discovery-landing-action"
          onClick={() => answerCheckIn('could_be_better')}
          variant="secondary"
        >
          {copy.couldBeBetter}
        </Button>
      </div>
      <Button className="discovery-ideas-toggle" onClick={dismissCheckIn} size="inline" variant="text">
        {copy.dismiss}
        <Codicon aria-hidden name="close" />
      </Button>
    </section>
  )
}
