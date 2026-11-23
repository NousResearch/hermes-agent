import './intro-copy.css'

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'
import { chatMessageText } from '@/lib/chat-messages'
import { useStoreSelector } from '@/lib/use-session-slice'

import { $introCopy, $introTurnSent } from './intro'

// The boot overlay (the `starting` screen) clears its text before it fades; typing starts under the fade.
const START_MS = 660
const LINE_GAP_MS = 300
// Typing never runs past 2.5 s: a throttled frame snaps the rest in.
const MAX_TYPED_MS = 2500
const HOLD_MS = 250
const SPRING_MS = 530

type Lines = readonly [string, string]

function timing([first, second]: Lines) {
  const line1 = (first.length * 1000) / 26
  const line2 = (second.length * 1000) / 48

  return { line1, typed: Math.min(line1 + LINE_GAP_MS + line2, MAX_TYPED_MS), line2 }
}

function typedAt(lines: Lines, elapsed: number): Lines {
  const { line1, line2, typed } = timing(lines)

  if (elapsed >= typed) {
    return lines
  }

  const first = Math.floor((Math.min(elapsed, line1) / line1) * lines[0].length)
  const second = Math.floor((Math.max(0, elapsed - line1 - LINE_GAP_MS) / line2) * lines[1].length)

  return [lines[0].slice(0, first), lines[1].slice(0, Math.min(second, lines[1].length))]
}

/** Centred intro copy that springs up into the empty setup chat. Primary chat view only. */
export function IntroCopy() {
  // The backend's prelude writes the same words in the same locale (_GREETINGS in
  // agent/onboarding_text.py) as the onboarding chat's first assistant row, which replaces this copy.
  const { t } = useI18n()
  const lines: Lines = [t.firstRun.intro.greeting, t.firstRun.intro.plan]
  const typedMs = timing(lines).typed
  const stage = useStore($introCopy)
  const view = useSessionView()
  const reducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)')
  const [elapsed, setElapsed] = useState(0)
  const [risen, setRisen] = useState(false)

  // The first visible assistant words land here; a turn that ended without any (an error) shows itself.
  const landed = useStoreSelector(view.$messages, messages =>
    messages.some(message => message.role === 'assistant' && !message.hidden && chatMessageText(message).trim())
  )

  const busy = useStore(view.$busy)
  const awaiting = useStore(view.$awaitingResponse)
  const turnSent = useStore($introTurnSent)
  const settledWithoutWords = turnSent && !busy && !awaiting

  useEffect(() => {
    if (stage !== 'playing') {
      return
    }

    if (reducedMotion) {
      setElapsed(typedMs)
      setRisen(true)
      $introCopy.set('landed')

      return
    }

    const started = performance.now() + START_MS
    let frame = 0
    let timer = 0

    const tick = () => {
      const now = Math.max(0, performance.now() - started)
      setElapsed(now)

      if (now < typedMs) {
        frame = requestAnimationFrame(tick)

        return
      }

      timer = window.setTimeout(() => {
        setRisen(true)
        timer = window.setTimeout(() => $introCopy.set('landed'), SPRING_MS)
      }, HOLD_MS)
    }

    frame = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(frame)
      window.clearTimeout(timer)
    }
  }, [reducedMotion, stage, typedMs])

  useEffect(() => {
    if (stage === 'landed' && (landed || settledWithoutWords)) {
      $introCopy.set('hidden')
    }
  }, [landed, settledWithoutWords, stage])

  if (stage === 'hidden') {
    return null
  }

  const [first, second] = typedAt(lines, elapsed)

  return (
    <div aria-label={lines.join(' ')} data-risen={risen ? '' : undefined} data-slot="intro-copy" role="status">
      <div aria-hidden="true" className="intro-copy-text">
        <p>{first}</p>
        <p className="intro-copy-second">{second}</p>
      </div>
    </div>
  )
}
