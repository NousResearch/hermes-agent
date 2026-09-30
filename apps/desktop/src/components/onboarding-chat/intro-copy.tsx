import './guide-loading.css'
import './intro-copy.css'

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useMediaQuery } from '@/hooks/use-media-query'
import { chatMessageText } from '@/lib/chat-messages'
import { useStoreSelector } from '@/lib/use-session-slice'

import { $introCopy, $introTurnSent } from './intro'

// App-owned, English only. The backend's prelude writes the same words as the setup chat's first
// assistant row, which replaces this copy once it lands.
const LINES = ["Hi, I'm Hermes.", "Let's set things up for you. Then we'll get something cool done."] as const

// The boot overlay (the `starting` screen) clears its text before it fades; typing starts under the fade.
const START_MS = 660
const LINE_1_MS = (LINES[0].length * 1000) / 26
const LINE_GAP_MS = 300
const LINE_2_MS = (LINES[1].length * 1000) / 48
// Typing never runs past 2.5 s: a throttled frame snaps the rest in.
const TYPED_MS = Math.min(LINE_1_MS + LINE_GAP_MS + LINE_2_MS, 2500)
const HOLD_MS = 250
const SPRING_MS = 530

function typedAt(elapsed: number): readonly [string, string] {
  if (elapsed >= TYPED_MS) {
    return LINES
  }

  const first = Math.floor((Math.min(elapsed, LINE_1_MS) / LINE_1_MS) * LINES[0].length)
  const second = Math.floor((Math.max(0, elapsed - LINE_1_MS - LINE_GAP_MS) / LINE_2_MS) * LINES[1].length)

  return [LINES[0].slice(0, first), LINES[1].slice(0, Math.min(second, LINES[1].length))]
}

/** Centred intro copy that springs up into the empty setup chat. Primary chat view only. */
export function IntroCopy() {
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
      setElapsed(TYPED_MS)
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

      if (now < TYPED_MS) {
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
  }, [reducedMotion, stage])

  useEffect(() => {
    if (stage === 'landed' && (landed || settledWithoutWords)) {
      $introCopy.set('hidden')
    }
  }, [landed, settledWithoutWords, stage])

  if (stage === 'hidden') {
    return null
  }

  const [first, second] = typedAt(elapsed)

  return (
    <div aria-label={LINES.join(' ')} data-risen={risen ? '' : undefined} data-slot="intro-copy" role="status">
      <div aria-hidden="true" className="intro-copy-text">
        <p>{first}</p>
        <p className="intro-copy-second">{second}</p>
      </div>
    </div>
  )
}
