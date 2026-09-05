// Ambient activity in the bottom-right: what's happening right now, as text.
// Each line appears when something happens and fades a few seconds later. No
// chrome, no scrollback, nothing to dismiss — the full event log stays behind
// its control button for when you actually need to read it.
//
// This is a terminal tail, not a notification stack. Earlier it rendered a
// glyph plus the step's title, which meant three identical lines that told
// you nothing — the step was already visibly running on the canvas. What is
// actually informative is the WORK: which tool, on what file.
// So the line is `step  tool · arg`, no pill, right-aligned to the
// edge it hangs off.
import { useEffect, useRef, useState } from 'react'

import type { FeedLine } from './protocol-feed'

const LIFETIME_MS = 3400
// Six at a time — a glance of the last few beats, not a second event log.
const MAX_VISIBLE = 6

interface Emote {
  key: number
  line: FeedLine
  at: number
}

/** Step id → the label the canvas shows, so a line reads like the card. */
function labelFor(step: string, titles: Record<string, string>): string {
  return titles[step] ?? step
}

// The feed's `msg` is written for the log pane, where there's room for a full
// sentence. Out here the line is ~20 characters wide, so the prefixes that make
// sense in a scrollable list are pure noise — but the ARGUMENT stays: "patch"
// alone says nothing, "patch · H1 font-weight" says what's happening. CSS does
// the actual truncating, so a long tail costs nothing.
function terse(line: FeedLine): string {
  return line.msg
    .replace(/^delegate_task spawned · /, '')
    .replace(/^skipped · /, 'skipped ')
    .replace(/^take (\d+) → \w+ · /, 'take $1 ')
}

export function LiveLog({
  lines,
  titles,
  hidden
}: {
  lines: FeedLine[]
  titles: Record<string, string>
  /** Suppressed while the full event log is open — the emotes are the ambient
   *  version of the same stream, so showing both is redundant, and the log's
   *  translucent glass lets them bleed through rather than covering them. */
  hidden?: boolean
}) {
  const [emotes, setEmotes] = useState<Emote[]>([])
  const seen = useRef(0)

  // eslint-disable-next-line no-restricted-syntax -- `seen` is a high-water mark over the feed, not a mirrored atom.
  useEffect(() => {
    // Scrubbing backwards shortens the stream — drop the ambient trail with it
    // rather than showing activity that hasn't happened at this playhead.
    if (lines.length < seen.current) {
      seen.current = lines.length
      setEmotes([])

      return
    }

    if (lines.length === seen.current) {return}
    const now = Date.now()
    const fresh = lines.slice(seen.current).map((line, i) => ({ key: seen.current + i, line, at: now }))
    seen.current = lines.length
    setEmotes(prev => [...prev, ...fresh].slice(-MAX_VISIBLE))
  }, [lines])

  useEffect(() => {
    if (emotes.length === 0) {return}
    const id = setInterval(() => setEmotes(prev => prev.filter(e => Date.now() - e.at < LIFETIME_MS)), 250)

    return () => clearInterval(id)
  }, [emotes.length])

  if (hidden || emotes.length === 0) {return null}

  return (
    <div className="emotes">
      {emotes.map(e => (
        <div
          className={`emote k-${e.line.kind}`}
          key={e.key}
          // The step name is dropped at this width — "Implement UI" alone is
          // 12 of ~20 characters, and the canvas is already showing you which
          // step is running. The work is the only part worth the room. It
          // stays on the title for a hover.
          title={`${labelFor(e.line.step, titles)} · ${e.line.msg}`}
        >
          {terse(e.line)}
        </div>
      ))}
    </div>
  )
}
