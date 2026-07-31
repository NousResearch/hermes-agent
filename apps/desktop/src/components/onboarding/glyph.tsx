import { useEffect, useState } from 'react'

import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

// Borrowed from the gateway "connecting" overlay: a mono, letter-spaced label
// that decodes left-to-right from scrambled glyphs into the real text, with a
// blinking block cursor. Ties onboarding's success moment to that same motif.
// Cuneiform glyphs (array, since each is a surrogate pair) for the scramble.
// Hero "X CONNECTED" decode uses the SAME ascii map as the connecting overlay.
const ASCII_GLYPHS = [...'/\\|-_=+<>~:*']
const pickAscii = () => ASCII_GLYPHS[(Math.random() * ASCII_GLYPHS.length) | 0]
// Cuneiform is reserved for the subtle "other text" (model name + BEGIN) easter egg.
const SCRAMBLE_GLYPHS = [...'𒀀𒀁𒀂𒀅𒀊𒀖𒀜𒀭𒀲𒀸𒁀𒁉𒁒𒁕𒁹𒂊𒃻𒄆𒄴𒅀𒆍𒇽𒈨𒉡']
const GLYPH_SET = new Set(SCRAMBLE_GLYPHS)
const pickGlyph = () => SCRAMBLE_GLYPHS[(Math.random() * SCRAMBLE_GLYPHS.length) | 0]
// How many trailing characters of each word scramble during decode-in.
const DECODE_TAIL = 4

// Renders text where cuneiform scramble-glyphs are dropped to a smaller em-size
// (resolved Latin chars stay full size) — keeps the easter-egg glyphs subtle.
export function GlyphText({ text }: { text: string }) {
  return (
    <>
      {Array.from(text, (ch, i) =>
        GLYPH_SET.has(ch) ? (
          <span className="text-[0.62em]" key={i}>
            {ch}
          </span>
        ) : (
          ch
        )
      )}
    </>
  )
}

function useDecoded(text: string): string {
  const [out, setOut] = useState(text)

  useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setOut(text)

      return
    }

    // Each WORD keeps its head static and only churns its tail (last few chars),
    // resolving left-to-right across all tails — same anchor-the-prefix trick the
    // connecting overlay uses ("CONN" static, "ECTING" churns), applied per word
    // so both the provider and "CONNECTED" decode and time stays constant.
    const chars = [...text]
    const scrambleable = chars.map(() => false)

    for (let i = 0; i < chars.length; ) {
      if (!/[a-z0-9]/i.test(chars[i])) {
        i += 1

        continue
      }

      let j = i

      while (j < chars.length && /[a-z0-9]/i.test(chars[j])) {
        j += 1
      }

      for (let k = Math.max(i, j - DECODE_TAIL); k < j; k += 1) {
        scrambleable[k] = true
      }

      i = j
    }

    const tailIndices = chars.map((_, idx) => idx).filter(idx => scrambleable[idx])
    let resolved = 0

    const id = window.setInterval(() => {
      resolved += 0.5
      const settled = new Set(tailIndices.slice(0, Math.floor(resolved)))

      setOut(chars.map((ch, idx) => (scrambleable[idx] && !settled.has(idx) ? pickAscii() : ch)).join(''))

      if (Math.floor(resolved) >= tailIndices.length) {
        window.clearInterval(id)
      }
    }, 45)

    return () => window.clearInterval(id)
  }, [text])

  return out
}

// Continuously scrambles alphanumeric chars while `active` (used on exit so the
// model name / button decay into ascii noise as they fade).
export function useScramble(text: string, active: boolean): string {
  const [out, setOut] = useState(text)

  useEffect(() => {
    if (!active) {
      setOut(text)

      return
    }

    const id = window.setInterval(() => {
      setOut(Array.from(text, ch => (/[a-z0-9]/i.test(ch) ? pickGlyph() : ch)).join(''))
    }, 45)

    return () => window.clearInterval(id)
  }, [text, active])

  return out
}

export function DecodedLabel({ leaving, text }: { leaving?: boolean; text: string }) {
  const decoded = useDecoded(text.toUpperCase())

  return (
    <span
      className={cn(
        'inline-flex items-center font-mono text-xs font-semibold uppercase tracking-[0.28em] tabular-nums text-primary transition duration-[360ms] ease-out',
        leaving ? 'translate-y-2 opacity-0 saturate-0' : 'translate-y-0 opacity-100 saturate-100'
      )}
    >
      <GlyphText text={decoded} />
      <span
        aria-hidden="true"
        className="dither ml-1.5 -mr-[0.875rem] inline-block size-2 shrink-0 -translate-y-px rounded-[1px] text-primary"
        style={{ animation: 'ob-decode-cursor 1s step-end infinite' }}
      />
      <style>{'@keyframes ob-decode-cursor { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }'}</style>
    </span>
  )
}

// Terminal-flavored CTA to match the connecting overlay's hacker aesthetic:
// mono, uppercase, letter-spaced, wrapped in primary brackets that light up on
// hover. The whole onboarding "you're in" moment leans into this motif.
export function HackeryButton({
  disabled,
  label,
  loading,
  onClick
}: {
  disabled?: boolean
  label: React.ReactNode
  loading?: boolean
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'group inline-flex items-center gap-2 rounded-md border border-(--stroke-nous) px-6 py-2.5',
        'font-mono text-xs font-semibold uppercase text-primary',
        'transition-all duration-150 hover:border-primary/60 hover:bg-primary/[0.06]',
        'disabled:pointer-events-none disabled:opacity-50'
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <span className="text-primary/40 transition-colors group-hover:text-primary">[</span>
      {loading ? <Loader2 className="size-3 animate-spin" /> : null}
      <span className="-mr-[0.25em] pl-[0.25em] tracking-[0.25em]">{label}</span>
      <span className="text-primary/40 transition-colors group-hover:text-primary">]</span>
    </button>
  )
}
