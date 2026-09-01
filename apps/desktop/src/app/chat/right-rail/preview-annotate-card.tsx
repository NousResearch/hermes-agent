import { useEffect, useRef } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  ANNOTATE_CARD_HEIGHT,
  ANNOTATE_CARD_WIDTH,
  ANNOTATE_MARKER_SIZE,
  ANNOTATE_PILL_BG,
  ANNOTATE_PILL_FG,
  ANNOTATE_PILL_SEND
} from '@/lib/preview-annotate'

const PAD = 12
const GAP = 8

interface AnnotateCardPlacement {
  left: number
  top: number
}

interface PlaceAnnotateCardInput {
  paneHeight: number
  paneWidth: number
  rect: { height: number; width: number; x: number; y: number }
}

interface PreviewAnnotateCardProps extends AnnotateCardPlacement {
  note: string
  number: number
  onCancel: () => void
  onChange: (note: string) => void
  onSave: () => void
  placeholder: string
  saveLabel: string
  title: string
}

export function placeAnnotateCard({
  paneHeight,
  paneWidth,
  rect
}: PlaceAnnotateCardInput): AnnotateCardPlacement {
  const width = Math.min(ANNOTATE_CARD_WIDTH, Math.max(0, paneWidth - PAD * 2))
  const maxLeft = Math.max(PAD, paneWidth - width - PAD)
  const maxTop = Math.max(PAD, paneHeight - ANNOTATE_CARD_HEIGHT - PAD)
  const pinRight = rect.x + ANNOTATE_MARKER_SIZE / 2
  const preferRight = pinRight + GAP
  const preferLeft = rect.x - ANNOTATE_MARKER_SIZE / 2 - GAP - width
  const left =
    preferRight + width + PAD <= paneWidth || preferLeft < PAD
      ? Math.min(Math.max(PAD, preferRight), maxLeft)
      : Math.min(Math.max(PAD, preferLeft), maxLeft)
  const top = Math.min(Math.max(PAD, rect.y - ANNOTATE_CARD_HEIGHT / 2), maxTop)

  return { left, top }
}

export function PreviewAnnotateCard({
  left,
  note,
  number,
  onCancel,
  onChange,
  onSave,
  placeholder,
  saveLabel,
  title,
  top
}: PreviewAnnotateCardProps) {
  const field = useRef<HTMLInputElement>(null)

  useEffect(() => {
    field.current?.focus()
  }, [number])

  return (
    <form
      aria-label={title}
      className="absolute z-20 flex h-11 w-[min(17.5rem,calc(100%-1.5rem))] items-center gap-1 rounded-full pl-4 pr-1 shadow-nous"
      data-annotate-card="true"
      data-annotate-number={number}
      onSubmit={event => {
        event.preventDefault()
        onSave()
      }}
      style={{
        background: ANNOTATE_PILL_BG,
        boxShadow: '0 10px 28px rgba(0, 0, 0, 0.32), 0 0 0 1px rgba(255, 255, 255, 0.08)',
        color: ANNOTATE_PILL_FG,
        left,
        top
      }}
    >
      <input
        aria-label={placeholder}
        autoComplete="off"
        className="min-w-0 flex-1 bg-transparent text-[0.8125rem] leading-5 outline-none placeholder:text-white/45"
        onChange={event => onChange(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.preventDefault()
            onCancel()
          }
        }}
        placeholder={placeholder}
        ref={field}
        spellCheck
        style={{ caretColor: ANNOTATE_PILL_FG, color: ANNOTATE_PILL_FG, colorScheme: 'dark' }}
        value={note}
      />
      <button
        aria-label={saveLabel}
        className="grid size-8 shrink-0 place-items-center rounded-full text-white/85 hover:text-white"
        style={{ background: ANNOTATE_PILL_SEND }}
        type="submit"
      >
        <Codicon name="arrow-up" size="0.875rem" />
      </button>
    </form>
  )
}
