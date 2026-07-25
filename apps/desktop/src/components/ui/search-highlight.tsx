import type * as React from 'react'

import { parseHighlightSegments, wrapRanges } from '@/lib/search-match'
import { cn } from '@/lib/utils'

/** Inline mark styling shared by search surfaces. */
const MARK_CLASS = 'rounded-[2px] bg-(--ui-accent)/25 px-px font-medium text-foreground'

export function HighlightText({
  text,
  ranges,
  className
}: {
  text: string
  ranges?: Array<[number, number]>
  className?: string
}): React.ReactNode {
  if (!text) {
    return null
  }

  if (!ranges?.length) {
    return <span className={className}>{text}</span>
  }

  return <HighlightMarked className={className} text={wrapRanges(text, ranges)} />
}

/** Render text that already contains FTS `>>>/<<<` or client `[[m]]` markers. */
export function HighlightMarked({ text, className }: { text: string; className?: string }): React.ReactNode {
  const segments = parseHighlightSegments(text)

  if (segments.length === 1 && !segments[0].hit) {
    return <span className={className}>{segments[0].text}</span>
  }

  return (
    <span className={className}>
      {segments.map((seg, i) =>
        seg.hit ? (
          <mark className={MARK_CLASS} key={i}>
            {seg.text}
          </mark>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
    </span>
  )
}

/** Tiny in-row field chip (fits locked sidebar row height). */
export function MatchFieldChip({ label, className }: { label: string; className?: string }): React.ReactNode {
  if (!label) {
    return null
  }

  return (
    <span
      className={cn(
        'shrink-0 rounded px-1 py-px text-[0.55rem] leading-none text-(--ui-text-tertiary) bg-(--ui-bg-quinary)',
        className
      )}
    >
      {label}
    </span>
  )
}
