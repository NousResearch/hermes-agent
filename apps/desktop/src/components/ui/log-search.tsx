import { splitLogSearchMatches } from '@/lib/log-search'

export function HighlightedLogText({ query, text }: { query: string; text: string }) {
  return splitLogSearchMatches(text, query).map((segment, index) =>
    segment.match ? (
      <mark className="rounded-[2px] bg-[color:var(--ui-yellow)]/35 text-foreground" key={index}>
        {segment.text}
      </mark>
    ) : (
      <span key={index}>{segment.text}</span>
    )
  )
}

export function LogSearchMatchCount({ label, visible }: { label: string; visible: boolean }) {
  if (!visible) {
    return null
  }

  return (
    <span className="shrink-0 text-[0.65rem] tabular-nums text-(--ui-text-tertiary)" aria-live="polite">
      {label}
    </span>
  )
}
