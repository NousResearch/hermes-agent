import { Badge } from '@/components/ui/badge'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { SessionAttentionKind } from '@/store/session-dot-state'

// One glyph per kind, and the same amber the status dot already speaks: the
// chip is the dot's words, not a second vocabulary.
const GLYPH: Record<SessionAttentionKind, string> = {
  approval: 'shield',
  question: 'question'
}

/** The sidebar's "this one needs you" cue, in words: a small amber chip on a
 *  row whose turn is parked on a blocking prompt. The 6px status dot already
 *  turns amber for this — but amber next to the accent's orange is not a
 *  difference anyone catches while working in another chat, and the OS
 *  notification names no session. This is the row saying it out loud.
 *  Status, not identity: it appears and disappears with the prompt. */
export function SessionAttentionChip({ className, kind }: { className?: string; kind: SessionAttentionKind }) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const label = kind === 'approval' ? r.attentionApproval : r.attentionQuestion

  return (
    <Tip label={r.waitingForAnswer}>
      <Badge
        aria-label={r.needsInput}
        className={cn('gap-0.5 px-1 py-px text-[0.625rem] leading-none', className)}
        data-attention={kind}
        role="status"
        size="xs"
        variant="warn"
      >
        <Codicon name={GLYPH[kind]} size="0.625rem" />
        {label}
      </Badge>
    </Tip>
  )
}
