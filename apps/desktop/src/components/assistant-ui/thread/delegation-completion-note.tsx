import { type FC, useId, useState } from 'react'

import { type DelegationCompletion } from '@/components/assistant-ui/thread/delegation-completion'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { LogView } from '@/components/ui/log-view'
import { useI18n } from '@/i18n'

export const DelegationCompletionNote: FC<{ completion: DelegationCompletion }> = ({ completion }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const [expanded, setExpanded] = useState(false)
  const payloadId = useId()
  const title = completion.kind === 'batch' ? copy.subagentsCompleted : copy.subagentCompleted
  const toggleLabel = expanded ? copy.hideFullPayload : copy.showFullPayload

  return (
    <section
      aria-label={title}
      className="mx-auto flex w-full max-w-[44rem] flex-col gap-1.5 border-l border-(--ui-stroke-tertiary) px-3 py-1.5 text-[0.6875rem] leading-5 text-(--ui-text-secondary)"
      data-slot="delegation-completion-note"
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="check-all" size="0.75rem" />
        <strong className="text-(--ui-text-primary)">{title}</strong>
        {completion.status && <span className="truncate text-(--ui-text-tertiary)">{completion.status}</span>}
        {completion.duration && (
          <span className="ml-auto shrink-0 font-mono text-(--ui-text-tertiary)">{completion.duration}</span>
        )}
      </div>
      {completion.goal && <div className="truncate text-(--ui-text-secondary)">{completion.goal}</div>}
      {completion.summary && <div className="line-clamp-2 text-(--ui-text-tertiary)">{completion.summary}</div>}
      <button
        aria-controls={payloadId}
        aria-expanded={expanded}
        aria-label={toggleLabel}
        className="flex w-fit items-center gap-1 text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)"
        onClick={() => setExpanded(value => !value)}
        type="button"
      >
        <DisclosureCaret aria-hidden open={expanded} />
        <span>{toggleLabel}</span>
      </button>
      {expanded && (
        <LogView className="max-h-64" id={payloadId}>
          {completion.raw}
        </LogView>
      )}
    </section>
  )
}
