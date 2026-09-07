import { ThreadPrimitive, useAuiState } from '@assistant-ui/react'
import { type ComponentProps, useState } from 'react'

import { SCAFFOLD_LABEL_CLASS, SCAFFOLD_META_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { useI18n } from '@/i18n'

interface BackgroundUpdatesProps {
  components: ComponentProps<typeof ThreadPrimitive.MessageByIndex>['components']
  count: number
  indices: number[]
}

/** Presentation only: keep the actual handoff replies available through the
 * existing message renderer. Never guess from prose whether a result matters. */
export function BackgroundUpdates({ components, count, indices }: BackgroundUpdatesProps) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)

  const needsAttention = useAuiState(
    s =>
      (s.thread.isRunning && indices.includes(s.thread.messages.length - 1)) ||
      indices.some(index => {
        const message = s.thread.messages[index]

        return (
          message?.role === 'assistant' && (message.status?.type === 'running' || message.status?.type === 'incomplete')
        )
      })
  )

  const open = expanded || needsAttention

  return (
    <div className="flex min-w-0 flex-col gap-2" data-slot="background-updates">
      <div data-conversation-scaffold>
        <ScaffoldRow onToggle={needsAttention ? undefined : () => setExpanded(value => !value)} open={open}>
          <span className={SCAFFOLD_LABEL_CLASS}>{t.assistant.thread.backgroundUpdates}</span>
          <span className={SCAFFOLD_META_CLASS}>{count}</span>
        </ScaffoldRow>
      </div>
      {open && (
        <div className="flex min-w-0 flex-col gap-(--conversation-turn-gap)">
          {indices.map(index => (
            <ThreadPrimitive.MessageByIndex components={components} index={index} key={index} />
          ))}
        </div>
      )}
    </div>
  )
}
