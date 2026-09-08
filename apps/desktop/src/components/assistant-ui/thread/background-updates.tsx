import { ThreadPrimitive, useAuiState } from '@assistant-ui/react'
import { type ComponentProps, useState } from 'react'

import { SCAFFOLD_LABEL_CLASS, SCAFFOLD_META_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { useI18n } from '@/i18n'

interface BackgroundUpdatesProps {
  components: ComponentProps<typeof ThreadPrimitive.MessageByIndex>['components']
  count: number
  indices: number[]
}

/** Only internal notifications are collapsible. Their assistant replies stay
 * in the transcript, regardless of what triggered them or what they say. */
export function BackgroundUpdates({ components, count, indices }: BackgroundUpdatesProps) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)

  const needsAttention = useAuiState(s => {
    let replyIndex = indices[indices.length - 1] + 1
    let hasReply = false

    // A settled tool/reasoning row is not a deliverable. Check the whole
    // response sequence, stopping at the next user or system boundary.
    for (; replyIndex < s.thread.messages.length; replyIndex++) {
      const reply = s.thread.messages[replyIndex]

      if (reply.role !== 'assistant') {
        break
      }

      if (reply.status?.type !== 'complete') {
        return true
      }

      hasReply ||=
        reply.metadata?.custom?.interim !== true &&
        reply.content.some(part => part.type === 'text' && part.text.trim().length > 0)
    }

    return !hasReply || (s.thread.isRunning && replyIndex === s.thread.messages.length)
  })

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
