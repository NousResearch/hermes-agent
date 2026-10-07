import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { answeredAfter } from '@/lib/chat-messages/parts'
import { cn } from '@/lib/utils'

const settled = new Set<string>()

export function AskDirective({ attrs, streaming }: { attrs: Record<string, string>; streaming: boolean }) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const messageId = useAuiState(state => state.message.id)
  const question = (attrs.question ?? '').trim()
  const identity = JSON.stringify([storedId ?? runtimeId, messageId, question])
  const target = view.kind === 'tile' ? `tile:${storedId}` : 'main'

  const options = (attrs.options ?? '')
    .split('|')
    .map(option => option.trim())
    .filter(Boolean)
    .slice(0, 6)

  const wantsInput = attrs.input === 'true' || attrs.input === 'yes'
  const [picked, setPicked] = useState<null | string>(() => (settled.has(identity) ? '' : null))

  const answeredInComposer = answeredAfter(useStore(view.$messages), messageId)

  const closed = picked !== null || answeredInComposer

  if (!question || (options.length === 0 && !wantsInput)) {
    return null
  }

  const submit = (value: string) => {
    if (closed || streaming || !value.trim()) {
      return
    }

    if (requestComposerSubmit(value.trim(), { target })) {
      settled.add(identity)
      setPicked(value.trim())
    }
  }

  return (
    <div
      className="my-3 flex min-w-0 max-w-full flex-col gap-2 overflow-visible duration-300 animate-in fade-in-0 slide-in-from-bottom-2"
      data-onboarding-card
    >
      <div className="text-[13px] font-medium">{question}</div>
      {options.length > 0 && (
        <div className="flex min-w-0 max-w-full flex-wrap gap-2">
          {options.map(option => (
            <button
              className={cn(
                'max-w-full shrink-0 rounded-full border px-3 py-1.5 text-left text-[12px] whitespace-normal wrap-anywhere transition-colors',
                picked === option
                  ? 'border-primary bg-primary text-primary-foreground'
                  : closed
                    ? 'border-border/60 text-muted-foreground/50'
                    : 'border-border bg-card hover:border-primary/50 hover:bg-primary/10'
              )}
              disabled={closed || streaming}
              key={option}
              onClick={() => submit(option)}
              type="button"
            >
              {option}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
