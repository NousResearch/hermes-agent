/**
 * `::ask{...}` — the model's interactive question, inline in its message.
 *
 * The conversational counterpart of a wall of text: whenever the agent needs
 * a decision, it emits ONE line and the transcript renders real controls —
 * option pills; a single click submits the pick as a visible user turn.
 * Works in every session (it is a core transcript directive, not an
 * onboarding-only one), so dashboard button responses, refinement dialogues,
 * and ordinary chats can all fork interactively.
 *
 *   ::ask{question="Which angle leads?" options="Lead story|Exclusive|Embargoed brief"}
 *   ::ask{question="Paste the runway number" input="true"}
 *
 * Options are pipe-separated. `input="true"` means a typed answer is welcome —
 * that keeps the QUESTION rendering even with no options, but it draws no
 * input row of its own: the composer is always right below the transcript,
 * and a second "type here" bar beside it read as clutter (first live-run
 * feedback). A pick submits VISIBLY so the user sees their choice become a
 * turn.
 */

import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { cn } from '@/lib/utils'

// Picked questions, module-scoped: transcript virtualization remounts
// directives with fresh local state, which would resurrect a settled picker.
// A repeated question in a later message or another session is a new choice.
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

  if (!question || (options.length === 0 && !wantsInput)) {
    return null
  }

  const submit = (value: string) => {
    if (picked !== null || streaming || !value.trim()) {
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
                  : picked !== null
                    ? 'border-border/60 text-muted-foreground/50'
                    : 'border-border bg-card hover:border-primary/50 hover:bg-primary/10'
              )}
              disabled={picked !== null || streaming}
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
