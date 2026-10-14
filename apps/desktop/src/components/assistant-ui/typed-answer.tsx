'use client'

import { USER_BUBBLE_BASE_CLASS } from '@/components/assistant-ui/thread/user-message'
import { UserMessageText } from '@/components/assistant-ui/thread/user-message-text'
import { cn } from '@/lib/utils'

/** Words typed in the composer while a card waited, drawn as the user's bubble where the card stood.
 *  They live only in the tool result's `said`: a user row between a tool call and its result breaks the transcript. */
export function TypedAnswer({ text }: { text: string }) {
  return (
    <div className="my-1.5 flex w-full min-w-0 flex-col items-stretch" data-slot="typed-answer">
      <div
        className={cn(
          USER_BUBBLE_BASE_CLASS,
          'border-(--ui-stroke-tertiary) text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground/95'
        )}
      >
        <UserMessageText className="wrap-anywhere" text={text} />
      </div>
    </div>
  )
}
