/**
 * What every in-chat onboarding card is made of: the frame it sits in, the
 * props it receives, and the one thing it does when the user is finished —
 * report the pick so the model moves on.
 */

import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export interface CardProps {
  /** The directive's raw attrs — the model-written payload. */
  attrs: Record<string, string>
  /** True while the surrounding turn is still streaming — same card, no clicks. */
  locked: boolean
}

export function useCardCommit() {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const target = view.kind === 'tile' ? `tile:${storedId}` : 'main'
  const [done, setDone] = useState(false)

  const commit = (summary: string): boolean => {
    const sent = requestComposerSubmit(`[setup] ${summary}`, { displayKind: 'hidden', target })

    if (sent) {
      setDone(true)
    }

    return sent
  }

  return { commit, done }
}

/** No chrome — the picker sits directly in the transcript like any other
 *  message content. The interaction IS the affordance; a border would make it
 *  read as a form. */
export function CardFrame({
  children,
  continueLabel = 'Continue',
  disabled = false,
  done,
  locked = false,
  onContinue
}: {
  children: React.ReactNode
  /** The action, named for what it does when the default reads as a shrug —
   *  "Continue with 2" tells them the picks registered. */
  continueLabel?: string
  disabled?: boolean
  done: boolean
  locked?: boolean
  onContinue: () => void
}) {
  return (
    <div
      className={cn(
        'my-3 grid w-full min-w-0 max-w-md gap-4 duration-300 animate-in fade-in-0 slide-in-from-bottom-2',
        done && 'opacity-75 transition-opacity duration-500'
      )}
      data-onboarding-card
      inert={locked || undefined}
    >
      {children}
      <div className="flex justify-start">
        <Button
          className={cn(done && 'scale-95 transition-transform duration-200')}
          disabled={done || disabled || locked}
          onClick={onContinue}
          size="sm"
        >
          {done ? '✓ Done' : continueLabel}
        </Button>
      </div>
    </div>
  )
}
