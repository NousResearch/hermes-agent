/**
 * The parts every in-chat onboarding card shares: the frame it renders in, the props it receives, and the commit
 * helper that submits the pick as a hidden [setup] message so the model moves on.
 */

import { useStore } from '@nanostores/react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { $onboardingAnswers, markStepCommitted } from '@/store/onboarding-answers'

export interface CardProps {
  /** The directive's raw attrs, written by the model. */
  attrs: Record<string, string>
  /** True while the surrounding turn is still streaming; the card renders but does not accept clicks. */
  locked: boolean
}

/** `step` names the card so Done survives the card remounting: the hidden
 *  submit and the turn-end hydrate both rebuild the transcript, and a flag in
 *  component state came back false each time. */
export function useCardCommit(step: string) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const target = view.kind === 'tile' ? `tile:${storedId}` : 'main'
  const done = useStore($onboardingAnswers).committed.includes(step)

  const commit = (summary: string): boolean => {
    const sent = requestComposerSubmit(`[setup] ${summary}`, { displayKind: 'hidden', target })

    if (sent) {
      markStepCommitted(step)
    }

    return sent
  }

  return { commit, done }
}

/** The frame draws no border or background, so the picker reads as message content in the transcript rather than as
 *  a form. */
export function CardFrame({
  children,
  continueLabel = 'Continue',
  disabled = false,
  done,
  locked = false,
  onContinue
}: {
  children: React.ReactNode
  /** The action, named for what it does when the default label says nothing specific.
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
