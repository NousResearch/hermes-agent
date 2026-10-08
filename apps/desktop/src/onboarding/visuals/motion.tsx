/**
 * The questionnaire's two motion effects (D26), both answering something the person did, at the
 * house timing (`components/ui/card-stack.tsx`): the step change, and the confirmed answer moving
 * into the trail through a shared `layoutId`. Reduced motion settles both at once.
 */

import { motion, useReducedMotion } from 'motion/react'
import type { ReactNode } from 'react'

import type { StepId } from '../flow'

const EASE = [0.23, 1, 0.32, 1] as const
const MOVE = { duration: 0.22, ease: EASE }
const STILL = { duration: 0 }

export const answerLayoutId = (stepId: StepId) => `questionnaire-answer-${stepId}`

/** One step on screen: fades in with 4px of rise, fades out the same way. */
export function StepTransition({ children }: { children: ReactNode }) {
  const reduced = useReducedMotion()

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      initial={{ opacity: 0, y: 4 }}
      transition={reduced ? STILL : MOVE}
    >
      {children}
    </motion.div>
  )
}

/** The picked option's outline; the trail chip for the same step wears the same id, so confirming moves it there. */
export function AnswerMark({ stepId }: { stepId: StepId }) {
  const reduced = useReducedMotion()

  return (
    <motion.span
      aria-hidden
      className="pointer-events-none absolute inset-0 rounded-[inherit] border border-primary"
      layoutId={answerLayoutId(stepId)}
      transition={reduced ? STILL : MOVE}
    />
  )
}

/** A trail chip: reflows by position as answers join, and receives the answer's outline. */
export function TrailChip({
  children,
  disabled,
  label,
  onClick,
  stepId
}: {
  children: ReactNode
  disabled: boolean
  label: string
  onClick: () => void
  stepId: StepId
}) {
  const reduced = useReducedMotion()

  return (
    <motion.button
      aria-label={label}
      className="relative inline-flex max-w-48 items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs text-foreground transition-colors enabled:hover:bg-accent/60 disabled:opacity-60"
      disabled={disabled}
      layout="position"
      onClick={onClick}
      transition={reduced ? STILL : MOVE}
      type="button"
    >
      <span className="truncate">{children}</span>
      <AnswerMark stepId={stepId} />
    </motion.button>
  )
}
