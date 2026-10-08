import type { ComponentType, FormEvent, ReactNode } from 'react'

import { ClarifyConfirmBar } from '@/components/assistant-ui/clarify/core/confirm-bar'
import { CLARIFY_ICON_CLASS, ClarifyShell } from '@/components/assistant-ui/clarify/core/shell'

/**
 * The real clarify card: an optional heading, the shell with its kind label and icon, and the
 * Skip / Confirm and continue bar. Enter anywhere in the card confirms.
 */
export function StepCard({
  canConfirm,
  children,
  icon: Icon,
  kind,
  onConfirm,
  onSkip,
  title
}: {
  canConfirm: boolean
  children: ReactNode
  icon: ComponentType<{ 'aria-hidden'?: boolean; className?: string }>
  kind: string
  onConfirm: () => void
  onSkip: () => void
  title?: string
}) {
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (canConfirm) {
      onConfirm()
    }
  }

  return (
    <form className="grid gap-4" onSubmit={submit}>
      {title ? <h3 className="text-lg font-semibold tracking-tight">{title}</h3> : null}
      <ClarifyShell className="grid gap-3">
        <div className="flex items-start gap-2">
          <span className="flex-1 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">{kind}</span>
          <Icon aria-hidden className={CLARIFY_ICON_CLASS} />
        </div>
        {children}
      </ClarifyShell>
      <ClarifyConfirmBar canConfirm={canConfirm} disabled={false} onSkip={onSkip} submitting={false} />
    </form>
  )
}
