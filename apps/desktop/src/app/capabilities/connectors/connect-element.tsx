import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'

import { type AccountOperation, syncAccountOperation } from './data/account-operations'

export interface ConnectElementProps {
  operation: AccountOperation
  onStopWaiting: () => void
}

export function ConnectElement({ operation, onStopWaiting }: ConnectElementProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const link = operation.targets.find(target => target.connectUrl)?.connectUrl ?? null

  useEffect(() => {
    if (link === null && !operation.settled) {
      void syncAccountOperation(operation.opId)
    }
  }, [link, operation.opId, operation.settled])

  if (operation.settled) {
    return <p className="text-[0.78rem] text-(--ui-text-primary)">{copy.dialog.connectEnded}</p>
  }

  return (
    <div className="grid justify-items-start gap-2">
      <p className="flex items-center gap-2 text-[0.78rem] text-(--ui-text-primary)">
        <Loader2 aria-hidden className="size-3.5 shrink-0 animate-spin text-primary motion-reduce:animate-none" />
        {copy.card.reason.finishSignIn}
      </p>

      <div className="flex flex-wrap items-center gap-2">
        {link ? (
          <Button onClick={() => void window.hermesDesktop?.openExternal?.(link)} size="xs" variant="secondary">
            {copy.dialog.connectOpenAgain}
          </Button>
        ) : null}

        <Button onClick={onStopWaiting} size="xs" variant="text">
          {copy.card.verb.stopWaiting}
        </Button>
      </div>
    </div>
  )
}
