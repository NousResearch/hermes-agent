import { type ReactNode, useState } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'

/** A load failure above the catalog results that keeps what's already shown. */
export function CatalogAlert({
  title,
  children,
  retryLabel,
  onRetry
}: {
  title: string
  children?: ReactNode
  retryLabel: string
  onRetry: () => void
}) {
  const { t } = useI18n()
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) {return null}

  return (
    <Alert className="mx-6 w-auto shrink-0 pr-12" variant="warning">
      <Codicon name="warning" />
      <AlertTitle>{title}</AlertTitle>
      <Button aria-label={t.common.close} className="absolute right-3 top-3" onClick={() => setDismissed(true)} size="icon-xs" variant="ghost"><Codicon name="close" /></Button>
      <AlertDescription>
        {children}
        <Button onClick={onRetry} size="xs" variant="text">
          {retryLabel}
        </Button>
      </AlertDescription>
    </Alert>
  )
}
