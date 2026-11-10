import { type ReactNode, useState } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SurfaceCloseButton } from '@/components/ui/surface-close'
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

  if (dismissed) {
    return null
  }

  return (
    <Alert className="mx-6 w-auto shrink-0 grid-cols-[auto_minmax(0,1fr)_auto]" variant="warning">
      <Codicon name="warning" />
      <AlertTitle>{title}</AlertTitle>
      <SurfaceCloseButton
        aria-label={t.common.close}
        className="col-start-3 row-start-1"
        onClick={() => setDismissed(true)}
      />
      <AlertDescription className="col-start-2">
        {children}
        <Button onClick={onRetry} size="xs" variant="text">
          {retryLabel}
        </Button>
      </AlertDescription>
    </Alert>
  )
}
