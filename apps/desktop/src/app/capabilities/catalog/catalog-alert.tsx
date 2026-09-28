import type { ReactNode } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'

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
  return (
    <Alert className="mx-3 w-auto shrink-0" variant="warning">
      <Codicon name="warning" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        {children}
        <Button onClick={onRetry} size="xs" variant="text">
          {retryLabel}
        </Button>
      </AlertDescription>
    </Alert>
  )
}
