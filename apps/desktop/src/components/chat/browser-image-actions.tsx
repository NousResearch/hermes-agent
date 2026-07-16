'use client'

import {
  BrowserTabLimitError,
  BrowserUnsupportedUrlError,
  openBrowserQc,
  openBrowserSurface
} from '@/app/browser/store'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { CheckCircle2, Globe } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

export interface BrowserImageActionsProps {
  className?: string
  src?: string
}

export function BrowserImageActions({ className, src }: BrowserImageActionsProps) {
  const { t } = useI18n()

  if (!src) {
    return null
  }

  const openImageInBrowser = (open: (input: { url: string }) => unknown) => {
    try {
      open({ url: src })
    } catch (error) {
      if (error instanceof BrowserTabLimitError) {
        notifyError(undefined, t.desktop.browserTabLimit)

        return
      }

      if (error instanceof BrowserUnsupportedUrlError) {
        notifyError(undefined, t.desktop.browserInvalidUrl)

        return
      }

      notifyError(error, t.common.failed)
    }
  }

  return (
    <span
      className={cn(
        'absolute left-2 top-2 z-10 flex gap-1 opacity-0 transition-opacity focus-within:opacity-100',
        className
      )}
    >
      <Button
        aria-label={t.desktop.openInBrowser}
        onClick={event => {
          event.stopPropagation()
          openImageInBrowser(openBrowserSurface)
        }}
        size="icon-sm"
        title={t.desktop.openInBrowser}
        type="button"
        variant="secondary"
      >
        <Globe />
      </Button>
      <Button
        aria-label={t.desktop.openInQc}
        onClick={event => {
          event.stopPropagation()
          openImageInBrowser(openBrowserQc)
        }}
        size="icon-sm"
        title={t.desktop.openInQc}
        type="button"
        variant="secondary"
      >
        <CheckCircle2 />
      </Button>
    </span>
  )
}
