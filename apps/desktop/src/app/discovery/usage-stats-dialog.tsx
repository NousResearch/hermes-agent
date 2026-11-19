import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/icons'
import { SHARED_METRICS_DOCS_URL, type SharedMetricsChoice } from '@/store/shared-metrics'

import { useUsageStatsCopy } from './usage-stats-copy'

interface UsageStatsDialogProps {
  choose: (choice: SharedMetricsChoice) => Promise<void>
  onClose: () => void
  /** An earlier build could save "No thanks" unseen: say why it asks again. */
  reask: boolean
  saving: boolean
}

/** Where focus goes as the dialog closes: back to what opened it (the card's Details); when that is
 *  gone (the card answered, the tray closed), the bell for a tray opener, else the chat sidebar's New
 *  chat. Radix would aim at a DialogTrigger, which this dialog has none of, and drop focus on body. */
function returnFocus(opener: Element | null): HTMLElement | null {
  if (opener instanceof HTMLElement && opener.isConnected) {
    return opener
  }

  const fromTray = opener?.closest('[data-slot="notification-tray"]')

  return document.querySelector<HTMLElement>(
    fromTray ? '[data-nav-spot="notifications"]' : '[data-slot="new-session-split"] button'
  )
}

/** Neither answer is focused for the person: the dialog itself takes focus, so Tab reaches them in
 *  order and a screen reader reads the title first. */
function focusDialogItself(event: Event) {
  event.preventDefault()

  if (event.target instanceof HTMLElement) {
    event.target.focus()
  }
}

/**
 * "Help improve Hermes" (Figma 1365:34379): the redesign's short explainer for the usage-stats offer,
 * opened from the sidebar card's Details or the tray. Painted by the classic consent host, which keeps
 * every rule about when the question may be asked. Neither answer takes focus (the dialog itself
 * does); "Not now", × and Escape only close it.
 */
export function UsageStatsDialog({ choose, onClose, reask, saving }: UsageStatsDialogProps) {
  const copy = useUsageStatsCopy().dialog
  const { t } = useI18n()
  // What had focus as the dialog opened, read on the first render, before the dialog takes focus.
  const [opener] = useState(() => document.activeElement)

  return (
    <Dialog onOpenChange={open => !open && onClose()} open>
      <DialogContent
        className="max-w-md"
        onCloseAutoFocus={event => {
          event.preventDefault()
          returnFocus(opener)?.focus()
        }}
        onOpenAutoFocus={focusDialogItself}
      >
        <DialogHeader>
          <DialogTitle>{copy.title}</DialogTitle>
        </DialogHeader>
        <div className="grid gap-3 text-ds-body text-ds-text-2">
          {reask && <p>{t.sharedMetrics.stripReaskBody}</p>}
          <p>{copy.intro}</p>
          <p>{copy.sending}</p>
          <p>
            <Button asChild size="inline" variant="link">
              <a href={SHARED_METRICS_DOCS_URL} rel="noreferrer" target="_blank">
                {copy.seeShared}
                <ExternalLink />
              </a>
            </Button>
          </p>
        </div>
        <DialogFooter>
          <Button disabled={saving} onClick={() => void choose('share')} type="button">
            {copy.share}
          </Button>
          <Button onClick={onClose} type="button" variant="ghost">
            {copy.notNow}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
