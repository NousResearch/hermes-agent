import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'

import type { CatalogEntry } from './catalog-data'

interface CatalogDetailDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** On-screen order to page through; wraps at both ends. */
  order: CatalogEntry[]
  selectedId: string
  onSelect: (id: string) => void
  children: ReactNode
}

const EDITABLE = 'input, textarea, [contenteditable="true"]'

export function CatalogDetailDialog({
  open,
  onOpenChange,
  order,
  selectedId,
  onSelect,
  children
}: CatalogDetailDialogProps) {
  const { t } = useI18n()
  const pageable = order.length > 1

  const step = (delta: number) => {
    if (!pageable) {
      return
    }
    const index = order.findIndex(entry => entry.id === selectedId)
    onSelect(order[(index + delta + order.length) % order.length].id)
  }

  const pager = (delta: -1 | 1) => (
    <Button
      aria-label={delta < 0 ? t.ui.pagination.previous : t.ui.pagination.next}
      className={
        delta < 0
          ? 'absolute -left-14 top-1/2 -translate-y-1/2 max-sm:left-2'
          : 'absolute -right-14 top-1/2 -translate-y-1/2 max-sm:right-2'
      }
      onClick={() => step(delta)}
      size="icon-lg"
      variant="ghost"
    >
      <Codicon name={delta < 0 ? 'chevron-left' : 'chevron-right'} />
    </Button>
  )

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        aria-describedby={undefined}
        bodyClassName="gap-5"
        chrome={
          pageable && (
            <>
              {pager(-1)}
              {pager(1)}
            </>
          )
        }
        className="max-w-2xl"
        // Focus returns to the card of the entry the user paged to, when it's rendered.
        onCloseAutoFocus={event => {
          event.preventDefault()
          document
            .querySelector<HTMLElement>(`[data-entry-id="${CSS.escape(selectedId)}"] [aria-haspopup="dialog"]`)
            ?.focus({ preventScroll: true })
        }}
        onKeyDown={event => {
          if (
            (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') ||
            (event.target as HTMLElement).closest(EDITABLE)
          ) {
            return
          }
          event.preventDefault()
          step(event.key === 'ArrowLeft' ? -1 : 1)
        }}
      >
        {children}
      </DialogContent>
    </Dialog>
  )
}
