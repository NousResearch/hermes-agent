import { useCallback, useEffect, useState } from 'react'

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import type { HermesWindowInfo } from '@/global'
import { useI18n } from '@/i18n'
import { AlertCircle, AppWindow, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

/**
 * Lists open desktop windows (via the hermes-eats-world sidecar, exposed as
 * window.hermesDesktop.listWindows) and lets the user attach one as a live
 * target. Fetches fresh on each open so the list reflects the current desktop.
 */
export function WindowPickerDialog({
  open,
  onOpenChange,
  onSelect
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSelect: (win: HermesWindowInfo) => void
}) {
  const { t } = useI18n()
  const c = t.composer

  const [windows, setWindows] = useState<HermesWindowInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)

  const load = useCallback(async () => {
    const list = window.hermesDesktop?.listWindows

    if (!list) {
      setError(true)

      return
    }

    setLoading(true)
    setError(false)

    try {
      const result = await list()
      // Newest/topmost-ish first is fine; sidecar returns desktop z-order.
      setWindows(Array.isArray(result) ? result.filter(w => (w.name || '').trim()) : [])
    } catch {
      setError(true)
    } finally {
      setLoading(false)
    }
  }, [])

  // Refetch every time the dialog opens — the desktop changes between uses.
  useEffect(() => {
    if (open) {
      void load()
    }
  }, [open, load])

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md gap-3">
        <DialogHeader>
          <DialogTitle>{c.windowPickerTitle}</DialogTitle>
          <DialogDescription>{c.windowPickerDesc}</DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center gap-2 px-1 py-6 text-sm text-(--ui-text-tertiary)">
            <Loader2 className="size-4 animate-spin" />
            <span>{c.windowPickerLoading}</span>
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 px-1 py-6 text-sm text-destructive">
            <AlertCircle className="size-4" />
            <span>{c.windowPickerError}</span>
          </div>
        ) : windows.length === 0 ? (
          <div className="px-1 py-6 text-sm text-(--ui-text-tertiary)">{c.windowPickerEmpty}</div>
        ) : (
          <ul className="grid max-h-80 gap-1 overflow-y-auto">
            {windows.map(win => (
              <li key={`${win.pid}:${win.name}:${win.class_name}`}>
                <button
                  className={cn(
                    'group/win flex w-full cursor-pointer items-start gap-2.5 rounded-md border border-transparent px-2.5 py-2 text-left transition-colors',
                    'hover:border-(--ui-stroke-tertiary) hover:bg-(--ui-control-hover-background)',
                    'focus-visible:border-(--ui-stroke-tertiary) focus-visible:bg-(--ui-control-hover-background) focus-visible:outline-none'
                  )}
                  onClick={() => {
                    onSelect(win)
                    onOpenChange(false)
                  }}
                  type="button"
                >
                  <AppWindow className="mt-0.5 size-3.5 shrink-0 text-(--ui-text-tertiary) group-hover/win:text-foreground" />
                  <span className="grid min-w-0 gap-0.5">
                    <span className="truncate text-sm font-medium text-foreground">{win.name}</span>
                    <span className="truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                      {win.class_name ? `${win.class_name} · PID ${win.pid}` : `PID ${win.pid}`}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </DialogContent>
    </Dialog>
  )
}
