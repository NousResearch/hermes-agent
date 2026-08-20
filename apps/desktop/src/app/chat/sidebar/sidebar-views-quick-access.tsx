import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $activeSavedSidebarViewId,
  $savedSidebarViews,
  applySavedSidebarView,
  type SavedSidebarView,
  savedSidebarViewRequiresProfileSwitch
} from '@/store/sidebar-views'

import { SidebarViewDialog, type SidebarViewDialogState } from './sidebar-view-dialog'

const HOVER_CLOSE_DELAY_MS = 180

export function SidebarSavedViewsQuickAccess({ className }: { className?: string }) {
  const { t } = useI18n()
  const savedViews = useStore($savedSidebarViews).views
  const activeViewId = useStore($activeSavedSidebarViewId)
  const [open, setOpen] = useState(false)
  const [dialog, setDialog] = useState<SidebarViewDialogState | null>(null)
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const openedByPointer = useRef(false)
  const pointerFocus = useRef<HTMLElement | null>(null)

  const cancelClose = useCallback(() => {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current)
      closeTimer.current = null
    }
  }, [])

  const openNow = () => {
    cancelClose()
    openedByPointer.current = true
    pointerFocus.current = globalThis.document?.activeElement as HTMLElement | null
    setOpen(true)
    globalThis.requestAnimationFrame(() => {
      globalThis.requestAnimationFrame(() => pointerFocus.current?.focus())
    })
  }

  const scheduleClose = () => {
    cancelClose()
    closeTimer.current = setTimeout(() => setOpen(false), HOVER_CLOSE_DELAY_MS)
  }

  useEffect(() => cancelClose, [cancelClose])

  if (savedViews.length === 0) {
    return null
  }

  const label = t.sidebar.viewMenu.savedViews

  const selectView = (view: SavedSidebarView) => {
    setOpen(false)

    if (savedSidebarViewRequiresProfileSwitch(view)) {
      setDialog({ kind: 'apply', view })
    } else {
      applySavedSidebarView(view.id)
    }
  }

  return (
    <>
      <div className="grid size-6 place-items-center">
        <DropdownMenu modal={false} onOpenChange={setOpen} open={open}>
          <DropdownMenuTrigger asChild>
            <Button
              aria-label={label}
              className={cn(className, open && 'bg-(--ui-control-active-background) text-foreground opacity-100')}
              onKeyDown={() => {
                openedByPointer.current = false
              }}
              onPointerEnter={openNow}
              onPointerLeave={scheduleClose}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="eye" size="0.75rem" />
            </Button>
          </DropdownMenuTrigger>

          <DropdownMenuContent
            align="start"
            aria-label={label}
            className="w-52"
            onCloseAutoFocus={event => {
              if (openedByPointer.current) {
                event.preventDefault()
              }

              openedByPointer.current = false
              pointerFocus.current = null
            }}
            onFocusOutside={event => {
              if (openedByPointer.current) {
                event.preventDefault()
              }
            }}
            onPointerEnter={cancelClose}
            onPointerLeave={scheduleClose}
          >
            <DropdownMenuLabel>{label}</DropdownMenuLabel>
            {savedViews.map(view => (
              <DropdownMenuItem aria-label={view.name} key={view.id} onSelect={() => selectView(view)}>
                <span className="flex w-3 shrink-0 items-center justify-center">
                  {activeViewId === view.id && <Codicon name="check" size="0.75rem" />}
                </span>
                <span className="truncate">{view.name}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <SidebarViewDialog dialog={dialog} onClose={() => setDialog(null)} />
    </>
  )
}
