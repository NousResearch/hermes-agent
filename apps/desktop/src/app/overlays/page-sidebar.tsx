import { useStore } from '@nanostores/react'
import { atom, computed } from 'nanostores'
import { type ComponentProps, type KeyboardEvent, type ReactNode, useEffect } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { cn } from '@/lib/utils'
import { $sidebarOpen, toggleSidebarOpen } from '@/store/layout'

import { $widthClass, BELOW_WIDE_QUERY } from '../redesign/breakpoints'
import { $shellSheet, toggleShellSheet } from '../shell/shell-sheet'

/** A page's sidebar panel follows the chat sidebar's open state on a wide window: one ⌘B, and one
 *  place that folds it — the open destination's rail button, as Chats folds the chat sidebar.
 *  Switching between Agents, Schedules and Settings on the band or by link never re-opens a panel the
 *  user folded away; picking one from the rail shows it whole. Below wide it is a sheet the band's menu
 *  opens (app/shell/shell-sheet.ts), and the docked preference is left alone for when the window
 *  widens again. */
export function usePageSidebarOpen() {
  // Mounting a page that owns a panel tells the band (PageToolbar) to offer "show sidebar" while
  // the panel is folded away, as the chat band does, and below wide to open it as a sheet.
  useEffect(() => {
    $pagePanels.set($pagePanels.get() + 1)

    return () => $pagePanels.set($pagePanels.get() - 1)
  }, [])

  const docked = useStore($sidebarOpen)
  const wide = useStore($widthClass) === 'wide'
  const sheet = useStore($shellSheet) === 'page'

  return wide ? docked : sheet
}

/** The open destination's rail button and the band's menu: fold the docked panel, or toggle its sheet. */
export function togglePageSidebar(): void {
  if ($widthClass.get() === 'wide') {
    toggleSidebarOpen()
  } else {
    toggleShellSheet('page')
  }
}

const $pagePanels = atom(0)

/** Below wide the list panel is a sheet (portal-pages.css), and a destination that needs its list on
 *  screen (the Agents landing, Schedules) shows it in the stage itself. */
export const PAGE_SIDEBAR_NARROW = BELOW_WIDE_QUERY

/** A page with a sidebar panel is showing. */
export const $pageHasSidebar = computed($pagePanels, count => count > 0)

/** The panel's title row (895:8991 "Agents", 16292:16360 "Settings", 1163:17585 "Library"): the
 *  display title at the panel's inset, an optional caption under it and action at its end. The
 *  panel folds from its rail destination, never from a button of its own. */
export function PageSidebarHead({ action, caption, title }: { action?: ReactNode; caption?: string; title: string }) {
  return (
    <div className="page-sidebar-head">
      {caption ? (
        <div className="page-sidebar-titles">
          <h2 className="page-nav-title">{title}</h2>
          <p className="page-sidebar-summary">{caption}</p>
        </div>
      ) : (
        <h2 className="page-nav-title">{title}</h2>
      )}
      {action}
    </div>
  )
}

/** Up/Down, Home and End walk the panel's rows, like the chat sidebar's list. */
function moveRowFocus(event: KeyboardEvent<HTMLDivElement>) {
  const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('.page-sidebar-row')]
  const at = rows.indexOf(document.activeElement as HTMLElement)

  const target = {
    ArrowDown: rows[at + 1],
    ArrowUp: rows[at - 1],
    End: rows.at(-1),
    Home: rows[0]
  }[event.key]

  if (at >= 0 && target) {
    event.preventDefault()
    target.focus()
  }
}

interface PageSidebarPanelProps {
  /** The 40px row under the title that starts something new ("New agent", "New schedule"). */
  action?: ReactNode
  /** A one-line status under the title (Library 1163:17700 "// 799 items · Updated 2:24 PM"). */
  caption?: string
  /** A control at the end of the title row (Library's search, 1163:17705). */
  headAction?: ReactNode
  children: ReactNode
  title: string
}

/** A list→detail destination's list (Agents 895:8991, Schedules): the chat sidebar's panel at the
 *  stage's leading edge. It folds away with the chat sidebar; the stage keeps the selected item. */
export function PageSidebarPanel({ action, caption, children, headAction, title }: PageSidebarPanelProps) {
  const open = usePageSidebarOpen()

  if (!open) {
    return null
  }

  return (
    <aside aria-label={title} className="page-sidebar-panel" data-page-panel="">
      <PageSidebarHead action={headAction} caption={caption} title={title} />
      {action}
      <div className="page-sidebar-scroll">{children}</div>
    </aside>
  )
}

/** A run of rows the arrow keys walk; groups (captions and their rows) stay inside one list. */
export function PageSidebarList({ children }: { children: ReactNode }) {
  return (
    <div className="page-sidebar-list" onKeyDown={moveRowFocus}>
      {children}
    </div>
  )
}

/** A group caption in the panel ("3 AGENTS", "BLUEPRINTS"): 10px uppercase (895:9473). */
export function PageSidebarCaption({ children }: { children: ReactNode }) {
  return <p className="page-sidebar-caption">{children}</p>
}

interface PageSidebarRowProps extends ComponentProps<'button'> {
  current?: boolean
}

/** One 40px row; the open item wears the neutral selection (895:9473). */
export function PageSidebarRow({ className, current = false, ...props }: PageSidebarRowProps) {
  return (
    <RowButton aria-current={current ? 'page' : undefined} className={cn('page-sidebar-row', className)} {...props} />
  )
}
