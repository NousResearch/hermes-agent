import { useStore } from '@nanostores/react'
import { atom, computed } from 'nanostores'
import { type ComponentProps, type KeyboardEvent, type ReactNode, useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ArrowBarToLeft as IconArrowBarToLeft } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $sidebarOpen, toggleSidebarOpen } from '@/store/layout'

/** A page's sidebar panel follows the chat sidebar's open state: one ⌘B, one collapse button, and
 *  switching between Chats, Agents and Settings never re-opens a panel the user folded away. */
export function usePageSidebarOpen() {
  // Mounting a page that owns a panel tells the band (PageToolbar) to offer "show sidebar" while
  // the panel is folded away, as the chat band does.
  useEffect(() => {
    $pagePanels.set($pagePanels.get() + 1)

    return () => $pagePanels.set($pagePanels.get() - 1)
  }, [])

  return useStore($sidebarOpen)
}

const $pagePanels = atom(0)

/** Below this the list panel folds away (portal-pages.css) and a destination shows its list itself. */
export const PAGE_SIDEBAR_NARROW = '(max-width: 800px)'

/** A page with a sidebar panel is showing. */
export const $pageHasSidebar = computed($pagePanels, count => count > 0)

/** The panel's title row (895:8991 "Agents", 856:2741 "Settings"): the display title at the
 *  panel's 20px inset and the chat sidebar's 30px collapse button at its end. */
export function PageSidebarHead({ title }: { title: string }) {
  const { t } = useI18n()

  return (
    <div className="page-sidebar-head">
      <h2 className="page-nav-title">{title}</h2>
      <Tip label={t.titlebar.hideSidebar}>
        <Button
          aria-label={t.titlebar.hideSidebar}
          className="page-sidebar-collapse"
          onClick={toggleSidebarOpen}
          size="icon"
          variant="ghost"
        >
          <IconArrowBarToLeft aria-hidden />
        </Button>
      </Tip>
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
  children: ReactNode
  title: string
}

/** A list→detail destination's list (Agents 895:8991, Schedules): the chat sidebar's panel at the
 *  stage's leading edge. It folds away with the chat sidebar; the stage keeps the selected item. */
export function PageSidebarPanel({ action, children, title }: PageSidebarPanelProps) {
  const open = usePageSidebarOpen()

  if (!open) {
    return null
  }

  return (
    <aside aria-label={title} className="page-sidebar-panel">
      <PageSidebarHead title={title} />
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
