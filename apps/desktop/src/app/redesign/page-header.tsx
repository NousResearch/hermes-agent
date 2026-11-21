import './portal-page-header.css'

import { type ComponentProps, type KeyboardEvent, type ReactNode, useRef } from 'react'

import { tabMetaContent } from '@/components/ui/tab-dropdown'
import { cn } from '@/lib/utils'

// The redesign's one page header (DESIGN.md › Page header): every routed page with a title and/or
// tabs renders it, so the H1, its description, the actions and the tab bar sit at the same place on
// every page. Geometry is token-driven (tokens/space.css `--portal-page-*`); the page's own column
// rules place it horizontally.

interface PageHeaderProps extends Omit<ComponentProps<'header'>, 'title'> {
  title: ReactNode
  /** One line under the H1 (a page description, an agent's status meta). */
  description?: ReactNode
  /** A tile before the titles (the agent avatar); it spans the title and description rows. */
  leading?: ReactNode
  /** Controls at the end of the H1 row. */
  actions?: ReactNode
  /** A `PageTabs` row. With tabs the title block always reserves its full height, so the tab bar
   *  lands at one y whether or not a page has a description. */
  tabs?: ReactNode
}

export function PageHeader({ actions, className, description, leading, tabs, title, ...props }: PageHeaderProps) {
  return (
    <header className={cn('portal-page-header', className)} data-tabs={tabs ? '' : undefined} {...props}>
      <div className="portal-page-header-title" data-leading={leading ? '' : undefined}>
        {leading && <div className="portal-page-header-leading">{leading}</div>}
        <h1>{title}</h1>
        {description && <div className="portal-page-header-description">{description}</div>}
        {actions && <div className="portal-page-header-actions">{actions}</div>}
      </div>
      {tabs}
    </header>
  )
}

export interface PageTab {
  id: string
  label: string
  /** A count after the label: `null` while it loads (a reserved placeholder), `undefined` for none. */
  meta?: number | string | null
}

interface PageTabsProps {
  /** The tab list's accessible name. */
  label: string
  tabs: readonly PageTab[]
  value: string
  onChange: (id: string) => void
}

const STEP: Record<string, (index: number, count: number) => number> = {
  ArrowRight: (index, count) => (index + 1) % count,
  ArrowLeft: (index, count) => (index - 1 + count) % count,
  Home: () => 0,
  End: (_index, count) => count - 1
}

const MIRRORED: Record<string, string> = { ArrowLeft: 'ArrowRight', ArrowRight: 'ArrowLeft' }

/** The page tab bar (877:4200): equal-width tabs over a hairline rail, a 2px brand underline on the
 *  selected one. Every tab takes the widest tab's width and a count reserves its room while it
 *  loads, so neither switching tabs nor a count arriving moves anything. Arrow keys, Home and End
 *  move between tabs and select them (WAI-ARIA tabs, automatic activation). */
export function PageTabs({ label, onChange, tabs, value }: PageTabsProps) {
  const listRef = useRef<HTMLDivElement>(null)

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const rtl = getComputedStyle(event.currentTarget).direction === 'rtl'
    const step = STEP[(rtl && MIRRORED[event.key]) || event.key]
    const current = tabs.findIndex(tab => tab.id === value)

    if (!step || current < 0) {
      return
    }

    event.preventDefault()
    const next = step(current, tabs.length)
    onChange(tabs[next].id)
    listRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
  }

  return (
    <div className="portal-page-tabs" data-tour="page-tabs">
      <div aria-label={label} className="portal-page-tabs-list" onKeyDown={onKeyDown} ref={listRef} role="tablist">
        {tabs.map(tab => {
          const selected = tab.id === value

          return (
            <button
              aria-selected={selected}
              data-tour={`tab-${tab.id}`}
              key={tab.id}
              onClick={event => {
                // On a phone the row scrolls sideways: a tab tapped at its edge comes fully into view.
                event.currentTarget.scrollIntoView({ block: 'nearest', inline: 'nearest' })
                onChange(tab.id)
              }}
              role="tab"
              tabIndex={selected ? 0 : -1}
              type="button"
            >
              <span>{tab.label}</span>
              {tab.meta !== undefined && <span className="portal-page-tab-count">{tabMetaContent(tab.meta)}</span>}
            </button>
          )
        })}
      </div>
    </div>
  )
}
