import './global-overviews.css'

import { type ComponentProps, type ReactNode, type Ref, useLayoutEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { cn } from '@/lib/utils'
import { usePortalShell } from '@/store/shell-variant'

// Layout blocks for the redesigned settings pages (Figma 856:*): a page stack, grouped-row cards and
// collapsible "Advanced"-style disclosures. Classic renders the pre-redesign wrapper each call site
// already had (a `contents` pass-through by default), so classic layout stays byte-identical.

interface SettingsOverviewProps {
  children: ReactNode
  className?: string
}

/** The page's vertical rhythm: 36px between sections. */
export function SettingsOverview({ children, className }: SettingsOverviewProps) {
  const redesign = usePortalShell()

  return <div className={redesign ? cn('global-settings-overview', className) : 'contents'}>{children}</div>
}

interface SettingsGroupProps {
  children: ReactNode
  title: ReactNode
  /** One line under the heading (895:17703 "Messages sent here reach this agent."). */
  description?: ReactNode
  /** Right-aligned note on the heading row ("Saved to this browser", a count). */
  meta?: ReactNode
  id?: string
}

/** A titled section (856:3409): 15px Medium heading, 12px to its cards. Classic shows no heading. */
export function SettingsGroup({ children, description, id, meta, title }: SettingsGroupProps) {
  const redesign = usePortalShell()

  if (!redesign) {
    return <>{children}</>
  }

  return (
    <section className="global-settings-section" id={id}>
      <div className="global-settings-section-head">
        <h3>{title}</h3>
        {meta && <span>{meta}</span>}
        {description && <p className="global-settings-section-description">{description}</p>}
      </div>
      {children}
    </section>
  )
}

/** The page's one-line description under its H1 (856:3405). The heading belongs to the route frame,
 *  so the page renders only the line; classic pages never had one. */
export function SettingsPageDescription({ children }: { children: ReactNode }) {
  const redesign = usePortalShell()

  return redesign ? (
    <p className="mb-9" data-slot="page-description">
      {children}
    </p>
  ) : null
}

interface SettingsAddRowProps extends Omit<ComponentProps<'button'>, 'className' | 'type'> {
  children: ReactNode
  /** Codicon name; a list card's own add action keeps the default plus. */
  icon?: string
}

/** The last row of a list card (856:5142 "+ Add connection"): a full-width quiet button. */
export function SettingsAddRow({ children, icon = 'add', ...props }: SettingsAddRowProps) {
  return (
    <button className="global-settings-add-row" type="button" {...props}>
      <Codicon aria-hidden name={icon} />
      {children}
    </button>
  )
}

interface SettingsCardProps extends Omit<ComponentProps<'div'>, 'className'> {
  className?: string
  /** Classic wrapper class; `contents` keeps the rows flowing into the parent, `''` is a plain div. */
  classicClassName?: string
}

/** Grouped rows on one r8 card with hairline dividers between them. */
export function SettingsCard({ className, classicClassName = 'contents', ...props }: SettingsCardProps) {
  const redesign = usePortalShell()

  return <div className={redesign ? cn('global-settings-card', className) : classicClassName || undefined} {...props} />
}

/** A list card with nothing in it yet. The redesign uses the one empty block (877:3596) as the card's
 *  flush row; classic keeps the quiet line it always showed. */
export function SettingsEmptyRow({ title }: { title: string }) {
  return usePortalShell() ? <EmptyState title={title} /> : <p className="global-settings-empty-row">{title}</p>
}

interface SettingsLinkRowProps {
  title: ReactNode
  description?: ReactNode
  /** Quiet trailing note before the chevron (a source, a count). */
  meta?: ReactNode
  onClick: () => void
}

/** A list card row that opens something (a blueprint, a sub-page): label, help line, chevron. */
export function SettingsLinkRow({ description, meta, onClick, title }: SettingsLinkRowProps) {
  return (
    <button className="global-settings-link-row" onClick={onClick} type="button">
      <span className="global-settings-link-row-text">
        <span className="global-settings-link-row-title">{title}</span>
        {description && <span className="global-settings-link-row-description">{description}</span>}
      </span>
      {meta && <span className="global-settings-link-row-meta">{meta}</span>}
      <Codicon aria-hidden name="chevron-right" />
    </button>
  )
}

interface SettingsFooterProps {
  children: ReactNode
}

/** The page's closing hairline and right-aligned quiet action (856:3477 "Reset appearance to defaults"). */
export function SettingsFooter({ children }: SettingsFooterProps) {
  return <div className="global-settings-footer">{children}</div>
}

interface SettingsDisclosureProps {
  children: ReactNode
  title: ReactNode
  description?: ReactNode
  /** Badge beside the caret. Omitted, it counts the settings rendered inside; pass it when the
   *  body lists items rather than settings (archived chats, the full shortcut list). */
  count?: number
  /** What classic renders above the children instead of a summary row. */
  classicHeading?: ReactNode
  className?: string
  /** Starts expanded (a deep link points inside it). */
  open?: boolean
  /** The user opened or closed it (a route that owns the expanded state follows along). */
  onOpenChange?: (open: boolean) => void
  ref?: Ref<HTMLDetailsElement>
}

/** Every setting control row carries it: ListRow (and so ToggleRow), credential key cards. */
const SETTINGS_ROW = '[data-setting]'

/** Top-level settings under `body`: a row nested in another row's `below` is part of that one. */
function countSettingsRows(body: HTMLElement) {
  return [...body.querySelectorAll(SETTINGS_ROW)].filter(
    row => !body.contains(row.parentElement?.closest(SETTINGS_ROW) ?? null)
  ).length
}

/** Live count of the settings a disclosure body renders; rows mount late (config loads), so it follows the DOM. */
function useRenderedSettingsCount(enabled: boolean) {
  const body = useRef<HTMLDivElement>(null)
  const [count, setCount] = useState(0)

  useLayoutEffect(() => {
    const element = body.current

    if (!enabled || !element) {
      return
    }

    const update = () => setCount(countSettingsRows(element))
    const observer = new MutationObserver(update)
    update()
    observer.observe(element, { childList: true, subtree: true })

    return () => observer.disconnect()
  }, [enabled])

  return { body, count }
}

/** A collapsed card (856:3467): title, optional description and count, chevron that flips on open. */
export function SettingsDisclosure({
  children,
  classicHeading,
  className,
  count,
  description,
  open,
  onOpenChange,
  ref,
  title
}: SettingsDisclosureProps) {
  const redesign = usePortalShell()
  const rendered = useRenderedSettingsCount(redesign && count === undefined)
  const badge = count ?? (rendered.count || undefined)

  if (!redesign) {
    return (
      <div>
        {classicHeading}
        {children}
      </div>
    )
  }

  return (
    <details
      className={cn('global-settings-disclosure group', className)}
      onToggle={onOpenChange && (event => onOpenChange(event.currentTarget.open))}
      open={open || undefined}
      ref={ref}
    >
      <summary>
        <span className="global-settings-disclosure-text">
          <span className="global-settings-disclosure-title">{title}</span>
          {description && <span className="global-settings-disclosure-description">{description}</span>}
        </span>
        {badge !== undefined && <span className="global-settings-disclosure-count">{badge}</span>}
        <Codicon aria-hidden className="global-settings-disclosure-caret" name="chevron-down" />
      </summary>
      <div className="global-settings-disclosure-body" ref={rendered.body}>
        {children}
      </div>
    </details>
  )
}
