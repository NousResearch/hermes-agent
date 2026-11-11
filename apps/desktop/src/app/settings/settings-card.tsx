import './global-overviews.css'

import type { ComponentProps, ReactNode, Ref } from 'react'

import { Codicon } from '@/components/ui/codicon'
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

interface SettingsDisclosureProps {
  children: ReactNode
  title: ReactNode
  description?: ReactNode
  /** Number of settings inside, shown as a badge beside the caret. */
  count?: number
  /** What classic renders above the children instead of a summary row. */
  classicHeading?: ReactNode
  className?: string
  /** Starts expanded (a deep link points inside it). */
  open?: boolean
  ref?: Ref<HTMLDetailsElement>
}

/** A collapsed card (856:3467): title, optional description and count, chevron that flips on open. */
export function SettingsDisclosure({
  children,
  classicHeading,
  className,
  count,
  description,
  open,
  ref,
  title
}: SettingsDisclosureProps) {
  const redesign = usePortalShell()

  if (!redesign) {
    return (
      <div>
        {classicHeading}
        {children}
      </div>
    )
  }

  return (
    <details className={cn('global-settings-disclosure group', className)} open={open || undefined} ref={ref}>
      <summary>
        <span className="global-settings-disclosure-text">
          <span className="global-settings-disclosure-title">{title}</span>
          {description && <span className="global-settings-disclosure-description">{description}</span>}
        </span>
        {count !== undefined && <span className="global-settings-disclosure-count">{count}</span>}
        <Codicon aria-hidden className="global-settings-disclosure-caret" name="chevron-down" />
      </summary>
      <div className="global-settings-disclosure-body">{children}</div>
    </details>
  )
}
