import { type ReactNode, useId } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Skeleton } from '@/components/ui/skeleton'

// The per-agent capability pages (Figma 895:13886 Tools, 895:14967 Skills, 895:15892 Connectors &
// MCP, 895:16810 Plugins) share one frame: a library banner, then titled sections whose rows sit
// on one hairline card. These render only in the redesign's agent layout.

interface AgentCapabilityBannerProps {
  title: ReactNode
  description?: ReactNode
  /** The one action: a hand-off to the global library for this capability. */
  action: ReactNode
}

/** Banner (895:14660): brand-tinted card, sparkle tile, title + one line, action on the right. */
export function AgentCapabilityBanner({ title, description, action }: AgentCapabilityBannerProps) {
  return (
    <div className="agent-capability-banner">
      <span aria-hidden className="agent-capability-banner-icon">
        <Codicon name="sparkle" />
      </span>
      <div className="agent-capability-banner-text">
        <p>{title}</p>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  )
}

interface AgentCapabilitySectionProps {
  title: ReactNode
  description?: ReactNode
  children: ReactNode
  /** The card's last row: "Show all N" or a paging "Show more". */
  footer?: ReactNode
}

/** Section (895:14670): 15px title, 13px note, 12px to a card of 64px rows with hairline dividers. */
export function AgentCapabilitySection({ title, description, children, footer }: AgentCapabilitySectionProps) {
  return (
    <section className="agent-capability-section">
      <header>
        <h3>{title}</h3>
        {description && <p>{description}</p>}
      </header>
      <div className="agent-capability-rows">
        {children}
        {footer}
      </div>
    </section>
  )
}

/** Rows a loading section reserves: a section's usual preview. */
const SKELETON_ROWS = 4

/** A section while its list loads: the head's two lines and a card of 64px rows, so the rows land in
 *  place instead of replacing a spinner. */
export function AgentCapabilitySkeleton() {
  return (
    <section aria-busy="true" className="agent-capability-section" data-capability-skeleton>
      <header>
        <Skeleton className="agent-capability-skeleton-title" />
        <Skeleton className="agent-capability-skeleton-note" />
      </header>
      <div className="agent-capability-rows">
        {Array.from({ length: SKELETON_ROWS }, (_, index) => (
          <div className="agent-capability-row" key={index}>
            <div className="agent-capability-row-text">
              <Skeleton className="agent-capability-skeleton-name" />
              <Skeleton className="agent-capability-skeleton-note" />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

interface AgentCapabilityMoreProps {
  label: string
  onClick: () => void
}

/** "Show all 23" (895:14956): a card's closing row that reveals the rest of its list. */
export function AgentCapabilityMore({ label, onClick }: AgentCapabilityMoreProps) {
  return (
    <button className="agent-capability-show-all" onClick={onClick} type="button">
      {label}
    </button>
  )
}

interface AgentCapabilityRowProps {
  name: ReactNode
  description?: ReactNode
  /** Small tags under the name (a skill's category, a plugin's provenance). */
  tags?: readonly string[]
  /** Content below the description, inside the text column (expanded tool chips). */
  below?: ReactNode
  /** Opens the entry's details dialog; the name and description become one button. */
  onOpen?: () => void
  /** Catalog entry id: the details dialog hands focus back to this row's button on close. */
  entryId?: string
  controls: ReactNode
  expanded?: boolean
  id?: string
}

/** Row (895:14754): name 14 Medium, description 13 tertiary, controls (meta, switch, ⋯) at the end. */
export function AgentCapabilityRow({
  name,
  description,
  tags,
  below,
  onOpen,
  controls,
  expanded,
  entryId,
  id
}: AgentCapabilityRowProps) {
  const nameId = useId()
  const descriptionId = useId()

  const text = (
    <>
      <span className="agent-capability-row-name" id={nameId}>
        {name}
      </span>
      {tags && tags.length > 0 && (
        <span className="agent-capability-tags">
          {tags.map(tag => (
            <span key={tag}>{tag}</span>
          ))}
        </span>
      )}
      {description && (
        <span className="agent-capability-row-description" id={descriptionId}>
          {description}
        </span>
      )}
    </>
  )

  return (
    <div className="agent-capability-row" data-entry-id={entryId} data-expanded={expanded || undefined} id={id}>
      <div className="agent-capability-row-text">
        {onOpen ? (
          // Named by the entry alone; the tags and description describe it.
          <button
            aria-describedby={description ? descriptionId : undefined}
            aria-haspopup="dialog"
            aria-labelledby={nameId}
            className="agent-capability-entry"
            onClick={onOpen}
            type="button"
          >
            {text}
          </button>
        ) : (
          text
        )}
        {below}
      </div>
      <div className="agent-capability-row-controls">{controls}</div>
    </div>
  )
}
