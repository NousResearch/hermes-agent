import { type ComponentType, type ReactElement, useMemo } from 'react'

import { ContribBoundary } from './react/boundary'
import { useContributions } from './react/use-contributions'
import type { Contribution } from './types'

/**
 * Kanban card-drawer sections — the extension point for the card drawer's main
 * column, so a plugin can add a panel (a live output feed, a trace, a cost
 * meter) to the module a card opens, without touching the drawer's markup.
 *
 *   data kind (`data`):  kanban.card.drawer  (KanbanCardDrawerSectionContribution)
 *
 * A plugin supplies a React component; the host mounts it in the drawer's main
 * column, in the plugin's registration order, each inside its own error
 * boundary so a broken section degrades to a small inline error instead of
 * taking the drawer down. The component receives the drawer's read-only task
 * context and owns everything it draws.
 */
export const KANBAN_CARD_DRAWER_AREAS = { sections: 'kanban.card.drawer' } as const

/** The task context a drawer section renders from. Frozen; every field is a
 *  standing compatibility promise. Pick the newer key when both are present:
 *  `board` is null when the drawer did not resolve a board slug. */
export interface KanbanCardDrawerSectionProps {
  /** Board slug the card belongs to, or null when unresolved. Feed it to your
   *  own `ctx.rest` calls as the `board` query param. */
  readonly board: null | string
  /** True while the card's status is `running` — the cue to poll for updates. */
  readonly running: boolean
  /** The card's column (`triage`, `ready`, `running`, `done`, …). */
  readonly status: string
  /** The task id (`t_…`) the drawer is showing. */
  readonly taskId: string
}

/** Payload of a `kanban.card.drawer` data contribution. `Section` is mounted as
 *  a component (its hooks and errors belong to it), not called as a function. */
export interface KanbanCardDrawerSectionContribution {
  Section: ComponentType<KanbanCardDrawerSectionProps>
}

interface ResolvedSection {
  id: string
  key: string
  Section: ComponentType<KanbanCardDrawerSectionProps>
}

function resolveSections(contributions: readonly Contribution[]): ResolvedSection[] {
  const sections: ResolvedSection[] = []

  for (const contribution of contributions) {
    const Section = (contribution.data as Partial<KanbanCardDrawerSectionContribution> | undefined)?.Section

    if (typeof Section === 'function') {
      sections.push({ id: contribution.id, key: `${contribution.source ?? 'core'}:${contribution.id}`, Section })
    }
  }

  return sections
}

/** The drawer's main-column section host. Renders nothing when no plugin
 *  contributes a section, so an unused area costs no space. */
export function KanbanCardDrawerSections({
  board,
  running,
  status,
  taskId
}: KanbanCardDrawerSectionProps): null | ReactElement {
  const contributions = useContributions(KANBAN_CARD_DRAWER_AREAS.sections)
  const sections = useMemo(() => resolveSections(contributions), [contributions])

  if (sections.length === 0) {
    return null
  }

  return (
    <div className="flex flex-col gap-5" data-slot="kanban-card-drawer-sections">
      {sections.map(({ id, key, Section }) => (
        <ContribBoundary id={id} key={key} variant="pane">
          <Section board={board} running={running} status={status} taskId={taskId} />
        </ContribBoundary>
      ))}
    </div>
  )
}
