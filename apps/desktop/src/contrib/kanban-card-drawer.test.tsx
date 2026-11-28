/**
 * The `kanban.card.drawer` contract: a plugin adds a panel to the card drawer's main column
 * and core owns everything else — the host, the ordering and the blast wall around each
 * section. These drive the REAL plugin context and registry (as `kanban-card-badges.test.ts`
 * does for the badge row), so they test what a plugin can actually register and what the
 * host will actually accept.
 */
import {
  KANBAN_CARD_DRAWER_AREAS,
  type KanbanCardDrawerSectionContribution,
  type KanbanCardDrawerSectionProps
} from '@hermes/plugin-sdk'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { KanbanCardDrawerSections } from './kanban-card-drawer'
import { createPluginContext } from './plugin'
import { registry } from './registry'

const CONTEXT: KanbanCardDrawerSectionProps = {
  board: 'hermes-maintenance',
  running: true,
  status: 'running',
  taskId: 't_00c50f1c'
}

const loaded: Array<() => void> = []

/** Register a section the way the plugin loader does, collecting the disposers it will run
 *  on unload/disable. Returns that unload. */
function loadSection(
  pluginId: string,
  id: string,
  Section: KanbanCardDrawerSectionContribution['Section'],
  order?: number
) {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(pluginId, dispose => disposers.push(dispose))

  ctx.register({
    area: KANBAN_CARD_DRAWER_AREAS.sections,
    data: { Section } satisfies KanbanCardDrawerSectionContribution,
    id,
    order
  })

  const unload = () => disposers.splice(0).forEach(dispose => dispose())
  loaded.push(unload)

  return unload
}

const sectionText = (label: string) => () => <div>{label}</div>

const sectionsHost = () => document.querySelector('[data-slot="kanban-card-drawer-sections"]')

afterEach(() => {
  cleanup()
  loaded.splice(0).forEach(unload => unload())
  vi.restoreAllMocks()
})

describe('kanban.card.drawer', () => {
  it('renders nothing when no plugin contributes a section, so the area costs no space', () => {
    const { container } = render(<KanbanCardDrawerSections {...CONTEXT} />)

    expect(container.firstChild).toBeNull()
    expect(sectionsHost()).toBeNull()
  })

  it('mounts every registered section in `order`, lowest first', () => {
    loadSection('late', 'late', sectionText('late'), 100)
    loadSection('early', 'early', sectionText('early'), -5)

    render(<KanbanCardDrawerSections {...CONTEXT} />)

    expect(sectionsHost()?.textContent).toBe('earlylate')
  })

  it("hands each section the drawer's read-only task context", () => {
    const seen: Array<KanbanCardDrawerSectionProps> = []

    loadSection('spy', 'spy', props => {
      seen.push(props)

      return <div>spy</div>
    })

    render(<KanbanCardDrawerSections {...CONTEXT} />)

    expect(seen).toEqual([CONTEXT])
  })

  it('renders the newest context when the drawer moves to another card', () => {
    loadSection('spy', 'spy', props => <div>{`${props.taskId}:${props.board ?? 'no-board'}`}</div>)

    const { rerender } = render(<KanbanCardDrawerSections {...CONTEXT} />)
    expect(screen.getByText('t_00c50f1c:hermes-maintenance')).toBeDefined()

    rerender(
      <KanbanCardDrawerSections board={null} running={false} status="done" taskId="t_other" />
    )
    expect(screen.getByText('t_other:no-board')).toBeDefined()
  })

  it('isolates a section that throws: the drawer and the other sections keep working', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    loadSection('broken', 'broken', () => {
      throw new Error('boom')
    })
    loadSection('fine', 'fine', sectionText('still here'))

    render(<KanbanCardDrawerSections {...CONTEXT} />)

    // The throwing section degrades to its own inline error, named after the contribution,
    // and the healthy section beside it is untouched.
    expect(sectionsHost()?.textContent).toContain('broken')
    expect(sectionsHost()?.textContent).toContain('failed to render')
    expect(screen.getByText('still here')).toBeDefined()
  })

  it('ignores a contribution whose payload carries no Section component', () => {
    const disposers: Array<() => void> = []
    const ctx = createPluginContext('malformed', dispose => disposers.push(dispose))
    loaded.push(() => disposers.splice(0).forEach(dispose => dispose()))

    // A data contribution into this area that answers with the wrong shape.
    ctx.register({ area: KANBAN_CARD_DRAWER_AREAS.sections, data: { decorate: () => null }, id: 'nope' })
    loadSection('good', 'good', sectionText('good'))

    render(<KanbanCardDrawerSections {...CONTEXT} />)

    expect(sectionsHost()?.textContent).toBe('good')
  })

  it('removes a section when its plugin unloads', () => {
    const unload = loadSection('temp', 'temp', sectionText('temp'))

    render(<KanbanCardDrawerSections {...CONTEXT} />)
    expect(sectionsHost()?.textContent).toBe('temp')

    unload()
    cleanup()

    const { container } = render(<KanbanCardDrawerSections {...CONTEXT} />)
    expect(container.firstChild).toBeNull()
  })

  it('is registered on the area constant the SDK publishes', () => {
    expect(KANBAN_CARD_DRAWER_AREAS.sections).toBe('kanban.card.drawer')
    expect(registry.getArea(KANBAN_CARD_DRAWER_AREAS.sections)).toHaveLength(0)
  })
})
