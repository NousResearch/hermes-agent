/**
 * Coverage for the tooltip sweep on the task-detail drawer (t_da65e35c):
 * close, task-actions ellipsis, upload-attachment, reassign trigger,
 * dependency chips, and diagnostics recovery-action buttons all get an
 * explanatory hover tooltip via the same `<Tip>` mechanism the card already
 * used. Renders the REAL TaskDrawer against a mocked fetchTask.
 */

import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { translateKanban } from './i18n-test-helper'
import type { KanbanTaskDetail } from './types'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<Record<string, unknown>>()

  return { ...sdk, usePluginI18n: () => translateKanban }
})

const baseDetail: KanbanTaskDetail = {
  attachments: [],
  comments: [],
  events: [],
  links: { children: ['t_childone12'], parents: ['t_parentone1'] },
  runs: [],
  task: {
    assignee: 'alice',
    diagnostics: [
      {
        actions: [
          { kind: 'cli_hint', label: 'Copy command', payload: { command: 'hermes kanban reclaim t_x' } },
          { kind: 'reclaim', label: 'Reclaim' }
        ],
        count: 1,
        data: {},
        detail: 'Worker crashed.',
        kind: 'crash',
        last_seen_at: 0,
        severity: 'error',
        title: 'Worker crashed'
      }
    ],
    id: 't_drawertest1',
    status: 'blocked',
    title: 'Drawer test task'
  }
}

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchLog: vi.fn(async () => ({ content: '', exists: false, size_bytes: 0, truncated: false })),
  fetchProfiles: vi.fn(async () => ({ profiles: [{ description: '', description_auto: false, is_default: false, name: 'bob' }] })),
  fetchTask: vi.fn(async () => baseDetail)
}))

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

async function hoverTip(trigger: Element) {
  vi.useFakeTimers()
  act(() => {
    fireEvent.pointerMove(trigger, { pointerType: 'mouse' })
    vi.advanceTimersByTime(300)
  })
}

async function renderDrawer() {
  const { TaskDrawer } = await import('./drawer')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <TaskDrawer columns={['blocked', 'done']} id="t_drawertest1" onClose={vi.fn()} onOpen={vi.fn()} />
    </QueryClientProvider>
  )

  await screen.findByText('Drawer test task')
}

describe('drawer tooltips', () => {
  it('shows a tooltip on the close button', async () => {
    await renderDrawer()

    await hoverTip(screen.getByRole('button', { name: 'Close' }))

    expect(screen.getByRole('tooltip').textContent).toContain('Close')
  })

  it('shows a tooltip on the task-actions ellipsis trigger', async () => {
    await renderDrawer()

    await hoverTip(screen.getByRole('button', { name: 'Task actions' }))

    expect(screen.getByRole('tooltip').textContent).toContain('Task actions')
  })

  it('shows a tooltip on the upload-attachment button', async () => {
    await renderDrawer()

    await hoverTip(screen.getByRole('button', { name: 'Upload attachment' }))

    expect(screen.getByRole('tooltip').textContent).toContain('Upload attachment')
  })

  it('shows a tooltip on the assignee reassign trigger', async () => {
    await renderDrawer()

    const trigger = screen.getByText('alice').closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    expect(screen.getByRole('tooltip').textContent).toContain('reassign')
  })

  it('has NO native [title] descendant on the Tip-wrapped reassign trigger (no double tooltip)', async () => {
    // The Major from round 2: the reassign button is wrapped in <Tip>, but its
    // inner <Avatar> was still emitting title={name}, stacking a native OS
    // tooltip on top of the Tip. Assert the Tip trigger subtree carries no
    // [title] so the Tip is the single tooltip source.
    await renderDrawer()

    const trigger = screen.getByText('alice').closest('[data-slot="tooltip-trigger"]') as HTMLElement
    expect(trigger).toBeTruthy()
    expect(trigger.querySelector('[title]')).toBeNull()
    // The trigger element itself must not carry a native title either.
    expect(trigger.hasAttribute('title')).toBe(false)
  })

  it('keeps the reassign button accessible name to the assignee EXACTLY ONCE (no avatar stutter)', async () => {
    // Round-3 m1: the round-2 aria-label fallback on the reassign trigger's
    // Avatar made the button announce the name twice — accname computes the
    // button's name from content, and a child aria-label REPLACES that child's
    // text, so the avatar's "A" initials became "alice" next to the visible
    // <span>alice</span> ("alice alice"). The fix marks the avatar decorative
    // (aria-hidden), so the name comes only from the visible span — exactly once.
    await renderDrawer()

    const trigger = screen.getByText('alice').closest('button') as HTMLElement
    expect(trigger).toBeTruthy()

    // The avatar is hidden from the a11y tree and carries neither title nor
    // aria-label; the name lives solely in the adjacent visible text.
    const avatar = screen.getByText('A').closest('span') as HTMLElement
    expect(avatar.getAttribute('aria-hidden')).toBe('true')
    expect(avatar.hasAttribute('title')).toBe(false)
    expect(avatar.hasAttribute('aria-label')).toBe(false)

    // Accessible name (Testing Library computes it via the same accname algo):
    // querying the button by the SINGLE name resolves it, while the doubled
    // "alice alice" (the round-2 regression) does not exist.
    expect(screen.getByRole('button', { name: 'alice' })).toBe(trigger)
    expect(screen.queryByRole('button', { name: 'alice alice' })).toBeNull()
  })

  it('keeps the OPTED-IN reassign menu-item avatar carrying native title= (opt-in path coverage)', async () => {
    // Round-3 m3: the opted-in Avatar path (title={true}) had zero coverage —
    // mutations N4 (always emit aria-label even when opted in) and N9 (drop the
    // opt-in at the menu item) both survived. The menu items are dropdown
    // choices with NO adjacent avatar-labelled control, so they DO want the
    // native title. Open the menu and assert the item avatar still emits
    // title='bob' AND does NOT emit an aria-label (which would double-announce
    // against the item's own visible name text).
    await renderDrawer()

    const trigger = screen.getByText('alice').closest('button') as HTMLElement
    act(() => {
      fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      fireEvent.click(trigger)
    })

    // The roster mock exposes one profile, 'bob'; its menu-item avatar renders
    // the "B" initials with the opted-in native title.
    const itemAvatar = await screen.findByText('B')
    const itemAvatarSpan = itemAvatar.closest('span') as HTMLElement
    expect(itemAvatarSpan.getAttribute('title')).toBe('bob')
    // Opted-in path must NOT also emit aria-label (that is the suppressed-path
    // behaviour; emitting both is mutation N4).
    expect(itemAvatarSpan.hasAttribute('aria-label')).toBe(false)
    expect(itemAvatarSpan.hasAttribute('aria-hidden')).toBe(false)
  })

  it('shows distinct tooltips on parent vs child dependency chips', async () => {
    await renderDrawer()

    const parentChip = screen.getByText('parent').closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(parentChip)
    expect(screen.getByRole('tooltip').textContent).toContain('still blocking this card')

    const childChip = screen.getByText('childo').closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(childChip)
    expect(screen.getByRole('tooltip').textContent).toContain('waiting on this card')
  })

  it('shows tooltips on diagnostics recovery-action buttons', async () => {
    await renderDrawer()

    const copyBtn = screen.getByText('Copy command').closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(copyBtn)
    expect(screen.getByRole('tooltip').textContent).toContain('clipboard')

    const reclaimBtn = screen.getByText('Reclaim').closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(reclaimBtn)
    expect(screen.getByRole('tooltip').textContent).toContain('Reclaim this task')
  })

  it('shows a tooltip on the edit-description button', async () => {
    await renderDrawer()

    await hoverTip(screen.getByRole('button', { name: 'Edit description' }))

    expect(screen.getByRole('tooltip').textContent).toContain('Edit description')
  })

  it('shows a tooltip on the drawer short task id revealing the full id', async () => {
    await renderDrawer()

    await hoverTip(screen.getByText('drawer').closest('[data-slot="tooltip-trigger"]')!)

    expect(screen.getByRole('tooltip').textContent).toContain('t_drawertest1')
  })
})
