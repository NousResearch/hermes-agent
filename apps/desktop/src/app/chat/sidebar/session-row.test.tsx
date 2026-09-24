import { KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import type { SessionInfo } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import type * as ChatRuntime from '@/lib/chat-runtime'
import { SESSION_ROW_AREAS, type SessionRowSlotProps } from '@/lib/session-row-slots'
import type * as Time from '@/lib/time'
import type * as ComposerStatusStore from '@/store/composer-status'
import type * as SessionStore from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type * as SessionStatesStore from '@/store/session-states'
import type * as WindowsStore from '@/store/windows'

import { ReorderableList, useSortableBindings } from './reorderable-list'
import { SidebarSessionRow } from './session-row'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        messageCount: (count: number) => `${count} messages`,
        toolCallCount: (count: number) => `${count} tool calls`,
        projects: {
          home: 'Home'
        },
        row: {
          ageMin: 'm',
          ageNow: 'now',
          backgroundRunning: 'Running in background',
          finishedUnread: 'Finished',
          handoffOrigin: (platform: string) => `Started on ${platform}`,
          messageCount: (count: number) => `${count} messages`,
          needsInput: 'Needs input',
          sessionActions: 'Session actions',
          sessionRunning: 'Running',
          todoProgress: 'Tasks completed',
          waitingForAnswer: 'Waiting for answer'
        }
      },
      assistant: {
        thread: {
          today: (time: string) => `Today at ${time}`,
          yesterday: (time: string) => `Yesterday at ${time}`
        }
      }
    }
  })
}))

vi.mock('@/app/chat/profile-tag', () => ({ ProfileTag: () => null }))
vi.mock('@/app/chat/session-drag', () => ({ startSessionDrag: vi.fn() }))
// PlatformAvatar is intentionally NOT mocked (do not reintroduce this — see
// #67500, Gille's third pass): it's a forwardRef component that spreads its
// props onto the rendered span, and mocking it with a stand-in that spreads
// props itself only proves the MOCK forwards them, not that the real
// component does. This file exercises the actual production component so a
// regression in its ref/prop forwarding fails here again.
// Only `sessionTitle` is overridden (makeSession fakes a bare `title` the real
// one wouldn't read); the rest of the module is genuine so the arc test can
// build session state with the same factory the app uses.
const sessionTitle = vi.fn((s: SessionInfo) => (s as unknown as { title: string }).title)

vi.mock('@/lib/chat-runtime', async importOriginal => {
  const actual = await importOriginal<typeof ChatRuntime>()

  return { ...actual, sessionTitle: (s: SessionInfo) => sessionTitle(s) }
})
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/session-source', () => ({
  handoffOriginSource: (state?: string, platform?: string) => (state && platform ? platform : null),
  sessionSourceLabel: (source: string) => source
}))
vi.mock('@/lib/time', async importOriginal => {
  const actual = await importOriginal<typeof Time>()

  return { ...actual, coarseElapsed: () => ({ unit: 'minute' as const, value: 5 }) }
})

// These mocks use importOriginal rather than replacing the module wholesale:
// session-row.tsx (and its transitive imports, e.g. session-color.ts) reads
// several store exports beyond the ones this file cares about, and that set
// keeps growing as the app evolves upstream. A wholesale replacement mock
// silently turns every export it doesn't list into `undefined`, which then
// crashes nanostores' `computed()` the moment a new dependency is added
// upstream (as happened twice already: $stalledSessionIds, then $sessions).
// Overriding only the named atoms we actually control keeps this test
// resilient to that drift.
vi.mock('@/store/composer-status', async importOriginal => {
  const actual = await importOriginal<typeof ComposerStatusStore>()

  return { ...actual, $backgroundRunningSessionIds: atom<string[]>([]) }
})
vi.mock('@/store/session', async importOriginal => {
  const actual = await importOriginal<typeof SessionStore>()

  return { ...actual, $unreadFinishedSessionIds: atom<string[]>([]) }
})
vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStatesStore>()

  return {
    ...actual,
    $attentionSessionIds: atom<string[]>([]),
    $stalledSessionIds: atom<string[]>([]),
    openSessionTile: vi.fn()
  }
})
vi.mock('@/store/windows', async importOriginal => {
  const actual = await importOriginal<typeof WindowsStore>()

  return {
    ...actual,
    canOpenSessionWindow: () => false,
    openSessionInNewWindow: vi.fn()
  }
})

// SessionActionsMenu open behavior is covered in session-actions-menu.test.tsx
// against the real component. Stub it here so this file stays focused on the
// row chrome.
vi.mock('./session-actions-menu', () => ({
  SessionActionsMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SessionContextMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), notePointerMove: vi.fn(), startPrewarm: vi.fn() })
}))

function makeSession(overrides: Partial<SessionInfo> & { title: string }): SessionInfo {
  return {
    handoff_platform: null,
    handoff_state: null,
    id: 's1',
    last_active: 0,
    profile: 'default',
    started_at: 0,
    ...overrides
  } as unknown as SessionInfo
}

const noop = vi.fn()

const renderRow = (session: SessionInfo, extra?: { card?: boolean }) =>
  render(
    <SidebarSessionRow
      card={extra?.card}
      isPinned={false}
      isSelected={false}
      onArchive={noop}
      onDelete={noop}
      onPin={noop}
      onResume={noop}
      onToggleUnread={noop}
      session={session}
      unread={false}
    />
  )

// The row no longer takes its running state as a prop, so this drives the real
// store the way the app does. $workingSessionIds is the actual computed here
// (the mock above only overrides its siblings), which is what makes this cover
// the wiring rather than the predicate — the arc has gone missing before.
describe('SidebarSessionRow running arc', () => {
  afterEach(() => {
    clearAllSessionStates()
  })

  const arc = (container: HTMLElement) => container.querySelector('.arc-row')

  it('paints no arc for a settled session', () => {
    const { container } = renderRow(makeSession({ title: 'Settled' }))

    expect(arc(container)).toBeNull()
  })

  it('paints the arc while the session is running', () => {
    publishSessionState('rt1', { ...createClientSessionState('s1'), busy: true })

    const { container } = renderRow(makeSession({ title: 'Running' }))

    expect(arc(container)).toBeTruthy()
  })
})

describe('SidebarSessionRow', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  // Full-title tooltip on hover (#83000-class ask): the label is a tooltip
  // trigger, but the tip only opens when the title is actually truncated.
  describe('full-title overflow tooltip', () => {
    afterEach(() => {
      vi.useRealTimers()
    })

    const title = 'A very long session title that the sidebar cannot possibly fit'

    /** The rendered title label (tooltip trigger is the label itself). */
    const label = () => screen.getByText(title).closest('[data-slot="tooltip-trigger"]') as HTMLElement

    const setWidths = (el: HTMLElement, scrollWidth: number, clientWidth: number) => {
      Object.defineProperty(el, 'scrollWidth', { configurable: true, value: scrollWidth })
      Object.defineProperty(el, 'clientWidth', { configurable: true, value: clientWidth })
    }

    it('opens with the full title after a settled hover when the title overflows', () => {
      vi.useFakeTimers()
      renderRow(makeSession({ title }))

      const el = label()
      setWidths(el, 300, 100)

      act(() => {
        fireEvent.pointerEnter(el)
        vi.advanceTimersByTime(700)
      })

      expect(screen.getByRole('tooltip').textContent).toContain(title)
    })

    it('stays closed when the title fits', () => {
      vi.useFakeTimers()
      renderRow(makeSession({ title }))

      const el = label()
      setWidths(el, 100, 100)

      act(() => {
        fireEvent.pointerEnter(el)
        vi.advanceTimersByTime(700)
      })

      expect(screen.queryByRole('tooltip')).toBeNull()
    })

    it('cancels a pending open when the pointer leaves before the delay', () => {
      vi.useFakeTimers()
      renderRow(makeSession({ title }))

      const el = label()
      setWidths(el, 300, 100)

      act(() => {
        fireEvent.pointerEnter(el)
        vi.advanceTimersByTime(200)
        fireEvent.pointerLeave(el)
        vi.advanceTimersByTime(700)
      })

      expect(screen.queryByRole('tooltip')).toBeNull()
    })
  })
})

// Regression for #83617: the row shell once spread the FULL dnd-kit handle, so
// Space on a focused control inside the row (the ⋯ button that opens Rename)
// reached the KeyboardSensor's activator — a drag armed, and the sensor then
// ate the next Space at window level (the rename input dropped the keystroke).
describe('SidebarSessionRow inside the sortable list', () => {
  function SortableRow({ session }: { session: SessionInfo }) {
    const { dragHandleProps, dragging, ref, reorderable, style } = useSortableBindings(session.id)

    return (
      <SidebarSessionRow
        dragging={dragging}
        dragHandleProps={dragHandleProps}
        isPinned={false}
        isSelected={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        onToggleUnread={noop}
        ref={ref}
        reorderable={reorderable}
        session={session}
        style={style}
        unread={false}
      />
    )
  }

  function Host({ session }: { session: SessionInfo }) {
    // The sidebar's own sensor set (index.tsx dndSensors).
    const sensors = useSensors(
      useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
      useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
    )

    return (
      <ReorderableList ids={[session.id]} onReorder={noop} sensors={sensors}>
        <SortableRow session={session} />
      </ReorderableList>
    )
  }

  const space = { code: 'Space', key: ' ' }

  it('lets Space through to a focused row control instead of arming a keyboard drag', () => {
    const { container } = render(<Host session={makeSession({ title: 'Renamable' })} />)
    const kebab = screen.getByRole('button', { name: 'Session actions' })
    kebab.focus()

    // Not defaultPrevented (the ⋯ menu is stubbed in this file, so only
    // dnd-kit could have claimed the key) and no grabber reports a drag.
    expect(fireEvent.keyDown(kebab, space)).toBe(true)
    expect(container.querySelector('[aria-pressed="true"]')).toBeNull()
  })

  it('still starts a keyboard reorder from the grabber', () => {
    const { container } = render(<Host session={makeSession({ title: 'Renamable' })} />)
    const grabber = container.querySelector<HTMLElement>('[data-reorder-handle]')!

    grabber.focus()
    fireEvent.keyDown(grabber, space)
    expect(grabber.getAttribute('aria-pressed')).toBe('true')
  })
})

// Row-decoration slots: a plugin decorates rows through the registry with the
// row's stored session id handed to its render — the seam the session-list API
// pairs with (see #116305 item 3).
describe('SidebarSessionRow decoration slots', () => {
  const disposers: Array<() => void> = []

  afterEach(() => {
    disposers.splice(0).forEach(dispose => dispose())
  })

  const decorate = (area: string, id: string, testId: string) =>
    disposers.push(
      registry.register({
        area,
        data: {
          render: ({ sessionId }: SessionRowSlotProps) => <span data-testid={testId}>{sessionId}</span>
        },
        id,
        source: 'disk'
      })
    )

  it('mounts leading and trailing decorations, each handed the DURABLE row id', () => {
    act(() => {
      decorate(SESSION_ROW_AREAS.leading, 'deco-lead', 'lead-deco')
      decorate(SESSION_ROW_AREAS.trailing, 'deco-tail', 'tail-deco')
    })

    // Auto-compression rotates the live id. A plugin that remembered the live
    // one decorates this row until the next compaction and then silently stops
    // matching — so the slot hands the lineage root, the id core's own
    // pin/reorder and `host.sessions.*` address.
    renderRow(makeSession({ _lineage_root_id: 'root-9', id: 'live-9', title: 'Compressed' }))

    expect(screen.getByTestId('lead-deco').textContent).toBe('root-9')
    expect(screen.getByTestId('tail-deco').textContent).toBe('root-9')
  })

  it('renders nothing for an area with no registrations and survives an unmount', () => {
    const { container } = renderRow(makeSession({ id: 'row-7', title: 'Plain' }))

    expect(container.querySelector('[data-testid="lead-deco"]')).toBeNull()

    act(() => {
      decorate(SESSION_ROW_AREAS.leading, 'deco-lead', 'lead-deco')
    })

    // Same row, contribution arriving late: the slot mounts it in place.
    expect(screen.getByTestId('lead-deco').textContent).toBe('row-7')

    act(() => {
      disposers.splice(0).forEach(dispose => dispose())
    })

    expect(screen.queryByTestId('lead-deco')).toBeNull()
  })
})
