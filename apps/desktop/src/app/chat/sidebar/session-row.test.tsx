import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import type * as ChatRuntime from '@/lib/chat-runtime'
import type * as Time from '@/lib/time'
import { $backgroundRunningSessionIds } from '@/store/composer-status'
import type * as ComposerStatusStore from '@/store/composer-status'
import type * as SessionStore from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type * as SessionStatesStore from '@/store/session-states'
import type * as WindowsStore from '@/store/windows'

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
// build session state with the same factory the app uses. It is a spy because
// the row calls it exactly once per render, which is how the isolation test
// below counts repaints.
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
// row chrome (handoff avatar tip, etc.).
vi.mock('./session-actions-menu', () => ({
  SessionActionsMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SessionContextMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
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

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

// The status dot always paints an aria-hidden placeholder so every row's title
// keeps the same left edge, so "the row's aria-hidden span" no longer names the
// avatar on its own. `inline-grid` is PlatformAvatar's own layout class in both
// of its branches — brand glyph and first-letter fallback — and the row passes
// it no display class that tailwind-merge could drop it for.
const handoffAvatar = (container: HTMLElement) =>
  container.querySelector<HTMLElement>('span[aria-hidden="true"].inline-grid')

const noop = vi.fn()

// The production export is computed/read-only; this file replaces it with an
// atom above so individual row tests can drive the exact resolved state.
const writableBackgroundRunningSessionIds = $backgroundRunningSessionIds as unknown as {
  set: (ids: string[]) => void
}

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
    act(() => {
      writableBackgroundRunningSessionIds.set([])
      clearAllSessionStates()
    })
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

  it('keeps the active title treatment when background work outlives the turn', () => {
    act(() => {
      writableBackgroundRunningSessionIds.set(['s1'])
    })

    const { container } = renderRow(makeSession({ title: 'Background work' }))
    const row = container.querySelector<HTMLElement>('.row-hover')

    expect(row?.dataset.working).toBe('true')
    expect(arc(container)).toBeNull()
  })

  // The row owns its status subscription so a turn starting repaints that row
  // and nothing else — not its siblings, and not the list around them. Rows
  // render once per fiber, so counting `sessionTitle` counts repaints.
  it('repaints only the session whose turn started', () => {
    render(
      <>
        {[makeSession({ id: 's1', title: 'One' }), makeSession({ id: 's2', title: 'Two' })].map(session => (
          <SidebarSessionRow
            isPinned={false}
            isSelected={false}
            key={session.id}
            onArchive={noop}
            onDelete={noop}
            onPin={noop}
            onResume={noop}
            onToggleUnread={noop}
            session={session}
            unread={false}
          />
        ))}
      </>
    )
    sessionTitle.mockClear()

    act(() => {
      publishSessionState('rt1', { ...createClientSessionState('s1'), busy: true })
    })

    expect(sessionTitle).toHaveBeenCalledTimes(1)
    expect(sessionTitle).toHaveBeenCalledWith(expect.objectContaining({ id: 's1' }))
  })
})

describe('SidebarSessionRow', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('keeps an aria-label on the kebab without wrapping it in a Tip', () => {
    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        onToggleUnread={noop}
        session={makeSession({ title: 'Hermes doctor health check results' })}
        unread={false}
      />
    )

    const kebab = screen.getByRole('button', { name: 'Session actions' })
    expect(tipTrigger(kebab)).toBeNull()
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

    it('wraps the title in a tooltip trigger', () => {
      renderRow(makeSession({ title }))

      expect(label()).toBeTruthy()
    })

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

  it('exposes the exact session time through a focusable Tip trigger', () => {
    // Pin the clock before deriving the timestamp.  The assertion below is
    // about the *composition* of the label (relative age + absolute time),
    // but "5 minutes ago" only falls on today when the run does not straddle
    // local midnight.  Between 00:00 and 00:05 the row correctly renders
    // "Yesterday at 11:5x PM" and this test failed for a day boundary it was
    // never written to exercise.  Only `Date` is faked, so the component's
    // own timers (the running arc, the tooltip open delay) keep running for
    // real.
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(new Date(2026, 2, 5, 12, 0, 0))

    const startedAt = Math.floor(Date.now() / 1000) - 5 * 60

    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        onToggleUnread={noop}
        session={makeSession({ started_at: startedAt, title: 'Timestamped session' })}
        unread={false}
      />
    )

    const age = screen.getByText('5m')
    expect(age.tagName).toBe('TIME')
    expect(age.getAttribute('datetime')).toBe(new Date(startedAt * 1000).toISOString())
    expect(age.getAttribute('aria-label')).toMatch(/^5m, Today at /)
    expect(age.getAttribute('tabindex')).toBe('0')
    expect(age.getAttribute('title')).toBeNull()
    expect(tipTrigger(age)).toBeTruthy()
  })

  it('does not render a handoff avatar for a locally-started session', () => {
    const { container } = render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        onToggleUnread={noop}
        session={makeSession({ title: 'Local session' })}
        unread={false}
      />
    )

    expect(handoffAvatar(container)).toBeNull()
  })

  it('wraps the handoff platform avatar in a Tip for a session started on another platform', () => {
    const { container } = render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        onToggleUnread={noop}
        session={makeSession({
          handoff_platform: 'telegram',
          handoff_state: 'active',
          title: 'Continued from Telegram'
        })}
        unread={false}
      />
    )

    // PlatformAvatar is the REAL component here (see the note above the vi.mock
    // block, #67500 third pass) — it renders the Telegram brand SVG rather
    // than the platform name as text, so query the avatar span itself rather
    // than text content, and confirm its tooltip trigger actually attaches to
    // it — proving the real forwardRef/...rest path works, not a mock that
    // fakes it.
    const avatar = handoffAvatar(container)
    expect(avatar).toBeTruthy()
    expect(tipTrigger(avatar as HTMLElement)).toBeTruthy()
  })
})

describe('Inbox-style session card', () => {
  it('gives truncated card lines room for glyph ink instead of clipping them', () => {
    renderRow(
      makeSession({
        cwd: '/Users/tomek/pursuit-support-agent',
        message_count: 133,
        model: 'gpt-4.1',
        title: 'Ruff lint and pytest verification'
      }),
      { card: true }
    )

    const workspace = screen.getByText('pursuit-support-agent')
    const title = screen.getByText('Ruff lint and pytest verification').parentElement
    const footer = screen.getByText('GPT-4.1').parentElement

    expect(title).toBeTruthy()
    expect(footer).toBeTruthy()

    for (const el of [workspace, title!, footer!]) {
      expect(el.className).not.toMatch(/\bleading-none\b/)
      expect(el.className).toMatch(/leading-\[1\.35\]/)
    }

    expect(workspace.className).toMatch(/\btruncate\b/)
    expect(screen.getByText('133 messages')).toBeTruthy()
  })
})
