import { act, cleanup, render } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $sidebarRowMeta } from '@/store/layout'
import { MINUTE_CLOCK_INTERVAL_MS } from '@/store/minute-clock'

import { SidebarSessionRow } from './session-row'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        messageCount: (count: number) => `${count} messages`,
        toolCallCount: (count: number) => `${count} tool calls`,
        projects: { home: 'Home' },
        row: {
          ageDay: 'd',
          ageHour: 'h',
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
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

vi.mock('./session-actions-menu', () => ({
  SessionActionsMenu: (props: { children?: React.ReactNode }) => <>{props.children}</>,
  SessionContextMenu: (props: { children?: React.ReactNode }) => <>{props.children}</>
}))

vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), notePointerMove: vi.fn(), startPrewarm: vi.fn() })
}))

const noop = vi.fn()
const T0 = Date.UTC(2026, 10, 5, 16, 0, 0)

function makeSession(lastActiveMs: number): SessionInfo {
  return {
    handoff_platform: null,
    handoff_state: null,
    id: 'age-1',
    last_active: lastActiveMs / 1000,
    profile: 'default',
    started_at: lastActiveMs / 1000,
    title: 'Idle session'
  } as unknown as SessionInfo
}

const age = (container: HTMLElement) => container.querySelector('time, [aria-label*="Today"]')?.textContent?.trim()

describe('SidebarSessionRow age', () => {
  afterEach(() => {
    cleanup()
    // Let nanostores' 1 s unmount delay run on the fake clock, so the next test
    // starts with the clock's interval torn down, not orphaned on fake timers.
    vi.advanceTimersByTime(1000)
    vi.useRealTimers()
  })

  // The row is memoized and the session list only republishes changed rows,
  // so an idle row never re-renders on its own. Its age must still advance.
  it('keeps counting while the row is idle', () => {
    vi.useFakeTimers()
    vi.setSystemTime(T0)
    $sidebarRowMeta.set([...$sidebarRowMeta.get().filter(m => m !== 'updated'), 'updated'])

    const { container } = render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        onToggleUnread={noop}
        session={makeSession(T0)}
        unread={false}
      />
    )

    expect(age(container)).toBe('now')

    act(() => {
      vi.advanceTimersByTime(12 * 60_000 + MINUTE_CLOCK_INTERVAL_MS)
    })

    expect(age(container)).toBe('12m')

    act(() => {
      vi.advanceTimersByTime(2 * 60 * 60_000)
    })

    expect(age(container)).toBe('2h')
  })
})
