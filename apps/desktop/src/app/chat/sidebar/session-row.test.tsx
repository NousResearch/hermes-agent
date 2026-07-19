import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $projectTree } from '@/store/projects'

import { SidebarSessionRow } from './session-row'

// The row pulls a lot of live wiring; stub the seams that need a gateway.
vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
}))

vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => false,
  // layout.ts reads this at import time (window-scoped store init).
  isSecondaryWindow: () => false,
  openSessionInNewWindow: vi.fn()
}))

const session = (over: Partial<SessionInfo>): SessionInfo =>
  ({
    cwd: '/Users/s/work/api',
    ended_at: null,
    id: 'ses_1',
    input_tokens: 0,
    is_active: true,
    last_active: 0,
    message_count: 1,
    output_tokens: 0,
    started_at: 0,
    title: 'A session',
    ...over
  }) as SessionInfo

const noop = () => undefined

function renderRow(over: Partial<SessionInfo> = {}, isWorking = false) {
  return render(
    <SidebarSessionRow
      isPinned={false}
      isSelected={false}
      isWorking={isWorking}
      onArchive={noop}
      onDelete={noop}
      onPin={noop}
      onResume={noop}
      session={session(over)}
    />
  )
}

/** The lead dot: the row's first status span (idle dots carry no role). */
const leadDot = (container: HTMLElement) =>
  container.querySelector('[data-row-actions]')?.parentElement?.querySelector('span.rounded-full') ??
  container.querySelector('span.rounded-full')

describe('SidebarSessionRow project color', () => {
  beforeEach(() => {
    $projectTree.set([
      {
        archived: false,
        color: '#4a9eff',
        icon: null,
        id: 'work',
        label: 'work',
        path: '/Users/s/work',
        repos: [],
        sessionCount: 1
      }
    ] as never)
  })

  afterEach(() => {
    cleanup()
    $projectTree.set([])
  })

  it('tints the idle dot with the owning project color', () => {
    const { container } = renderRow()
    const dot = leadDot(container)

    expect(dot).toBeTruthy()
    expect((dot as HTMLElement).style.backgroundColor).toBe('rgb(74, 158, 255)')
  })

  it('leaves sessions outside any colored project untinted', () => {
    const { container } = renderRow({ cwd: '/Users/s/elsewhere' })
    const dot = leadDot(container)

    expect(dot).toBeTruthy()
    expect((dot as HTMLElement).style.backgroundColor).toBe('')
  })

  it('never tints a working session — status color wins', () => {
    const { container } = renderRow({}, true)
    const running = screen.getByRole('status')

    expect(running.style.backgroundColor).toBe('')
    // And the working dot keeps its accent class.
    expect(running.className).toContain('bg-(--ui-accent)')
    expect(container.querySelector('[style*="background-color"]')).toBeNull()
  })
})
