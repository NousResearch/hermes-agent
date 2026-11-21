import { act, cleanup, render, screen } from '@testing-library/react'
import { atom, type WritableAtom } from 'nanostores'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import type * as ChatRuntime from '@/lib/chat-runtime'
import type * as ComposerStatusStore from '@/store/composer-status'
import type * as SessionStore from '@/store/session'
import { $unreadFinishedSessionIds } from '@/store/session'
import { $attentionSessionIds } from '@/store/session-states'
import type * as SessionStatesStore from '@/store/session-states'
import type * as WindowsStore from '@/store/windows'

import { SidebarSessionRow } from './session-row'

// The mock swaps this for a writable atom; the real one is computed.
const attention = $attentionSessionIds as unknown as WritableAtom<string[]>

// Same harness as session-row.test.tsx: real stores, with the input atoms the
// dot state reads replaced by plain atoms this file drives.
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        messageCount: (count: number) => `${count} messages`,
        toolCallCount: (count: number) => `${count} tool calls`,
        projects: { home: 'Home' },
        row: {
          ageMin: 'm',
          ageNow: 'now',
          backgroundRunning: 'Running in background',
          finishedUnread: 'Finished',
          handoffOrigin: (platform: string) => `Started on ${platform}`,
          messageCount: (count: number) => `${count} messages`,
          needsInput: 'Needs input',
          needsYouBadge: 'Needs you',
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
vi.mock('@/lib/chat-runtime', async importOriginal => {
  const actual = await importOriginal<typeof ChatRuntime>()

  return { ...actual, sessionTitle: (s: SessionInfo) => (s as unknown as { title: string }).title }
})
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
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

  return { ...actual, canOpenSessionWindow: () => false, openSessionInNewWindow: vi.fn() }
})
vi.mock('./session-actions-menu', () => ({
  SessionActionsMenu: (props: { children?: React.ReactNode }) => <>{props.children}</>,
  SessionContextMenu: (props: { children?: React.ReactNode }) => <>{props.children}</>
}))
vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), notePointerMove: vi.fn(), startPrewarm: vi.fn() })
}))

const session = {
  handoff_platform: null,
  handoff_state: null,
  id: 's1',
  input_tokens: 0,
  last_active: 0,
  output_tokens: 0,
  profile: 'default',
  started_at: 0,
  title: 'Deploy plan'
} as unknown as SessionInfo

const noop = vi.fn()

const row = (props: { card?: boolean; unread?: boolean } = {}) => (
  <SidebarSessionRow
    card={props.card}
    isPinned={false}
    isSelected={false}
    onArchive={noop}
    onDelete={noop}
    onPin={noop}
    onResume={noop}
    onToggleUnread={noop}
    session={session}
    unread={props.unread ?? false}
  />
)

const titleLabel = () => screen.getByText('Deploy plan').parentElement as HTMLElement

afterEach(() => {
  cleanup()
  attention.set([])
  $unreadFinishedSessionIds.set([])
})

describe.each([
  ['one-line row', false],
  ['card row', true]
])('needs-you cue (%s)', (_name, card) => {
  it('shows the "Needs you" pill and the amber edge while the session waits on the owner', () => {
    attention.set(['s1'])
    const { container } = render(row({ card }))

    expect(screen.getByText('Needs you')).toBeTruthy()
    const edge = container.querySelector('[data-needs-you-edge]')
    expect(edge).not.toBeNull()
    expect(edge?.className).toContain('bg-amber-500')
    // The dot stays: the cue sits beside it, not instead of it.
    expect(screen.getByRole('status', { name: 'Needs input' })).toBeTruthy()
  })

  it('draws neither for a session nobody is waiting on, and clears when the prompt is answered', () => {
    attention.set(['s1'])
    const { container } = render(row({ card }))

    act(() => attention.set([]))

    expect(screen.queryByText('Needs you')).toBeNull()
    expect(container.querySelector('[data-needs-you-edge]')).toBeNull()
  })
})

describe.each([
  ['one-line row', false],
  ['card row', true]
])('unread title weight (%s)', (_name, card) => {
  it('is semibold and bright when the row is unread', () => {
    render(row({ card, unread: true }))

    expect(titleLabel().className).toContain('font-semibold')
    expect(titleLabel().className).toContain('text-foreground')
  })

  it('is semibold when a turn finished in the background (runtime unread)', () => {
    $unreadFinishedSessionIds.set(['s1'])
    render(row({ card }))

    expect(titleLabel().className).toContain('font-semibold')
  })

  it('is normal weight once read, through the memoized row', () => {
    const { rerender } = render(row({ card, unread: true }))

    rerender(row({ card, unread: false }))

    expect(titleLabel().className).toContain('font-normal')
    expect(titleLabel().className).not.toContain('font-semibold')
  })
})
