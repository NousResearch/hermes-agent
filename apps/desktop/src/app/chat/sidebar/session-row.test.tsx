import { atom } from 'nanostores'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SidebarSessionRow } from './session-row'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        row: {
          actionsFor: (title: string) => `Actions for ${title}`,
          ageMin: 'm',
          ageNow: 'now',
          backgroundRunning: 'Running in background',
          finishedUnread: 'Finished',
          handoffOrigin: (platform: string) => `Started on ${platform}`,
          needsInput: 'Needs input',
          sessionRunning: 'Running',
          waitingForAnswer: 'Waiting for answer'
        }
      }
    }
  })
}))

vi.mock('@/app/chat/profile-tag', () => ({ ProfileTag: () => null }))
vi.mock('@/app/chat/session-drag', () => ({ startSessionDrag: vi.fn() }))
vi.mock('@/app/messaging/platform-icon', () => ({
  PlatformAvatar: ({ platformName, ...rest }: { platformName: string } & Record<string, unknown>) => (
    <span {...rest}>{platformName}</span>
  )
}))
vi.mock('@/lib/chat-runtime', () => ({ sessionTitle: (s: SessionInfo) => (s as unknown as { title: string }).title }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/session-source', () => ({
  handoffOriginSource: (state?: string, platform?: string) => (state && platform ? platform : null),
  sessionSourceLabel: (source: string) => source
}))
vi.mock('@/lib/time', () => ({ coarseElapsed: () => ({ unit: 'minute' as const, value: 5 }) }))

vi.mock('@/store/composer-status', () => ({ $backgroundRunningSessionIds: atom<string[]>([]) }))
vi.mock('@/store/session', () => ({
  $sessions: atom<Record<string, unknown>>({}),
  $unreadFinishedSessionIds: atom<string[]>([])
}))
// session-row.tsx reads the idle-dot project color from this derived store
// directly — mock it here rather than letting it recompute from $sessions/
// $projects, since this file only exercises the Tip wrapping, not coloring.
vi.mock('@/store/session-color', () => ({ $sessionColorById: atom<Record<string, string>>({}) }))
vi.mock('@/store/session-states', () => ({
  $attentionSessionIds: atom<string[]>([]),
  openSessionTile: vi.fn()
}))
vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => false,
  openSessionInNewWindow: vi.fn()
}))

// SessionActionsMenu/SessionContextMenu carry their own menu-item deps
// (archive/pin/delete wiring) that are irrelevant here — this file only
// exercises the Tip fix, so pass their children straight through.
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

const noop = vi.fn()

describe('SidebarSessionRow', () => {
  it('wraps the actions kebab in a Tip with the session title', () => {
    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        session={makeSession({ title: 'Hermes doctor health check results' })}
      />
    )

    const button = screen.getByRole('button', { name: 'Actions for Hermes doctor health check results' })
    expect(tipTrigger(button)).toBeTruthy()
  })

  it('does not render a handoff avatar for a locally-started session', () => {
    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        session={makeSession({ title: 'Local session' })}
      />
    )

    expect(screen.queryByText('telegram')).toBeNull()
  })

  it('wraps the handoff platform avatar in a Tip for a session started on another platform', () => {
    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        session={makeSession({
          handoff_platform: 'telegram',
          handoff_state: 'active',
          title: 'Continued from Telegram'
        })}
      />
    )

    // PlatformAvatar is stubbed to render its platformName as text, and
    // sessionSourceLabel is mocked as an identity function, so the visible
    // text is the raw platform id.
    const avatar = screen.getByText('telegram')
    expect(tipTrigger(avatar)).toBeTruthy()
  })
})
