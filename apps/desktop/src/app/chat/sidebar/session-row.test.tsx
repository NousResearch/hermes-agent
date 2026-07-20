import type * as React from 'react'

import { atom } from 'nanostores'
import { cleanup, render } from '@testing-library/react'
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
// PlatformAvatar is intentionally NOT mocked here (unlike before): it forwards
// ref/props for real now (#67500), so this file exercises the actual
// production component to verify its Tip wiring, instead of a stand-in that
// spreads props the real component didn't.
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
// (archive/pin/delete wiring, plus a dozen unrelated stores) that are
// irrelevant here — this file only exercises the Tip fix, so pass their
// children straight through. The Tip-wraps-DropdownMenuTrigger composition
// itself (and that the menu still opens) is covered directly against the
// real component in session-actions-menu.test.tsx (#67500) — that test would
// have caught the original regression; this passthrough mock intentionally
// does NOT re-verify it, to avoid masking a future regression the way the
// old inline <Tip> mock here once did.
const sessionActionsMenuCalls: Array<{ tooltip?: React.ReactNode }> = []

vi.mock('./session-actions-menu', () => ({
  SessionActionsMenu: (props: { children: React.ReactNode; tooltip?: React.ReactNode }) => {
    sessionActionsMenuCalls.push({ tooltip: props.tooltip })

    return <>{props.children}</>
  },
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
  it('passes the actions-kebab tooltip label through to SessionActionsMenu', () => {
    sessionActionsMenuCalls.length = 0

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

    // The Tip itself now lives INSIDE SessionActionsMenu (composed around
    // DropdownMenuTrigger, not around the children it receives) — see
    // session-actions-menu.test.tsx for proof the menu still opens with it
    // there. This just confirms session-row hands over the right label.
    expect(sessionActionsMenuCalls[0]?.tooltip).toBe('Actions for Hermes doctor health check results')
  })

  it('does not render a handoff avatar for a locally-started session', () => {
    const { container } = render(
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

    // PlatformAvatar's span is the only aria-hidden SPAN this row ever
    // renders (idle dot / arc-border / branch-stem are all inactive here) —
    // Codicon icons (e.g. the kebab trigger) are also aria-hidden but render
    // as <i>, not <span>, so this selector doesn't accidentally match them.
    expect(container.querySelector('span[aria-hidden="true"]')).toBeNull()
  })

  it('wraps the handoff platform avatar in a Tip for a session started on another platform', () => {
    const { container } = render(
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

    // PlatformAvatar is now the REAL component (see the note above the
    // vi.mock block), which renders the Telegram brand SVG rather than the
    // platform name as text — so query the avatar span itself (it's the row's
    // only aria-hidden element in this state) rather than text content, and
    // confirm its tooltip trigger actually attaches to it, not to a mock that
    // faked the wiring (#67500).
    const avatar = container.querySelector('span[aria-hidden="true"]')
    expect(avatar).toBeTruthy()
    expect(tipTrigger(avatar as HTMLElement)).toBeTruthy()
  })
})
