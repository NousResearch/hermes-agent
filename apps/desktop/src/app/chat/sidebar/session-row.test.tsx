import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import type * as ComposerStatusStore from '@/store/composer-status'
import type * as SessionStore from '@/store/session'
import type * as SessionStatesStore from '@/store/session-states'
import type * as WindowsStore from '@/store/windows'

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
// PlatformAvatar is intentionally NOT mocked (do not reintroduce this — see
// #67500, Gille's third pass): it's a forwardRef component that spreads its
// props onto the rendered span, and mocking it with a stand-in that spreads
// props itself only proves the MOCK forwards them, not that the real
// component does. This file exercises the actual production component so a
// regression in its ref/prop forwarding fails here again.
vi.mock('@/lib/chat-runtime', () => ({ sessionTitle: (s: SessionInfo) => (s as unknown as { title: string }).title }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/session-source', () => ({
  handoffOriginSource: (state?: string, platform?: string) => (state && platform ? platform : null),
  sessionSourceLabel: (source: string) => source
}))
vi.mock('@/lib/time', () => ({ coarseElapsed: () => ({ unit: 'minute' as const, value: 5 }) }))

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

// SessionActionsMenu owns the Tip-around-DropdownMenuTrigger composition
// itself now (see session-actions-menu.test.tsx, which exercises that real,
// unmocked end-to-end) — testing it again here via the mock would just
// duplicate that coverage and silently stop testing anything the moment the
// mock's shape drifts from the real component's props (as happened when
// `tooltip` was introduced). This file only needs to confirm session-row
// wires the right tooltip text into the `tooltip` prop, so the mock renders
// it in a way we can assert on directly instead of re-deriving Tip's
// internal DOM structure.
vi.mock('./session-actions-menu', () => ({
  SessionActionsMenu: ({ children, tooltip }: { children: React.ReactNode; tooltip?: string }) => (
    <div data-testid="session-actions-menu" data-tooltip={tooltip}>
      {children}
    </div>
  ),
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
  it('wires the actions kebab tooltip text through to SessionActionsMenu', () => {
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

    expect(screen.getByTestId('session-actions-menu').getAttribute('data-tooltip')).toBe(
      'Actions for Hermes doctor health check results'
    )
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

    // PlatformAvatar is the REAL component here (see the note above the vi.mock
    // block, #67500 third pass) — it renders the Telegram brand SVG rather
    // than the platform name as text, so query the avatar span itself (the
    // row's only aria-hidden span in this state) rather than text content,
    // and confirm its tooltip trigger actually attaches to it — proving the
    // real forwardRef/...rest path works, not a mock that fakes it.
    const avatar = container.querySelector('span[aria-hidden="true"]')
    expect(avatar).toBeTruthy()
    expect(tipTrigger(avatar as HTMLElement)).toBeTruthy()
  })
})
