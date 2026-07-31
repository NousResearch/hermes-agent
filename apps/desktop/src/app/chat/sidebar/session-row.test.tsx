import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { openSession } from '@/app/open-session'
import type { SessionInfo } from '@/hermes'
import { LOCALE_OPTIONS, TRANSLATIONS } from '@/i18n'
import type * as I18n from '@/i18n'
import type * as ComposerStatusStore from '@/store/composer-status'
import type * as SessionStore from '@/store/session'
import type * as SessionStatesStore from '@/store/session-states'
import type * as WindowsStore from '@/store/windows'

import { SidebarSessionRow } from './session-row'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

vi.mock('@/i18n', async importOriginal => {
  const actual = await importOriginal<typeof I18n>()

  return {
    ...actual,
    useI18n: () => ({
      t: {
        sidebar: {
          row: {
            ageMin: 'm',
            ageNow: 'now',
            backgroundRunning: 'Running in background',
            finishedUnread: 'Finished',
            handoffOrigin: (platform: string) => `Started on ${platform}`,
            needsInput: 'Needs input',
            sessionActions: 'Session actions',
            sessionRunning: 'Running',
            waitingForAnswer: 'Waiting for answer'
          }
        }
      }
    })
  }
})

vi.mock('@/app/chat/profile-tag', () => ({ ProfileTag: () => null }))
vi.mock('@/app/chat/session-drag', () => ({ startSessionDrag: vi.fn() }))
vi.mock('@/app/open-session', () => ({ openSession: vi.fn() }))
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

const noop = vi.fn()

type GestureAction = 'archive' | 'pin' | 'resume' | 'tab' | 'window'

const GESTURE_CASES: readonly {
  action: GestureAction
  event: 'click' | 'middle-click'
  label: string
  modifiers?: MouseEventInit
}[] = [
  { action: 'resume', event: 'click', label: 'ordinary click' },
  { action: 'tab', event: 'click', label: 'exact Ctrl-click', modifiers: { ctrlKey: true } },
  { action: 'tab', event: 'click', label: 'exact Cmd-click', modifiers: { metaKey: true } },
  { action: 'pin', event: 'click', label: 'exact Shift-click', modifiers: { shiftKey: true } },
  {
    action: 'window',
    event: 'click',
    label: 'exact Cmd+Shift-click',
    modifiers: { metaKey: true, shiftKey: true }
  },
  {
    action: 'archive',
    event: 'click',
    label: 'exact Ctrl+Shift-click',
    modifiers: { ctrlKey: true, shiftKey: true }
  },
  {
    action: 'window',
    event: 'click',
    label: 'Ctrl+Shift+Alt-click',
    modifiers: { altKey: true, ctrlKey: true, shiftKey: true }
  },
  {
    action: 'window',
    event: 'click',
    label: 'Ctrl+Shift+Meta-click',
    modifiers: { ctrlKey: true, metaKey: true, shiftKey: true }
  },
  { action: 'tab', event: 'middle-click', label: 'middle click', modifiers: { button: 1 } }
]

describe('SidebarSessionRow', () => {
  it('keeps an aria-label on the kebab without wrapping it in a Tip', () => {
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

    const kebab = screen.getByRole('button', { name: 'Session actions' })
    expect(tipTrigger(kebab)).toBeNull()
  })

  it.each(GESTURE_CASES)('$label dispatches only $action', ({ action, event, modifiers = {} }) => {
    const onArchive = vi.fn()
    const onDelete = vi.fn()
    const onPin = vi.fn()
    const onResume = vi.fn()

    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={onArchive}
        onDelete={onDelete}
        onPin={onPin}
        onResume={onResume}
        session={makeSession({ title: 'Gesture target' })}
      />
    )

    const row = screen.getByRole('button', { name: 'Gesture target' })

    if (event === 'middle-click') {
      fireEvent.pointerDown(row, modifiers)
      fireEvent.pointerUp(row, modifiers)
    } else {
      fireEvent.click(row, modifiers)
    }

    const callbackActions: Record<Exclude<GestureAction, 'tab' | 'window'>, ReturnType<typeof vi.fn>> = {
      archive: onArchive,
      pin: onPin,
      resume: onResume
    }

    for (const [candidate, callback] of Object.entries(callbackActions)) {
      expect(callback).toHaveBeenCalledTimes(action === candidate ? 1 : 0)
    }

    expect(onDelete).not.toHaveBeenCalled()

    if (action === 'tab' || action === 'window') {
      expect(openSession).toHaveBeenCalledOnce()
      expect(openSession).toHaveBeenCalledWith('s1', expect.any(Function), action)
    } else {
      expect(openSession).not.toHaveBeenCalled()
    }
  })

  it('keeps archived-session settings copy gesture-neutral in every registered locale', () => {
    const requiredLocales = LOCALE_OPTIONS.map(locale => locale.id)

    expect(Object.keys(TRANSLATIONS).sort()).toEqual([...requiredLocales].sort())

    for (const locale of requiredLocales) {
      expect(TRANSLATIONS[locale].settings.sessions.archivedIntro, locale).not.toMatch(
        /\b(?:ctrl|cmd|shift|click)\b|[⌘⌃⌥]|点击|點擊|クリック|اضغط|النقر/iu
      )
    }
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
