// Bug #2: the Branch-in-new-chat button used to render unconditionally even
// when its handler was a no-op (session-tile.tsx passed `() => undefined`
// for branched/tiled chats, where nested branching isn't supported). That
// left a visibly clickable button that silently did nothing. The fix makes
// AssistantMessage's action bar hide the button entirely when no handler is
// supplied, matching how onDismissError/onRestoreToMessage already behave.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TRANSLATIONS } from '@/i18n'
import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { formatMessageTimestamp, formatTimelineDuration } from './timestamp'

import { Thread } from '.'

const requestFreshSession = vi.hoisted(() => vi.fn())
const startManualProviderOAuth = vi.hoisted(() => vi.fn())

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestFreshSession: () => requestFreshSession()
}))

vi.mock('@/store/onboarding', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  startManualProviderOAuth: (...args: unknown[]) => startManualProviderOAuth(...args)
}))

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const completedAt = createdAt.getTime() / 1000 + 1.25
stubThreadEnvironment()

afterEach(() => {
  cleanup()
  requestFreshSession.mockClear()
  startManualProviderOAuth.mockClear()
})

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'question one' }],
    attachments: [],
    createdAt,
    metadata: { custom: { timelineTimestamp: createdAt.getTime() / 1000 } }
  } as unknown as ThreadMessage
}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [
      {
        type: 'reasoning',
        text: 'checked carefully',
        timestamp: createdAt.getTime() / 1000 + 0.05,
        completedAt: createdAt.getTime() / 1000 + 0.1
      },
      {
        type: 'text',
        text: 'done',
        timestamp: createdAt.getTime() / 1000 + 0.125,
        completedAt: createdAt.getTime() / 1000 + 0.5
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: { timelineCompletedAt: completedAt, timelineTimestamp: createdAt.getTime() / 1000 }
    }
  } as unknown as ThreadMessage
}

function ownershipRefusalMessage(): ThreadMessage {
  return {
    id: 'assistant-error-1',
    role: 'assistant',
    content: [],
    status: {
      type: 'incomplete',
      reason: 'error',
      error:
        'Session 20260909_095312_6b93f5 already has a live owner (tui, pid 32977, lease age 22m). ' +
        'Attach through a compatible owner, or close the session in its owning surface before resuming here.'
    },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      // What submit.ts stamps on a 4090 / SESSION_NOT_OWNED refusal.
      custom: { errorSurface: { layer: 'gateway', code: 'SESSION_NOT_OWNED', retryable: false } }
    }
  } as unknown as ThreadMessage
}

function oauthExpiredMessage(): ThreadMessage {
  return {
    id: 'assistant-error-2',
    role: 'assistant',
    content: [],
    status: { type: 'incomplete', reason: 'error', error: 'HTTP 401: User not found.' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      // What agent/error_surface.py stamps on a rejected OAuth grant.
      custom: {
        errorSurface: {
          authKind: 'oauth',
          code: 'auth',
          layer: 'auth',
          provider: 'nous',
          providerLabel: 'Nous Portal',
          retryable: false
        }
      }
    }
  } as unknown as ThreadMessage
}

function Harness({
  assistant = assistantMessage(),
  onBranchInNewChat
}: {
  assistant?: ThreadMessage
  onBranchInNewChat?: (messageId: string) => void
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistant],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread onBranchInNewChat={onBranchInNewChat} />
    </AssistantRuntimeProvider>
  )
}

describe('AssistantMessage branch button visibility (bug #2 fix)', () => {
  it('shows the Branch in new chat button when a handler is provided (open chat)', async () => {
    render(<Harness onBranchInNewChat={() => undefined} />)

    expect(await screen.findByRole('button', { name: 'Branch in new chat' })).toBeTruthy()
  })

  it('hides the Branch in new chat button when no handler is provided (session-tile / branched chat)', async () => {
    render(<Harness />)

    // Wait for the assistant message to actually mount before asserting
    // absence, so a missing button isn't just a false negative from an
    // unrendered message.
    await screen.findByText('done')

    expect(screen.queryByRole('button', { name: 'Branch in new chat' })).toBeNull()
  })
})

describe('ownership refusal recovery (#106217)', () => {
  it('offers Start new session and suppresses Retry for live-owner refusals', async () => {
    render(<Harness assistant={ownershipRefusalMessage()} />)

    expect(await screen.findByRole('button', { name: 'Start new session' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()

    screen.getByRole('button', { name: 'Start new session' }).click()
    expect(requestFreshSession).toHaveBeenCalledTimes(1)
  })
})

describe('expired OAuth grant recovery', () => {
  it('explains the expiry and re-runs that provider sign-in in one click', async () => {
    render(<Harness assistant={oauthExpiredMessage()} />)

    expect(await screen.findByText(/Nous Portal sign-in has expired/)).toBeTruthy()
    // Signing in changes the outcome, so Retry stays as the follow-up click.
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()

    screen.getByRole('button', { name: 'Sign in to Nous Portal again' }).click()
    expect(startManualProviderOAuth).toHaveBeenCalledWith('nous', undefined)
  })
})

describe('message timeline timestamps', () => {
  const MILLISECOND_CLOCK = /\d{1,2}:\d{2}:\d{2}\.\d{3}/
  const friendlyMessageTime = formatMessageTimestamp(createdAt, TRANSLATIONS.en.assistant.thread)

  const stampsIn = (container: HTMLElement) =>
    Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node => node.textContent?.trim())

  it('renders friendly message times and durations, never millisecond ranges', async () => {
    const { container } = render(<Harness />)

    await screen.findByText('done')

    const stamps = stampsIn(container)
    const startedAt = createdAt.getTime() / 1000

    // Message rows read as a day + clock, not a wall clock with milliseconds.
    expect(stamps).toContain(friendlyMessageTime)
    // Settled reasoning / text parts read as their duration. Both fixture steps
    // are sub-second, so neither may print a start → end range.
    expect(stamps).toContain(formatTimelineDuration(startedAt + 0.05, startedAt + 0.1))
    expect(stamps).toContain(formatTimelineDuration(startedAt + 0.125, startedAt + 0.5))
    // Nothing in the default view prints a millisecond wall clock.
    expect(stamps.filter(stamp => MILLISECOND_CLOCK.test(stamp ?? ''))).toEqual([])
  })

  it('suppresses an aggregate assistant stamp that exactly duplicates its sole part', async () => {
    const startedAt = createdAt.getTime() / 1000

    const assistant = {
      ...assistantMessage(),
      content: [{ completedAt, text: 'done', timestamp: startedAt, type: 'text' }]
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)

    await screen.findByText('done')

    const stamps = stampsIn(container)

    // The sole part's stamp replaces the aggregate — one event, one line, so
    // the friendly message time survives only on the user's row.
    expect(stamps.filter(stamp => stamp === friendlyMessageTime)).toHaveLength(1)
    expect(stamps).toContain(formatTimelineDuration(startedAt, completedAt))
  })
})
