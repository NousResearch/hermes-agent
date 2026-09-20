// Bug #2: the Branch-in-new-chat button used to render unconditionally even
// when its handler was a no-op (session-tile.tsx passed `() => undefined`
// for branched/tiled chats, where nested branching isn't supported). That
// left a visibly clickable button that silently did nothing. The fix makes
// AssistantMessage's action bar hide the button entirely when no handler is
// supplied, matching how onDismissError/onRestoreToMessage already behave.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { formatTimelineRange, formatTimelineTimestamp } from './timestamp'

import { Thread } from '.'

const requestFreshSession = vi.hoisted(() => vi.fn())
const startManualProviderOAuth = vi.hoisted(() => vi.fn())
const requestModelMenuToggle = vi.hoisted(() => vi.fn<() => boolean>(() => true))

vi.mock('@/app/chat/composer/focus', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelMenuToggle: () => requestModelMenuToggle()
}))

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
  requestModelMenuToggle.mockReset().mockReturnValue(true)
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

/** A failed turn carrying an arbitrary error_surface descriptor. */
function failedMessage(errorSurface: Record<string, unknown>, error = 'HTTP 400: raw provider body'): ThreadMessage {
  return {
    id: 'assistant-error-3',
    role: 'assistant',
    content: [],
    status: { type: 'incomplete', reason: 'error', error },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: { errorSurface }
    }
  } as unknown as ThreadMessage
}

/** Renders the router's current URL so a test can assert where a deep link went. */
function LocationProbe() {
  const location = useLocation()

  return <span data-testid="location">{`${location.pathname}${location.search}`}</span>
}

function Harness({
  assistant = assistantMessage(),
  onBranchInNewChat,
  onReload
}: {
  assistant?: ThreadMessage
  onBranchInNewChat?: (messageId: string) => void
  onReload?: () => Promise<void>
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistant],
    isRunning: false,
    onNew: async () => {},
    ...(onReload ? { onReload } : {})
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

  it('explains the refusal in plain words and demotes the lease text to details', async () => {
    render(<Harness assistant={ownershipRefusalMessage()} />)

    expect(await screen.findByText(/open in another Hermes window or terminal/)).toBeTruthy()
    // The raw refusal ("live owner", "pid", "lease") is kept only inside the
    // collapsed Details disclosure, never as the headline.
    const raw = screen.getByText(/already has a live owner/)
    expect(raw.closest('details')).not.toBeNull()
  })
})

describe('code-keyed error card copy and actions', () => {
  it('hides Retry and offers Edit message for a safety refusal', async () => {
    render(
      <Harness
        assistant={failedMessage({
          code: 'content_policy_blocked',
          layer: 'provider',
          provider: 'openai',
          retryable: false
        })}
      />
    )

    expect(await screen.findByText('The AI service declined this request')).toBeTruthy()

    // The user bubble itself is also labelled "Edit message"; assert on the
    // card's own action button.
    const editActions = screen
      .getAllByRole('button', { name: 'Edit message' })
      .filter(button => button.classList.contains('aui-error-action'))

    expect(editActions).toHaveLength(1)
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })

  it('offers Choose a model and no Retry when the model is not available', async () => {
    render(
      <Harness
        assistant={failedMessage({ code: 'model_not_found', layer: 'provider', provider: 'openai', retryable: false })}
      />
    )

    expect(await screen.findByRole('button', { name: 'Choose a model' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    // The raw HTTP body is not the lead sentence.
    expect(screen.getByText(en.assistant.thread.errorCodes.model_not_found.title as string)).toBeTruthy()
    expect(screen.getByText(/HTTP 400/).closest('details')).not.toBeNull()
  })

  it('offers Compress conversation and Start new session for a context overflow', async () => {
    render(<Harness assistant={failedMessage({ code: 'context_overflow', layer: 'provider', retryable: true })} />)

    expect(await screen.findByText('This conversation is too long')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Compress conversation' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Start new session' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })

  it('names the provider and keeps Retry for a rate limit', async () => {
    render(
      <Harness
        assistant={failedMessage(
          { code: 'rate_limit', layer: 'provider', provider: 'openai', retryable: true },
          'HTTP 429: {"error":{"message":"Rate limit reached"}}'
        )}
      />
    )

    expect(await screen.findByText('The AI service is busy')).toBeTruthy()
    expect(screen.getByText(/openai is limiting requests right now/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('falls back to the generic headline when no descriptor was sent (older backend)', async () => {
    const legacy = {
      ...failedMessage({}),
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as unknown as ThreadMessage

    render(<Harness assistant={legacy} />)

    expect(await screen.findByText("Hermes couldn't finish this reply")).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })
})

describe('scheduled retry at the usage-limit reset (#98852)', () => {
  const rateLimited = (resetsAt: number) =>
    failedMessage(
      { code: 'rate_limit', layer: 'provider', provider: 'openai', resetsAt, retryable: true },
      'HTTP 429: {"error":{"message":"Rate limit reached"}}'
    )

  afterEach(() => {
    vi.useRealTimers()
  })

  it('fires the same reload as Retry exactly once, at resets_at and not before', async () => {
    const onReload = vi.fn(async () => {})
    const resetsAt = Math.floor(Date.now() / 1000) + 600

    render(<Harness assistant={rateLimited(resetsAt)} onReload={onReload} />)

    const arm = await screen.findByRole('button', { name: /^Retry when the limit resets \(\d\d:\d\d\)$/ })

    vi.useFakeTimers()
    fireEvent.click(arm)

    expect(screen.getByTestId('error-retry-scheduled').textContent).toMatch(/Retrying at \d\d:\d\d — in \d+m \d\ds/)
    expect(screen.queryByRole('button', { name: /^Retry when the limit resets/ })).toBeNull()

    await act(async () => vi.advanceTimersByTime(resetsAt * 1000 - Date.now() - 1_000))
    expect(onReload).not.toHaveBeenCalled()

    await act(async () => vi.advanceTimersByTime(1_000))
    expect(onReload).toHaveBeenCalledTimes(1)

    await act(async () => vi.advanceTimersByTime(3_600_000))
    expect(onReload).toHaveBeenCalledTimes(1)
  })

  it('never fires after Cancel or unmount, and hides the button once the reset has passed', async () => {
    const onReload = vi.fn(async () => {})
    const resetsAt = Math.floor(Date.now() / 1000) + 600

    const view = render(<Harness assistant={rateLimited(resetsAt)} onReload={onReload} />)
    const armName = /^Retry when the limit resets/

    fireEvent.click(await screen.findByRole('button', { name: armName }))
    vi.useFakeTimers()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByTestId('error-retry-scheduled')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: armName }))
    view.unmount()

    await act(async () => vi.advanceTimersByTime(3_600_000))
    expect(onReload).not.toHaveBeenCalled()
    vi.useRealTimers()

    render(<Harness assistant={rateLimited(Math.floor(Date.now() / 1000) - 60)} onReload={onReload} />)
    expect(await screen.findByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: armName })).toBeNull()
  })
})

describe('rejected API key recovery', () => {
  it('names the key as the problem and deep-links Settings → Keys to that env var', async () => {
    render(
      <MemoryRouter>
        <LocationProbe />
        <Harness
          assistant={failedMessage(
            {
              apiKeyEnv: 'OPENAI_API_KEY',
              authKind: 'api_key',
              code: 'auth',
              layer: 'auth',
              provider: 'openai',
              providerLabel: 'OpenAI',
              retryable: false
            },
            'HTTP 401: {"error":{"message":"Incorrect API key provided: sk-…"}}'
          )}
        />
      </MemoryRouter>
    )

    expect(await screen.findByText('OpenAI rejected your API key')).toBeTruthy()
    // Fixing the key changes the outcome, so Retry stays as the follow-up click.
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()

    screen.getByRole('button', { name: 'Update API key' }).click()
    await waitFor(() => expect(screen.getByTestId('location').textContent).toMatch(/\?tab=keys&key=OPENAI_API_KEY$/))
  })
})

describe('switch provider on a live session (#95066)', () => {
  const billingFailure = () =>
    failedMessage({ code: 'billing', layer: 'billing', provider: 'openai-codex', retryable: false }, 'HTTP 429: quota')

  it('opens the live session model menu instead of leaving the chat for Settings', async () => {
    render(
      <MemoryRouter>
        <LocationProbe />
        <Harness assistant={billingFailure()} />
      </MemoryRouter>
    )

    const button = await screen.findByRole('button', { name: 'Switch provider' })
    const before = screen.getByTestId('location').textContent

    button.click()

    expect(requestModelMenuToggle).toHaveBeenCalledTimes(1)
    // Still on the chat: the pick lands on THIS session through model.switch.
    expect(screen.getByTestId('location').textContent).toBe(before)
  })

  it('falls back to Settings → Models only when no chat surface is on screen', async () => {
    requestModelMenuToggle.mockReturnValue(false)
    render(
      <MemoryRouter>
        <LocationProbe />
        <Harness assistant={billingFailure()} />
      </MemoryRouter>
    )

    ;(await screen.findByRole('button', { name: 'Switch provider' })).click()

    expect(requestModelMenuToggle).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(screen.getByTestId('location').textContent).toMatch(/\?tab=config:model$/))
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
  it('always renders precise user and assistant lifecycle times', async () => {
    const { container } = render(<Harness />)

    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    const startedAt = createdAt.getTime() / 1000

    expect(stamps).toContain(formatTimelineTimestamp(startedAt))
    expect(stamps).toContain(formatTimelineRange(startedAt, completedAt))
    expect(stamps).toContain(formatTimelineRange(startedAt + 0.05, startedAt + 0.1))
    expect(stamps).toContain(formatTimelineRange(startedAt + 0.125, startedAt + 0.5))
  })

  it('suppresses an aggregate assistant stamp that exactly duplicates its sole part', async () => {
    const startedAt = createdAt.getTime() / 1000

    const assistant = {
      ...assistantMessage(),
      content: [{ completedAt, text: 'done', timestamp: startedAt, type: 'text' }]
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)

    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    expect(stamps.filter(stamp => stamp === formatTimelineRange(startedAt, completedAt))).toHaveLength(1)
  })
})
