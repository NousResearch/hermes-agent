import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $settledClarifyResults, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'

import { MESSAGE_PARTS_COMPONENTS } from './message-parts'

// The row is sealed, so the turn is not running. The generic tool row needs an
// assistant-ui store this unit test does not build; what is under test is which
// branch the clarify part takes, so the fallback is stubbed to a marker.
vi.mock('@assistant-ui/react', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  useAuiState: () => false
}))

vi.mock('@/components/assistant-ui/tool/fallback', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  ToolFallback: () => <div data-testid="tool-fallback" />
}))

// The renderer's clarify card is mounted through the same dispatcher a real
// transcript uses, so this covers the row-level seal, not the card in isolation.
const ToolPart = MESSAGE_PARTS_COMPONENTS.tools.Fallback

type SealedClarifyProps = ToolCallMessagePartProps & { clarifyRequestId?: string; completedAt?: number }

function sealedClarifyRow(clarifyRequestId?: string): SealedClarifyProps {
  const args = { questions: [{ choices: ['staging', 'production'], question: 'Which deployment target?' }] }

  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    // Sealed by a turn settle while the backend is still waiting on the answer.
    completedAt: 1_700_000_000,
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'complete' },
    toolCallId: 'clarify-sealed',
    toolName: 'clarify',
    type: 'tool-call',
    ...(clarifyRequestId === undefined ? {} : { clarifyRequestId })
  }
}

const parkedRequest = {
  questions: [{ choices: ['staging', 'production'], multiSelect: false, qid: 'q0', question: 'Which deployment target?' }],
  requestId: 'request-1',
  sessionId: 'session-1'
}

beforeEach(() => {
  resetServerRequestsForTests()
  $settledClarifyResults.set({})
  $activeSessionId.set('session-1')
  setClarifyRequest(parkedRequest)
})

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  $activeSessionId.set(null)
  $settledClarifyResults.set({})
  resetServerRequestsForTests()
})

describe('a clarify row sealed while its request is still parked', () => {
  it('stays answerable when the row names the parked request', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ToolPart {...sealedClarifyRow('request-1')} />
      </I18nProvider>
    )

    expect(screen.getByText(/Which deployment target\?/)).toBeTruthy()
  })

  it('renders as history when the row names a different request', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ToolPart {...sealedClarifyRow('an-earlier-request')} />
      </I18nProvider>
    )

    expect(screen.queryByText(/Which deployment target\?/)).toBeNull()
  })

  it('renders as history when the row names no request at all', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ToolPart {...sealedClarifyRow()} />
      </I18nProvider>
    )

    expect(screen.queryByText(/Which deployment target\?/)).toBeNull()
  })
})
