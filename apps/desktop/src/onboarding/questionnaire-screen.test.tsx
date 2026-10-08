import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $freeTierStatus } from '@/store/free-tier'
import { $notifications, clearNotifications } from '@/store/notifications'
import { markQuestionnaireDecided } from '@/store/onboarding-presence'
import type { FreeTierStatus } from '@/types/hermes'

import type { OnboardingRequester } from './due'
import { FIXTURES } from './fixtures.test-util'
import type { HandoffDeps } from './handoff'
import { Questionnaire, QuestionnaireScreen } from './Questionnaire'
import { $questionnaire, closeQuestionnaire, type FirstChat, openQuestionnaire, setFacts, skipStep } from './store'

/** A backend that answers `replies` and never answers anything else, so that stays in flight. */
function backend(replies: Record<string, object> = {}): OnboardingRequester {
  return <T,>(method: string) => {
    const reply = replies[method]

    // SAFETY: each method answers the shape the code under test reads back.
    return reply ? Promise.resolve(reply as T) : new Promise<T>(() => {})
  }
}

/** Inference is up and the flags save; the notice ack is refused so the free-tier store is not re-read. */
const READY_BACKEND = backend({
  'free_tier.ack_notice': { acked: false },
  'onboarding.set_run': {},
  'setup.runtime_check': { ok: true },
  'setup.status': { free_tier_account: true, free_tier_route: true, provider_configured: true, ready: true }
})

/** The review screen; the step change's exit transition runs before it mounts. */
async function renderReview({
  openDefaultChat = async () => 'runtime-1',
  openLandedChat = async () => false,
  request = backend()
}: Partial<Pick<HandoffDeps, 'openDefaultChat' | 'openLandedChat'>> & { request?: OnboardingRequester } = {}): Promise<HTMLElement> {
  render(
    <>
      <Questionnaire
        enabled={false}
        openDefaultChat={openDefaultChat}
        openLandedChat={openLandedChat}
        requestGateway={request}
      />
      <QuestionnaireScreen refreshReadiness={async () => {}} />
    </>
  )

  act(() => {
    setFacts(FIXTURES.spark)

    for (let stepId = $questionnaire.get().stepId; stepId; stepId = $questionnaire.get().stepId) {
      skipStep(stepId)
    }
  })

  return screen.findByRole('button', { name: 'Start' })
}

beforeEach(() => {
  markQuestionnaireDecided()
  $freeTierStatus.set(null)
  openQuestionnaire()
})

afterEach(() => {
  cleanup()
  closeQuestionnaire('skipped')
  clearNotifications()
})

describe('QuestionnaireScreen while Start or Skip is in flight', () => {
  it('locks the answer trail while Start waits, so the prompt it sends is the one on screen', async () => {
    fireEvent.click(await renderReview())

    const chips = screen.getAllByRole('button', { name: 'Change this answer' })

    expect(chips.length).toBeGreaterThan(0)
    expect(chips.every(chip => chip.hasAttribute('disabled'))).toBe(true)
  })

  it('locks Start, Back and the trail while Skip is saving', async () => {
    const start = await renderReview()

    fireEvent.click(screen.getByRole('button', { name: 'Skip setup' }))

    expect(start.hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Back' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getAllByRole('button', { name: 'Change this answer' }).every(chip => chip.hasAttribute('disabled'))).toBe(
      true
    )
  })
})

describe('QuestionnaireScreen after Start closed it', () => {
  it('offers the first chat again, with the same message, when it fails', async () => {
    const sent: string[] = []

    const openDefaultChat: HandoffDeps['openDefaultChat'] = async (text, onCreated) => {
      sent.push(text)

      if (sent.length === 1) {
        onCreated({ runtimeSessionId: 'runtime-1', sessionId: 'stored-1' })

        throw new Error('connection lost')
      }

      return 'runtime-2'
    }

    fireEvent.click(await renderReview({ openDefaultChat, request: READY_BACKEND }))

    const retry = await waitFor(() => {
      const action = $notifications.get().find(item => item.title === 'Could not start your first chat')?.action

      expect(action?.label).toBe('Try again')

      return action
    })

    expect($questionnaire.get().phase).toBe('done')

    act(() => retry?.onClick())

    await waitFor(() => expect(sent).toHaveLength(2))
    expect(sent[1]).toBe(sent[0])
  })

  it('opens the chat the failed try created, without sending again, once its first message is there', async () => {
    const sent: string[] = []
    const checked: string[] = []

    // session.create answered and prompt.submit was accepted, but its reply was lost.
    const openDefaultChat: HandoffDeps['openDefaultChat'] = async (text, onCreated) => {
      sent.push(text)
      onCreated({ runtimeSessionId: 'runtime-1', sessionId: 'stored-1' })

      throw new Error('prompt.submit timed out')
    }

    const openLandedChat = async ({ sessionId }: FirstChat) => {
      checked.push(sessionId)

      return true
    }

    fireEvent.click(await renderReview({ openDefaultChat, openLandedChat, request: READY_BACKEND }))

    const retry = await waitFor(() => {
      const action = $notifications.get().find(item => item.title === 'Could not start your first chat')?.action

      expect(action?.label).toBe('Try again')

      return action
    })

    act(() => retry?.onClick())

    await waitFor(() => expect(checked).toEqual(['stored-1']))
    expect(sent).toHaveLength(1)
  })
})

// A terminal free-tier refusal: the overlay hands over to the picker on its own (D23).
const REFUSED: FreeTierStatus = {
  available: false,
  enabled: true,
  error_code: 'anon_gate_closed',
  has_guest: false,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: true,
  retryable: false
}

describe('QuestionnaireScreen when the free account fails for good', () => {
  it('reports a failed run=false save and offers to save it again', async () => {
    const saves: string[] = []

    const request: OnboardingRequester = async <T,>(method: string) => {
      if (method !== 'onboarding.set_run') {
        return new Promise<T>(() => {})
      }

      saves.push(method)

      throw new Error('connection lost')
    }

    await renderReview({ request })
    act(() => $freeTierStatus.set(REFUSED))

    const retry = await waitFor(() => {
      const action = $notifications.get().find(item => item.title === 'Could not mark setup as done')?.action

      expect(action?.label).toBe('Try again')

      return action
    })

    expect($questionnaire.get().phase).toBe('failed')

    act(() => retry?.onClick())

    await waitFor(() => expect(saves).toHaveLength(2))
  })
})
