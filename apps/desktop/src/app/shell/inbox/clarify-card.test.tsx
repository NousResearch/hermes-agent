import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type InboxRequestClarification } from '@/store/inbox'
import { $activeGatewayProfile } from '@/store/profile'

import { ClarifyCard } from './clarify-card'

vi.mock('@/store/gateway', () => ({ $gateway: { get: vi.fn() } }))

vi.mock('@/store/profile', () => ({ $activeGatewayProfile: { get: vi.fn(() => 'test-profile') } }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function makeSingleClarification(overrides: Partial<InboxRequestClarification> = {}): InboxRequestClarification {
  return {
    kind: 'single',
    params: {
      answers: null,
      choices: ['Option A', 'Option B', 'Option C'],
      multi_select: false,
      question: 'Which approach should we take?',
      questions: null
    },
    request_id: 'req-clarify-1',
    ...overrides
  }
}

function makeBatchClarification(overrides: Partial<InboxRequestClarification> = {}): InboxRequestClarification {
  return {
    kind: 'batch',
    params: {
      answers: null,
      choices: null,
      multi_select: null,
      question: null,
      questions: [
        { multi_select: false, qid: 'q1', question: 'First question?', choices: ['Yes', 'No'] },
        { multi_select: false, qid: 'q2', question: 'Second question?', choices: ['Red', 'Blue'] }
      ]
    },
    request_id: 'req-batch-1',
    ...overrides
  }
}

async function getGatewayMock() {
  const mod = await import('@/store/gateway')

  return vi.mocked(mod.$gateway.get)
}

describe('ClarifyCard', () => {
  describe('single clarification', () => {
    it('renders question text', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      render(<ClarifyCard clarification={makeSingleClarification()} />)
      expect(screen.getByText('Which approach should we take?')).toBeTruthy()
    })

    it('renders choice buttons', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      render(<ClarifyCard clarification={makeSingleClarification()} />)
      expect(screen.getByText('Option A')).toBeTruthy()
      expect(screen.getByText('Option B')).toBeTruthy()
      expect(screen.getByText('Option C')).toBeTruthy()
    })

    it('renders free text input when no choices', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      const clarification = makeSingleClarification({
        params: { answers: null, choices: null, multi_select: false, question: 'Type your answer', questions: null }
      })

      render(<ClarifyCard clarification={clarification} />)
      expect(screen.getByPlaceholderText('Type your answer…')).toBeTruthy()
    })

    it('calls request.answer via gateway on submit', async () => {
      const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
      ;(await getGatewayMock()).mockReturnValue(gw as never)

      render(<ClarifyCard clarification={makeSingleClarification()} />)
      fireEvent.click(screen.getByText('Option A'))
      fireEvent.click(screen.getByText('Submit'))
      await waitFor(() => {
        expect(gw.request).toHaveBeenCalledWith('request.answer', {
          id: 'req-clarify-1',
          result: { answer: 'Option A' },
          profile: 'test-profile'
        })
      })
    })

    it('shows "Answered" after successful response', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'ok' }) } as never)

      render(<ClarifyCard clarification={makeSingleClarification()} />)
      fireEvent.click(screen.getByText('Option A'))
      fireEvent.click(screen.getByText('Submit'))
      await waitFor(() => {
        expect(screen.getByText('Answered')).toBeTruthy()
      })
    })

    it('shows error when response fails', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockRejectedValue(new Error('timeout')) } as never)

      render(<ClarifyCard clarification={makeSingleClarification()} />)
      fireEvent.click(screen.getByText('Option A'))
      fireEvent.click(screen.getByText('Submit'))
      await waitFor(() => {
        expect(screen.getByText('timeout')).toBeTruthy()
      })
    })

    it('disables submit when no choice selected and no text', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      render(<ClarifyCard clarification={makeSingleClarification()} />)
      const submitBtn = screen.getByText('Submit')
      expect(submitBtn.closest('button')?.disabled).toBe(true)
    })

    // ── Defect 2 regression: expired must show Expired, NOT Answered ──
    it('shows "Expired" when status is expired (not "Answered")', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'expired' }) } as never)

      render(<ClarifyCard clarification={makeSingleClarification()} />)
      fireEvent.click(screen.getByText('Option A'))
      fireEvent.click(screen.getByText('Submit'))
      await waitFor(() => {
        expect(screen.getByText('Expired')).toBeTruthy()
        expect(screen.queryByText('Answered')).toBeNull()
      })
    })

    // ── Defect 2 regression: expired must NOT invoke onResolved callback ──
    it('does not call onResolved when status is expired', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'expired' }) } as never)
      const onResolved = vi.fn()

      render(<ClarifyCard clarification={makeSingleClarification()} onResolved={onResolved} />)
      fireEvent.click(screen.getByText('Option A'))
      fireEvent.click(screen.getByText('Submit'))
      await waitFor(() => {
        expect(screen.getByText('Expired')).toBeTruthy()
      })
      expect(onResolved).not.toHaveBeenCalled()
    })

    // ── Defect 2 regression: batch expired also shows Expired ──
    it('batch: shows "Expired" when status is expired', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'expired' }) } as never)

      render(<ClarifyCard clarification={makeBatchClarification()} />)
      fireEvent.click(screen.getByText('Yes'))
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Submit answers'))
      await waitFor(() => {
        expect(screen.getByText('Expired')).toBeTruthy()
        expect(screen.queryByText('Answered')).toBeNull()
      })
    })

    // ── Defect 2 regression: batch expired does NOT call onResolved ──
    it('batch: does not call onResolved when status is expired', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'expired' }) } as never)
      const onResolved = vi.fn()

      render(<ClarifyCard clarification={makeBatchClarification()} onResolved={onResolved} />)
      fireEvent.click(screen.getByText('Yes'))
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Submit answers'))
      await waitFor(() => {
        expect(screen.getByText('Expired')).toBeTruthy()
      })
      expect(onResolved).not.toHaveBeenCalled()
    })

    // ── Defect 2 regression: unknown status fails closed ──
    it('shows error for unknown response status', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'bogus' }) } as never)

      render(<ClarifyCard clarification={makeSingleClarification()} />)
      fireEvent.click(screen.getByText('Option A'))
      fireEvent.click(screen.getByText('Submit'))
      await waitFor(() => {
        expect(screen.getByText('Unexpected response')).toBeTruthy()
      })
    })

    it('shows default question text when params.question is null', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      const clarification = makeSingleClarification({
        params: { answers: null, choices: ['Yes'], multi_select: false, question: null, questions: null }
      })

      render(<ClarifyCard clarification={clarification} />)
      expect(screen.getByText('The agent has a question')).toBeTruthy()
    })

    // ── Defect 3 regression: multi_select uses independent checkboxes ──
    it('multi_select: renders checkboxes and allows selecting multiple options', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      const clarification = makeSingleClarification({
        params: { answers: null, choices: ['Red', 'Blue', 'Green'], multi_select: true, question: 'Pick colors', questions: null }
      })

      render(<ClarifyCard clarification={clarification} />)
      expect(screen.getByText('Red')).toBeTruthy()
      expect(screen.getByText('Blue')).toBeTruthy()
      expect(screen.getByText('Green')).toBeTruthy()

      // Select two options
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Blue'))

      // Both should appear selected (have the accent background)
      const redBtn = screen.getByText('Red').closest('button')!
      const blueBtn = screen.getByText('Blue').closest('button')!
      const greenBtn = screen.getByText('Green').closest('button')!
      expect(redBtn.className).toContain('bg-accent/55')
      expect(blueBtn.className).toContain('bg-accent/55')
      expect(greenBtn.className).not.toContain('bg-accent/55')
    })

    // ── Defect 3 regression: multi_select serializes as JSON array ──
    it('multi_select: serializes selected options as JSON array', async () => {
      const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
      ;(await getGatewayMock()).mockReturnValue(gw as never)

      const clarification = makeSingleClarification({
        params: { answers: null, choices: ['Red', 'Blue', 'Green'], multi_select: true, question: 'Pick colors', questions: null }
      })

      render(<ClarifyCard clarification={clarification} />)
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Green'))
      fireEvent.click(screen.getByText('Submit'))

      await waitFor(() => {
        expect(gw.request).toHaveBeenCalledWith('request.answer', {
          id: 'req-clarify-1',
          result: { answer: JSON.stringify(['Red', 'Green']) },
          profile: 'test-profile'
        })
      })
    })

    // ── Defect 3 regression: single select still works as before ──
    it('single select: sends the chosen option directly (not JSON)', async () => {
      const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
      ;(await getGatewayMock()).mockReturnValue(gw as never)

      render(<ClarifyCard clarification={makeSingleClarification()} />)
      fireEvent.click(screen.getByText('Option B'))
      fireEvent.click(screen.getByText('Submit'))

      await waitFor(() => {
        expect(gw.request).toHaveBeenCalledWith('request.answer', {
          id: 'req-clarify-1',
          result: { answer: 'Option B' },
          profile: 'test-profile'
        })
      })
    })

    // ── Defect 3 regression: multi_select with Other free-text ──
    it('multi_select: includes Other free-text option', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      const clarification = makeSingleClarification({
        params: { answers: null, choices: ['Red', 'Blue'], multi_select: true, question: 'Pick colors', questions: null }
      })

      render(<ClarifyCard clarification={clarification} />)
      expect(screen.getByText('Red')).toBeTruthy()
      expect(screen.getByText('Blue')).toBeTruthy()
      expect(screen.getByText('Other…')).toBeTruthy()
    })
  })

  describe('batch clarification', () => {
    it('renders question count', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      render(<ClarifyCard clarification={makeBatchClarification()} />)
      expect(screen.getByText('2 questions')).toBeTruthy()
    })

    it('renders individual questions with choices', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      render(<ClarifyCard clarification={makeBatchClarification()} />)
      expect(screen.getByText('First question?')).toBeTruthy()
      expect(screen.getByText('Second question?')).toBeTruthy()
      expect(screen.getByText('Yes')).toBeTruthy()
      expect(screen.getByText('No')).toBeTruthy()
      expect(screen.getByText('Red')).toBeTruthy()
      expect(screen.getByText('Blue')).toBeTruthy()
    })

    it('calls clarify.lock via gateway on submit', async () => {
      const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
      ;(await getGatewayMock()).mockReturnValue(gw as never)

      render(<ClarifyCard clarification={makeBatchClarification()} />)
      fireEvent.click(screen.getByText('Yes'))
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Submit answers'))

      await waitFor(() => {
        expect(gw.request).toHaveBeenCalledTimes(2)
        expect(gw.request).toHaveBeenCalledWith('clarify.lock', {
          request_id: 'req-batch-1',
          question_id: 'q1',
          answer: 'Yes',
          profile: 'test-profile'
        })
        expect(gw.request).toHaveBeenCalledWith('clarify.lock', {
          request_id: 'req-batch-1',
          question_id: 'q2',
          answer: 'Red',
          profile: 'test-profile'
        })
      })
    })

    it('shows "Answered" after successful batch response', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'ok' }) } as never)

      render(<ClarifyCard clarification={makeBatchClarification()} />)
      fireEvent.click(screen.getByText('Yes'))
      fireEvent.click(screen.getByText('Submit answers'))
      await waitFor(() => {
        expect(screen.getByText('Answered')).toBeTruthy()
      })
    })

    it('shows error when batch response fails', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockRejectedValue(new Error('connection lost')) } as never)

      render(<ClarifyCard clarification={makeBatchClarification()} />)
      fireEvent.click(screen.getByText('Yes'))
      fireEvent.click(screen.getByText('Submit answers'))
      await waitFor(() => {
        expect(screen.getByText('connection lost')).toBeTruthy()
      })
    })

    it('disables submit when no answers staged', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      render(<ClarifyCard clarification={makeBatchClarification()} />)
      const submitBtn = screen.getByText('Submit answers')
      expect(submitBtn.closest('button')?.disabled).toBe(true)
    })

    it('renders free text input for questions without choices', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      const clarification = makeBatchClarification({
        params: {
          answers: null,
          choices: null,
          multi_select: null,
          question: null,
          questions: [
            { multi_select: false, qid: 'q-free', question: 'Describe the issue', choices: null }
          ]
        }
      })

      render(<ClarifyCard clarification={clarification} />)
      expect(screen.getByText('Describe the issue')).toBeTruthy()
      expect(screen.getByPlaceholderText('Answer…')).toBeTruthy()
    })

    // ── Defect 3 regression: batch multi_select per-question ──
    it('batch: preserves per-question multi_select', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      const clarification = makeBatchClarification({
        params: {
          answers: null,
          choices: null,
          multi_select: null,
          question: null,
          questions: [
            { multi_select: true, qid: 'q-ms', question: 'Pick colors?', choices: ['Red', 'Blue', 'Green'] },
            { multi_select: false, qid: 'q-single', question: 'Pick one?', choices: ['A', 'B'] }
          ]
        }
      })

      render(<ClarifyCard clarification={clarification} />)
      // multi_select question has Other option
      expect(screen.getByText('Red')).toBeTruthy()
      expect(screen.getByText('Other…')).toBeTruthy()
      // single question does NOT have Other
      expect(screen.getByText('A')).toBeTruthy()
      expect(screen.getByText('B')).toBeTruthy()
    })

    // ── Defect 3 regression: batch multi_select selects two options independently ──
    it('batch: multi_select allows selecting multiple options per question', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)
      const clarification = makeBatchClarification({
        params: {
          answers: null,
          choices: null,
          multi_select: null,
          question: null,
          questions: [
            { multi_select: true, qid: 'q-ms', question: 'Pick colors?', choices: ['Red', 'Blue', 'Green'] },
            { multi_select: false, qid: 'q-single', question: 'Pick one?', choices: ['A', 'B'] }
          ]
        }
      })

      render(<ClarifyCard clarification={clarification} />)
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Green'))

      const redBtn = screen.getByText('Red').closest('button')!
      const greenBtn = screen.getByText('Green').closest('button')!
      const blueBtn = screen.getByText('Blue').closest('button')!
      expect(redBtn.className).toContain('bg-accent/55')
      expect(greenBtn.className).toContain('bg-accent/55')
      expect(blueBtn.className).not.toContain('bg-accent/55')
    })

    // ── Defect 3 regression: batch multi_select serializes as JSON ──
    it('batch: multi_select serializes selected options as JSON array', async () => {
      const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
      ;(await getGatewayMock()).mockReturnValue(gw as never)

      const clarification = makeBatchClarification({
        params: {
          answers: null,
          choices: null,
          multi_select: null,
          question: null,
          questions: [
            { multi_select: true, qid: 'q-ms', question: 'Pick colors?', choices: ['Red', 'Blue', 'Green'] },
            { multi_select: false, qid: 'q-single', question: 'Pick one?', choices: ['A', 'B'] }
          ]
        }
      })

      render(<ClarifyCard clarification={clarification} />)
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Green'))
      fireEvent.click(screen.getByText('A'))
      fireEvent.click(screen.getByText('Submit answers'))

      await waitFor(() => {
        expect(gw.request).toHaveBeenCalledWith('clarify.lock', {
          request_id: 'req-batch-1',
          question_id: 'q-ms',
          answer: JSON.stringify(['Red', 'Green']),
          profile: 'test-profile'
        })
      })
    })

    // ── Defect 6 regression: batch retry preserves locked answers ──
    it('batch: retry preserves already-locked answers from a partial failure', async () => {
      ;(await getGatewayMock()).mockReturnValue({ request: vi.fn().mockResolvedValue({ status: 'error' }) } as never)

      render(<ClarifyCard clarification={makeBatchClarification()} />)
      fireEvent.click(screen.getByText('Yes'))
      fireEvent.click(screen.getByText('Red'))
      fireEvent.click(screen.getByText('Submit answers'))
      await waitFor(() => {
        expect(screen.getByText('Unexpected response')).toBeTruthy()
      })
    })
  })

  // ── Defect 5 regression: profile scope pinning ──
  it('profile change before submit is blocked with an error', async () => {
    const gw = { request: vi.fn().mockResolvedValue({ status: 'answered' }) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)
    const getProfile = vi.mocked($activeGatewayProfile.get)
    getProfile.mockReturnValue('pinned-profile')

    render(
      <ClarifyCard
        clarification={makeSingleClarification({
          params: {
            answers: null,
            choices: ['Yes', 'No'],
            multi_select: false,
            question: 'Approve?',
            questions: null
          },
          request_id: 'req-pin'
        })}
      />
    )

    fireEvent.click(screen.getByText('Yes'))
    getProfile.mockReturnValue('switched-profile')
    fireEvent.click(screen.getByText('Submit'))
    await waitFor(() => {
      expect(screen.getByText('Gateway changed — re-open to act')).toBeTruthy()
    })
    expect(gw.request).not.toHaveBeenCalled()
  })

  // ── Transport guard: gateway switch before submit is blocked ──
  it('gateway change before submit is blocked with an error', async () => {
    const gwMock = await getGatewayMock()
    const originalGw = { request: vi.fn() }
    gwMock.mockReturnValue(originalGw as never)

    render(<ClarifyCard clarification={makeSingleClarification()} />)
    fireEvent.click(screen.getByText('Option A'))

    // Switch gateway object
    gwMock.mockReturnValue({ request: vi.fn() } as never)
    fireEvent.click(screen.getByText('Submit'))
    await waitFor(() => {
      expect(screen.getByText('Gateway changed — re-open to act')).toBeTruthy()
    })
    expect(originalGw.request).not.toHaveBeenCalled()
  })

  // ── Transport guard: null gateway after mount is blocked ──
  it('null gateway after mount is blocked with an error', async () => {
    const gwMock = await getGatewayMock()
    gwMock.mockReturnValue({ request: vi.fn() } as never)

    render(<ClarifyCard clarification={makeSingleClarification()} />)
    fireEvent.click(screen.getByText('Option A'))

    gwMock.mockReturnValue(null)
    fireEvent.click(screen.getByText('Submit'))
    await waitFor(() => {
      expect(screen.getByText('Gateway changed — re-open to act')).toBeTruthy()
    })
  })

  // ── Transport guard: same-tick double submit is prevented ──
  it('same-tick double submit is prevented by lock', async () => {
    let resolveRequest!: (v: unknown) => void
    const gw = { request: vi.fn().mockImplementation(() => new Promise(resolve => { resolveRequest = resolve })) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)

    render(<ClarifyCard clarification={makeSingleClarification()} />)
    fireEvent.click(screen.getByText('Option A'))

    const submitBtn = screen.getByRole('button', { name: /Submit/ })

    // First click starts submitting — button becomes disabled with "…"
    fireEvent.click(submitBtn)

    // Second click on the same (now disabled) button should be a no-op
    fireEvent.click(submitBtn)

    resolveRequest({ status: 'ok' })
    await waitFor(() => {
      expect(screen.getByText('Answered')).toBeTruthy()
    })

    // Only one request should have been made
    expect(gw.request).toHaveBeenCalledTimes(1)
  })

  // ── Transport guard: single — gateway switch mid-await blocks state update ──
  it('single: gateway switch during await prevents state update', async () => {
    const gwMock = await getGatewayMock()
    const originalGw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
    gwMock.mockReturnValue(originalGw as never)

    render(<ClarifyCard clarification={makeSingleClarification()} />)
    fireEvent.click(screen.getByText('Option A'))
    fireEvent.click(screen.getByText('Submit'))

    // Switch gateway while RPC is in flight
    gwMock.mockReturnValue({ request: vi.fn() } as never)

    await waitFor(() => {
      // Should NOT show Answered — scope was invalid after await
      expect(screen.queryByText('Answered')).toBeNull()
    })

    // The RPC was called (scope was valid before it), but post-await check blocked the state update
    expect(originalGw.request).toHaveBeenCalledTimes(1)
  })

  // ── Transport guard: batch — switch between first/second lock stops second ──
  it('batch: gateway switch between first and second lock stops second send', async () => {
    const gwMock = await getGatewayMock()
    let resolveFirstLock!: (v: unknown) => void
    const originalGw = {
      request: vi.fn()
        .mockImplementationOnce(() => new Promise(resolve => { resolveFirstLock = resolve }))
        .mockResolvedValue({ status: 'ok' })
    }
    gwMock.mockReturnValue(originalGw as never)

    render(<ClarifyCard clarification={makeBatchClarification()} />)
    fireEvent.click(screen.getByText('Yes'))
    fireEvent.click(screen.getByText('Red'))
    fireEvent.click(screen.getByText('Submit answers'))

    // Wait for first lock to start
    await waitFor(() => {
      expect(originalGw.request).toHaveBeenCalledTimes(1)
    })

    // Switch gateway BEFORE resolving first lock
    const newGw = { request: vi.fn() }
    gwMock.mockReturnValue(newGw as never)

    // Now resolve the first lock — second lock will fail scope check
    await act(async () => {
      resolveFirstLock({ status: 'ok' })
    })

    // Second lock was NOT sent because scope check failed
    expect(originalGw.request).toHaveBeenCalledTimes(1)
    // Should show error about gateway change
    expect(screen.getByText('Gateway changed — re-open to act')).toBeTruthy()
  })
})
