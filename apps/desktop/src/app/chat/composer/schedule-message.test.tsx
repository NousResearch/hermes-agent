import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ScheduleMessageControl } from '@/app/chat/composer/schedule-message'
import { $scheduledMessagesBySession } from '@/store/scheduled-messages'

/**
 * (#111873) The composer's Schedule affordance: the picker's local wall clock must
 * reach the backend as the right instant, and the reply must land in the store.
 * Pure wire/time helpers are covered in `@/store/scheduled-messages.test.ts`.
 */

const gatewayMocks = vi.hoisted(() => ({ request: vi.fn() }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: gatewayMocks.request })
}))

// Radix measures its floating content with a ResizeObserver; jsdom has none.
class TestResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

async function scheduleFlow(dueValue = '2030-01-02T03:04') {
  fireEvent.click(screen.getByLabelText('Schedule'))

  const input = (await screen.findByLabelText('Send at')) as HTMLInputElement

  fireEvent.change(input, { target: { value: dueValue } })
  fireEvent.click(screen.getByRole('button', { name: 'Schedule message' }))
}

beforeEach(() => {
  gatewayMocks.request.mockReset()
  $scheduledMessagesBySession.set({})
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('ScheduleMessageControl', () => {
  it('sends the draft and the picked local time, then clears the composer', async () => {
    gatewayMocks.request.mockResolvedValue({
      messages: [{ due_at: 1_800_000_000, id: 'm1', status: 'pending', text: 'run it later' }]
    })
    const onScheduled = vi.fn()

    render(
      <ScheduleMessageControl getText={() => '  run it later  '} onScheduled={onScheduled} sessionId="sess-1" />
    )

    await scheduleFlow()
    await waitFor(() => expect(gatewayMocks.request).toHaveBeenCalled())

    const [method, params] = gatewayMocks.request.mock.calls[0] as [string, Record<string, unknown>]

    expect(method).toBe('schedule.message.create')
    expect(params.session_id).toBe('sess-1')
    expect(params.text).toBe('run it later')
    // Wire unit is epoch SECONDS of the picked local instant — `new Date()` on a
    // zone-less value is LOCAL by spec, which is exactly what the input means.
    expect(params.due_at).toBe(new Date('2030-01-02T03:04').getTime() / 1000)

    await waitFor(() => expect(onScheduled).toHaveBeenCalledTimes(1))
    expect($scheduledMessagesBySession.get()['sess-1']).toHaveLength(1)
  })

  it('refuses to schedule an empty draft', async () => {
    const onScheduled = vi.fn()

    render(<ScheduleMessageControl getText={() => '   '} onScheduled={onScheduled} sessionId="sess-1" />)

    await scheduleFlow()

    expect(gatewayMocks.request).not.toHaveBeenCalled()
    expect(onScheduled).not.toHaveBeenCalled()
  })

  it('keeps the draft when the backend rejects the schedule', async () => {
    gatewayMocks.request.mockRejectedValue(new Error('gateway down'))
    const onScheduled = vi.fn()

    render(<ScheduleMessageControl getText={() => 'keep me'} onScheduled={onScheduled} sessionId="sess-1" />)

    await scheduleFlow()
    await waitFor(() => expect(gatewayMocks.request).toHaveBeenCalled())

    expect(onScheduled).not.toHaveBeenCalled()
    expect($scheduledMessagesBySession.get()['sess-1']).toBeUndefined()
  })
})
