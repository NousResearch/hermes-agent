import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { AsyncResultDisclosure } from './system-message'

import { Thread } from '.'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const timestamp = new Date('2026-05-01T00:00:00.000Z')
stubThreadEnvironment()

function Harness({ asyncResult, text }: { asyncResult?: string; text: string }) {
  const message = {
    id: 'system-1',
    role: 'system',
    content: [{ type: 'text', text }],
    createdAt: timestamp,
    metadata: { custom: { asyncResult, timelineTimestamp: timestamp.getTime() / 1000 } }
  } as unknown as ThreadMessage

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function expectTimestampSeparated(container: HTMLElement, precedingText: string) {
  const row = container.querySelector('[data-role="system"]')
  const stamp = row?.querySelector('[data-slot="timeline-timestamp"]')?.textContent

  expect(stamp).toBeTruthy()
  expect(row?.textContent).toContain(`${precedingText} ${stamp}`)
}

afterEach(cleanup)

describe('system message timestamp text separation', () => {
  it('separates an ordinary system row timestamp in accessible and copied text', () => {
    const { container } = render(<Harness text="Review saved." />)

    expectTimestampSeparated(container, 'Review saved.')
  })

  it('separates a slash-status timestamp in accessible and copied text', () => {
    const { container } = render(<Harness text={'slash:/model\nmodel changed'} />)

    expectTimestampSeparated(container, 'model changed')
  })

  it('separates a steer timestamp in accessible and copied text', () => {
    const { container } = render(<Harness text="steer:rerun tests" />)

    expectTimestampSeparated(container, 'rerun tests')
  })

  it('separates a background-result timestamp in accessible and copied text', () => {
    const { container } = render(<Harness asyncResult="Private report details" text="1 background agent finished" />)

    expectTimestampSeparated(container, '1 background agent finished')
  })
})

describe('AsyncResultDisclosure', () => {
  it('keeps a completed background report collapsed until requested', () => {
    render(<AsyncResultDisclosure label="1 background agent finished" result="Private report details" />)

    const trigger = screen.getByRole('button', { name: '1 background agent finished' })

    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('Private report details')).toBeNull()

    fireEvent.click(trigger)

    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('Private report details')).toBeTruthy()
  })
})
