import { act, cleanup, fireEvent, render, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { fmtDate } from '@/lib/time'
import { $displayTimestamps } from '@/store/display-timestamps'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

beforeEach(() => {
  stubThreadEnvironment()
  $displayTimestamps.set(false)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

const seconds = (date: Date) => date.getTime() / 1000

const exchange = (label: string, date: Date): SessionMessage[] => [
  { role: 'user', content: `Question ${label}`, timestamp: seconds(date) },
  { role: 'assistant', content: `Answer ${label}`, timestamp: seconds(date) + 20 }
]

const runtimeMessages = (messages: SessionMessage[]) => toChatMessages(messages).map(toRuntimeMessage)

it('labels the first visible turn and each new local day, not every turn', async () => {
  const first = new Date(2026, 9, 22, 23, 50)
  const next = new Date(2026, 9, 23, 0, 10)

  const messages = runtimeMessages([
    ...exchange('first', first),
    ...exchange('next', next),
    ...exchange('same-day', new Date(2026, 9, 23, 1))
  ])

  const { container } = render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(container.textContent).toContain('Answer same-day'))
  const dates = within(container).getAllByRole('separator')
  expect(dates.map(el => el.textContent)).toEqual([fmtDate.format(first), fmtDate.format(next)])
  expect(container.querySelectorAll('[data-slot="aui_turn-pair"]')).toHaveLength(3)
  expect(
    dates[0].compareDocumentPosition(within(container).getByText('Question first')) & Node.DOCUMENT_POSITION_FOLLOWING
  ).toBeTruthy()
  expect(
    dates[1].compareDocumentPosition(within(container).getByText('Question next')) & Node.DOCUMENT_POSITION_FOLLOWING
  ).toBeTruthy()
})

const dateLabels = (container: HTMLElement) => within(container).queryAllByRole('separator').map(el => el.textContent)

const renderThread = (messages: ReturnType<typeof runtimeMessages>) =>
  render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )

it.each([undefined, null, 0, -1, NaN, Infinity, -Infinity, 1e20, '1792705800'])(
  'does not date a group from createdAt when its authoritative timestamp is %s',
  async value => {
    const messages = runtimeMessages([
      { role: 'user', content: 'Unknown question' },
      { role: 'assistant', content: 'Dated answer', timestamp: seconds(new Date(2026, 9, 23, 12)) }
    ])

    const withInvalidTimestamp = messages.map(message =>
      message.role === 'user'
        ? { ...message, metadata: { ...message.metadata, custom: { timelineTimestamp: value } } }
        : message
    )

    const { container } = renderThread(withInvalidTimestamp)

    await waitFor(() => expect(container.textContent).toContain('Dated answer'))
    expect(messages[0].createdAt).toBeInstanceOf(Date)
    expect(dateLabels(container)).toEqual([])
  }
)

it('starts a new dated run after an unknown group, even on the same day', async () => {
  const day = new Date(2026, 9, 23, 12)

  const messages = runtimeMessages([
    ...exchange('dated', day),
    { role: 'user', content: 'Unknown question' },
    { role: 'assistant', content: 'Unknown answer' },
    ...exchange('dated-again', day)
  ])

  const { container } = renderThread(messages)

  await waitFor(() => expect(container.textContent).toContain('Answer dated-again'))
  expect(dateLabels(container)).toEqual([fmtDate.format(day), fmtDate.format(day)])
  const groups = container.querySelectorAll('[data-slot="aui_message-group"]')
  expect(within(groups[1] as HTMLElement).queryByRole('separator')).toBeNull()
})

it('keeps a response and its continuation across midnight inside their opening turn', async () => {
  const before = new Date(2026, 9, 22, 23, 59)
  const after = new Date(2026, 9, 23, 0, 1)

  const messages = runtimeMessages([
    { role: 'user', content: 'Late question', timestamp: seconds(before) },
    { role: 'assistant', content: 'After-midnight answer', timestamp: seconds(after) },
    { role: 'assistant', content: 'After-midnight continuation', timestamp: seconds(after) + 30 },
    ...exchange('next-turn', after)
  ])

  const { container } = renderThread(messages)

  await waitFor(() => expect(container.textContent).toContain('After-midnight continuation'))
  expect(dateLabels(container)).toEqual([fmtDate.format(before), fmtDate.format(after)])
  const turns = container.querySelectorAll('[data-slot="aui_turn-pair"]')
  expect(turns).toHaveLength(2)
  expect(turns[0].textContent).toContain('After-midnight answer')
  expect(turns[0].textContent).toContain('After-midnight continuation')
  expect(turns[0].querySelector('[role="separator"]')).toBeNull()
})

it('labels non-monotonic days in transcript order without sorting turns', async () => {
  const days = [new Date(2026, 9, 23, 12), new Date(2026, 9, 22, 12), new Date(2026, 9, 23, 13)]
  const { container } = renderThread(runtimeMessages(days.flatMap((day, index) => exchange(String(index), day))))

  await waitFor(() => expect(container.textContent).toContain('Answer 2'))
  expect(dateLabels(container)).toEqual(days.map(day => fmtDate.format(day)))
  expect([...container.querySelectorAll('[data-slot="aui_turn-pair"]')].map(el => el.textContent)).toEqual([
    expect.stringContaining('Question 0'),
    expect.stringContaining('Question 1'),
    expect.stringContaining('Question 2')
  ])
})

it.each([
  ['European spring', 2, 29],
  ['European autumn', 9, 25],
  ['American spring', 2, 8],
  ['American autumn', 10, 1]
] as const)('uses civil dates through the %s daylight-saving transition', async (_name, month, day) => {
  const morning = new Date(2026, month, day, 0, 30)
  const evening = new Date(2026, month, day, 23, 30)
  const tomorrow = new Date(2026, month, day + 1, 0, 10)

  const { container } = renderThread(
    runtimeMessages([
      ...exchange('morning', morning),
      ...exchange('evening', evening),
      ...exchange('tomorrow', tomorrow)
    ])
  )

  await waitFor(() => expect(container.textContent).toContain('Answer tomorrow'))
  expect(dateLabels(container)).toEqual([fmtDate.format(morning), fmtDate.format(tomorrow)])
})

it.each([false, true])('moves the first date on prepend without remounting messages (cross-day: %s)', async crossDay => {
  const day = new Date(2026, 9, 23, 12)
  const older = new Date(2026, 9, crossDay ? 22 : 23, 8)
  const messages = runtimeMessages([...exchange('older', older), ...exchange('visible', day)])
  const { container, rerender } = renderThread(messages.slice(2))

  await waitFor(() => expect(dateLabels(container)).toEqual([fmtDate.format(day)]))
  const answer = container.querySelector('[data-slot="aui_assistant-message-root"]')
  const question = within(container).getByText('Question visible')
  expect(answer).not.toBeNull()
  rerender(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(container.querySelectorAll('[data-slot="aui_turn-pair"]')).toHaveLength(2))
  expect(dateLabels(container)).toEqual(crossDay ? [fmtDate.format(older), fmtDate.format(day)] : [fmtDate.format(day)])
  expect(within(container).getByText('Question visible')).toBe(question)
  expect(container.querySelectorAll('[data-slot="aui_assistant-message-root"]')[1]).toBe(answer)
})

it('updates dates from timestamp-only corrections without replacing messages', async () => {
  const day = new Date(2026, 9, 22, 12)
  const next = new Date(2026, 9, 23, 12)
  const messages = runtimeMessages([...exchange('first', day), ...exchange('second', next)])
  const { container, rerender } = renderThread(messages)

  await waitFor(() => expect(dateLabels(container)).toHaveLength(2))
  const question = within(container).getByText('Question second')

  const corrected = messages.map((message, index) =>
    index === 2 && message.role === 'user'
      ? { ...message, metadata: { ...message.metadata, custom: { timelineTimestamp: seconds(day) } } }
      : message
  )

  rerender(
    <ThreadRuntime messages={corrected}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(dateLabels(container)).toEqual([fmtDate.format(day)]))
  expect(within(container).getByText('Question second')).toBe(question)
})

it('dates standalone messages and the first group in a historical slice', async () => {
  const day = new Date(2026, 9, 22, 12)
  const next = new Date(2026, 9, 23, 12)

  const messages = runtimeMessages([
    ...exchange('outside-window', day),
    { role: 'assistant', content: 'Historical standalone', timestamp: seconds(day) },
    ...exchange('visible', next)
  ])

  const { container } = renderThread(messages.slice(2))

  await waitFor(() => expect(container.textContent).toContain('Historical standalone'))
  expect(dateLabels(container)).toEqual([fmtDate.format(day), fmtDate.format(next)])
  expect(container.querySelectorAll('[data-slot="aui_message-group"]')).toHaveLength(2)
  expect(container.textContent).not.toContain('outside-window')
})

it('labels the first rendered group when an earlier same-day group is outside the render budget', async () => {
  const day = new Date(2026, 9, 23, 12)

  const messages = runtimeMessages(
    Array.from({ length: 40 }, (_, index) => exchange(`window-${index} ${'x'.repeat(5000)}`, day)).flat()
  )

  const { container } = renderThread(messages)

  // Let stepped first-paint backfill settle before exercising Show earlier.
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 100))
  })
  const groups = container.querySelectorAll('[data-slot="aui_message-group"]')
  expect(groups.length).toBeGreaterThan(0)
  expect(groups.length).toBeLessThan(40)
  expect(dateLabels(container)).toEqual([fmtDate.format(day)])
  expect(within(groups[0] as HTMLElement).getByRole('separator').textContent).toBe(fmtDate.format(day))
  const tail = container.querySelectorAll('[data-slot="aui_assistant-message-root"]')
  const lastAnswer = tail[tail.length - 1]

  fireEvent.click(within(container).getByRole('button', { name: /show earlier/i }))
  await waitFor(() => expect(container.querySelectorAll('[data-slot="aui_message-group"]').length).toBeGreaterThan(groups.length))
  expect(dateLabels(container)).toEqual([fmtDate.format(day)])
  const expanded = container.querySelectorAll('[data-slot="aui_assistant-message-root"]')
  expect(expanded[expanded.length - 1]).toBe(lastAnswer)
})
