import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { $displayTimestamps } from '@/store/display-timestamps'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
afterEach(() => {
  cleanup()
  $displayTimestamps.set(false)
})

it('renders hydrated and newly received cron reports as full Markdown with the existing timestamp preference', async () => {
  const report = '# Scheduled report\n\n**Results**\n\n- First item\n- Final item'

  const row: SessionMessage = {
    role: 'user',
    content: report,
    display_kind: 'internal_notification',
    display_metadata: { completion_note: [JSON.stringify(['cron', 'test-route', 'first-delivery'])] },
    timestamp: 1,
    row_id: 12
  }

  const harness = (rows: SessionMessage[]) => (
    <ThreadRuntime messages={toChatMessages(rows).map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )

  const { container, rerender } = render(harness([row]))
  await waitFor(() => expect(container.querySelector('[data-slot="aui_cron-report"]')).toBeTruthy())
  const notice = container.querySelector('[data-slot="aui_cron-report"]')!
  await waitFor(() => expect(notice.querySelector('strong, [data-streamdown="strong"]')?.textContent).toBe('Results'))
  expect(notice.querySelector('h1')?.textContent).toBe('Scheduled report')
  expect(notice.querySelectorAll('li')).toHaveLength(2)
  expect(notice.textContent).toContain('Final item')
  expect(container.querySelector('[data-role="user"]')).toBeNull()
  expect(container.querySelector('.sticky-human-clamp')).toBeNull()
  expect(notice.querySelector('button, details')).toBeNull()
  expect(notice.querySelector('time')).toBeNull()

  act(() => $displayTimestamps.set(true))
  await waitFor(() => expect(notice.querySelector('time')?.getAttribute('datetime')).toBe('1970-01-01T00:00:01.000Z'))
  // Leading whitespace is Markdown syntax, not expendable message padding.
  const indentedReport = '    first line\n    second line'
  rerender(harness([row, { ...row, content: indentedReport, timestamp: 2, row_id: 13 }]))
  await waitFor(() => expect(container.querySelectorAll('[data-slot="aui_cron-report"]')).toHaveLength(2))
  const appended = container.querySelectorAll('[data-slot="aui_cron-report"]')[1]
  await waitFor(() => expect(appended.querySelector('code')?.textContent).toContain('first line'))
  expect(appended.querySelector('code')?.textContent).toContain('second line')
  expect(container.querySelector('[data-role="user"]')).toBeNull()
})
