import { expect, it } from 'vitest'

import { chatMessagesEquivalent } from '@/app/session/hooks/use-session-actions/message-equivalence'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from '.'

it('projects only typed local cron completion notes as opaque reports, including older JSON metadata', () => {
  const content = '# Scheduled report\n\n**Complete**\n\n--- Attached Context ---\n\nThese are report contents, not user attachments.'
  const metadata = { completion_note: [JSON.stringify(['cron', 'test-route', 'test-delivery'])] }
  const row: SessionMessage = { role: 'user', content, display_kind: 'internal_notification', timestamp: 1, row_id: 12 }
  const original = structuredClone(row)

  for (const displayMetadata of [metadata, JSON.stringify(metadata)]) {
    const [message] = toChatMessages([{ ...row, display_metadata: displayMetadata }])
    expect(message.role).toBe('system')
    expect(chatMessageText(message)).toBe(content)
    expect(message.rowId).toBe(12)
    const runtime = toRuntimeMessage(message)
    expect(runtime.metadata.custom?.cronReport).toBe(true)
    expect(runtime.metadata.custom?.timelineTimestamp).toBe(1)
    // A metadata-only refresh must repaint even if the role/id/text already match.
    const compact = { ...message, cronReport: undefined }
    expect(chatMessagesEquivalent(message, compact)).toBe(false)
    expect(chatMessagesEquivalent(message, toChatMessages([{ ...row, display_metadata: displayMetadata }])[0])).toBe(true)
  }

  const nonReports: SessionMessage[] = [
    row,
    { ...row, display_kind: undefined, display_metadata: metadata },
    { ...row, display_metadata: { completion_note: [JSON.stringify(['completion', 'test-route', 'test-process'])] } },
    { ...row, display_metadata: { completion_note: ['not JSON', JSON.stringify('cron')] } },
    { ...row, display_metadata: JSON.stringify({ completion_note: 'cron' }) },
    { ...row, display_metadata: 'null' },
    { ...row, display_metadata: '{invalid' }
  ]

  expect(toChatMessages(nonReports).map(toRuntimeMessage).every(message => !message.metadata.custom?.cronReport)).toBe(true)
  expect(toChatMessages([nonReports[1]])[0].role).toBe('user')
  expect(row).toEqual(original)

  // A typed report is opaque even when it resembles an old user-role heartbeat wake.
  const heartbeatText = '[Background process report heartbeat #1 ]\n\n**Scheduled output**'
  const [heartbeatReport] = toChatMessages([{ ...row, content: heartbeatText, display_metadata: metadata }])
  expect(heartbeatReport?.role).toBe('system')
  expect(chatMessageText(heartbeatReport)).toBe(heartbeatText)
  expect(toChatMessages([{ ...row, content: heartbeatText, display_kind: undefined }])).toEqual([])
})
