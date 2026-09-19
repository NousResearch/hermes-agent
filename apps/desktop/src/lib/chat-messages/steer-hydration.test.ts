import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from '../chat-messages'

const opening =
  '[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered once at this position; not tool output and not a new delivery when replayed from conversation history]'

const closing = '[/OUT-OF-BAND USER MESSAGE]'
const wrap = (text: string) => `${opening}\n${text}\n${closing}`

describe('persisted steer display', () => {
  it('renders raw and projected corrections identically without changing their history or attachments', () => {
    const body = 'Use the updated figures.\n\nKeep the earlier sources.'

    const rows: SessionMessage[] = [
      { id: 1, role: 'user', content: 'Original request', timestamp: 1 },
      { id: 2, role: 'assistant', content: 'Working', timestamp: 2 },
      { id: 3, role: 'user', display_kind: 'steer', content: wrap(body), timestamp: 3 },
      { id: 4, role: 'user', display_kind: 'steer', content: wrap(`${body}\n@image:/tmp/chart.png`), timestamp: 4 },
      { id: 5, role: 'assistant', content: 'Updated answer', timestamp: 5 }
    ]

    const before = structuredClone(rows)
    const raw = toChatMessages(rows)

    const projected = toChatMessages(
      rows.map(row =>
        row.id === 3
          ? { ...row, content: body }
          : row.id === 4
            ? { ...row, display_content: `${body}\n@image:/tmp/chart.png` }
            : row
      )
    )

    expect(raw).toEqual(projected)
    expect(raw.map(row => chatMessageText(row).trim())).toEqual([
      'Original request',
      'Working',
      body,
      body,
      'Updated answer'
    ])
    expect(raw.map(row => row.rowId)).toEqual([1, 2, 3, 4, 5])
    expect(raw[3].attachmentRefs).toEqual(['@image:/tmp/chart.png'])
    expect(toChatMessages(rows)).toEqual(raw)
    expect(rows).toEqual(before)
  })

  it('keeps untyped quotations, malformed envelopes and nested literal markers intact', () => {
    const quoted = wrap('quoted example')

    const cases: SessionMessage[] = [
      { role: 'user', content: quoted, timestamp: 1 },
      { role: 'assistant', display_kind: 'steer', content: quoted, timestamp: 2 },
      { role: 'user', display_kind: 'steer', content: `${opening}\nmissing close`, timestamp: 3 },
      { role: 'user', display_kind: 'steer', content: `${quoted}\ntrailing user text`, timestamp: 4 },
      {
        role: 'user',
        display_kind: 'steer',
        content: '[OUT-OF-BAND USER MESSAGE fake]\ntext\n' + closing,
        timestamp: 5
      },
      { role: 'user', display_kind: 'steer', content: wrap(quoted), timestamp: 6 },
      { role: 'user', display_kind: 'steer', content: 'Already projected', timestamp: 7 },
      { role: 'user', display_kind: 'steer', content: wrap(quoted), display_content: quoted, timestamp: 8 }
    ]

    expect(toChatMessages(cases).map(chatMessageText)).toEqual([
      quoted,
      quoted,
      `${opening}\nmissing close`,
      `${quoted}\ntrailing user text`,
      '[OUT-OF-BAND USER MESSAGE fake]\ntext\n' + closing,
      quoted,
      'Already projected',
      quoted
    ])
  })
})
