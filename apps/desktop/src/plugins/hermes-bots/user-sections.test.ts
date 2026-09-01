import { describe, expect, it } from 'vitest'

import {
  botDragPayload,
  groupRowsBySection,
  normalizeBotSections,
  readBotDragPayload,
  UNASSIGNED_SECTION_KEY
} from './user-sections'

const bot = (name: string) => ({ name }) as never

describe('user sections model', () => {
  it('normalizes: drops blanks and duplicates, defaults a name, keeps only an explicit icon=false', () => {
    const out = normalizeBotSections([
      { id: 'a', name: 'Clients' },
      { id: 'a', name: 'dupe' },
      { id: '', name: 'blank' },
      { id: 'b', name: '   ' },
      { id: 'c', name: 'Bare', icon: false },
      { id: 'd', name: 'On', icon: true },
      null,
      'junk'
    ])

    expect(out).toEqual([
      { id: 'a', name: 'Clients' },
      { id: 'b', name: 'Section' },
      { id: 'c', name: 'Bare', icon: false },
      { id: 'd', name: 'On' }
    ])
  })

  it('groups every row exactly once, unknown sections fall to Unassigned, Unassigned is last', () => {
    const rows = [
      { bot: bot('nanox'), kind: 'bot' },
      { bot: bot('scout'), kind: 'bot' },
      { bot: bot('ghost'), kind: 'bot' },
      { kind: 'group', name: 'Room' }
    ] as never[]

    const meta = {
      nanox: { sectionId: 'sec-clients' },
      scout: { sectionId: 'sec-workforce' },
      ghost: { sectionId: 'sec-deleted' }
    } as never

    const blocks = groupRowsBySection(rows, [{ id: 'sec-clients', name: 'Clients' }, { id: 'sec-workforce', name: 'Workforce' }], meta)

    expect(blocks.map(b => [b.key, b.rows.length])).toEqual([
      ['section:sec-clients', 1],
      ['section:sec-workforce', 1],
      [UNASSIGNED_SECTION_KEY, 2]
    ])
    expect(blocks.flatMap(b => b.rows)).toHaveLength(rows.length)
  })

  it('drag payload round-trips and a foreign drop yields no keys', () => {
    expect(readBotDragPayload(botDragPayload(['a', 'b']))).toEqual(['a', 'b'])
    expect(readBotDragPayload('not json')).toEqual([])
    expect(readBotDragPayload(JSON.stringify({ nope: 1 }))).toEqual([])
    expect(readBotDragPayload(JSON.stringify(['ok', 3, '', null]))).toEqual(['ok'])
  })
})
