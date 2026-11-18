import { describe, expect, it } from 'vitest'

import designDoc from '../../DESIGN.md?raw'

import { inlineCode, markdownSection, tableCells } from './markdown-section'

describe('markdownSection', () => {
  const doc = [
    '## Parent',
    '### Gaps',
    'Some intro',
    'continues here.',
    '',
    '| # | Role |',
    '|---|---|',
    '| U1 | a `x \\| y` cell |',
    'After the table.',
    '### Next',
    'not included'
  ].join('\n')

  it('returns paragraphs and tables up to the next heading of the same level', () => {
    expect(markdownSection(doc, 'Gaps')).toEqual([
      { kind: 'paragraph', text: 'Some intro continues here.' },
      { kind: 'table', table: { head: ['#', 'Role'], rows: [['U1', 'a `x | y` cell']] } },
      { kind: 'paragraph', text: 'After the table.' }
    ])
  })

  it('is null for a heading the document does not have', () => {
    expect(markdownSection(doc, 'Missing')).toBeNull()
  })

  it('finds DESIGN.md "Still to unify" with its table, as the design system page shows it', () => {
    const blocks = markdownSection(designDoc, 'Still to unify')
    const table = blocks?.find(block => block.kind === 'table')

    expect(table?.kind === 'table' && table.table.rows.length).toBeGreaterThan(0)
    expect(table?.kind === 'table' && table.table.rows.every(row => row.length === table.table.head.length)).toBe(true)
  })
})

describe('tableCells and inlineCode', () => {
  it('splits on unescaped pipes only', () => {
    expect(tableCells('| a | b \\| c |')).toEqual(['a', 'b | c'])
  })

  it('splits code spans out of a line', () => {
    expect(inlineCode('ink 5 `--portal-raw-ink-5` here')).toEqual([
      { code: false, text: 'ink 5 ' },
      { code: true, text: '--portal-raw-ink-5' },
      { code: false, text: ' here' }
    ])
  })
})
