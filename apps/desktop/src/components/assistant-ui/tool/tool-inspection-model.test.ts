import { describe, expect, it } from 'vitest'

import { INSPECTION_PAGE_CHARS, inspectionSections, inspectionText, inspectionWindow } from './tool-inspection-model'

describe('available inspection payloads', () => {
  it('retains empty and scalar values without treating them as missing', () => {
    for (const [value, text] of [
      [undefined, undefined],
      ['', ''],
      [null, 'null'],
      [false, 'false'],
      [0, '0'],
      [[], '[]']
    ] as const) {
      expect(inspectionText(value)).toBe(text)
    }

    const result = { stdout: '', stderr: 'diagnostic', inline_diff: '+line' }
    const sections = inspectionSections({ type: 'tool-call', toolName: 'terminal', args: { command: 'echo' }, result })
    expect(sections.find(section => section.id === 'result')?.value).toBe(result)
    expect(sections.find(section => section.id === 'stdout')?.value).toBe('')
    expect(sections.find(section => section.id === 'stderr')?.value).toBe('diagnostic')
    expect(sections.find(section => section.id === 'diff')?.value).toBe('+line')
  })

  it('bounds very long single lines without splitting surrogate pairs', () => {
    const text = 'x'.repeat(INSPECTION_PAGE_CHARS - 1) + '😀' + 'y'.repeat(INSPECTION_PAGE_CHARS * 2)
    const first = inspectionWindow(text, 0)
    const second = inspectionWindow(text, first.end)
    expect(first.text.endsWith('😀')).toBe(true)
    expect(first.text.length).toBeLessThanOrEqual(INSPECTION_PAGE_CHARS + 2)
    expect(second.text).toBe('y'.repeat(INSPECTION_PAGE_CHARS))
    expect(inspectionWindow(text, INSPECTION_PAGE_CHARS).text.startsWith('😀')).toBe(true)
  })
})
