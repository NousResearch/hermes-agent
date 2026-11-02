import { describe, expect, it } from 'vitest'

import { officeSourceName } from './office-source'

describe('office source name', () => {
  it('routes by path when the tab label is not the filename', () => {
    expect(officeSourceName('/tmp/notes.odt', 'ODT')).toMatch(/\.odt$/i)
    expect(officeSourceName('/tmp/book.ods', 'ODS')).toMatch(/\.ods$/i)
    expect(officeSourceName('/tmp/people.csv', 'CSV')).toMatch(/\.csv$/i)
    expect(/\.odt$/i.test('ODT')).toBe(false)
  })

  it('falls back to the label when no path is available', () => {
    expect(officeSourceName(undefined, 'notes.odt')).toBe('notes.odt')
    expect(officeSourceName('  ', 'book.xlsx')).toBe('book.xlsx')
  })
})
