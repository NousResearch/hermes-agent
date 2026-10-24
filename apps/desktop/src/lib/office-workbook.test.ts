import { strToU8, zipSync } from 'fflate'
import { describe, expect, it } from 'vitest'

import { parseSpreadsheet } from './office-workbook'
import { parseBiff8 } from './office-xls'
import { parseXlsx } from './office-xlsx'

function u16(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff]
}

function record(code: number, body: number[]): Uint8Array {
  return Uint8Array.from([code & 0xff, (code >> 8) & 0xff, body.length & 0xff, (body.length >> 8) & 0xff, ...body])
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0))
  let offset = 0

  for (const part of parts) {
    out.set(part, offset)
    offset += part.length
  }

  return out
}

function xlsxFixture(): Uint8Array {
  const contentTypes = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>`

  const rels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>`

  const workbookRels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>`

  const workbook = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Probe" sheetId="1" r:id="rId1"/></sheets>
</workbook>`

  const shared = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="1" uniqueCount="1"><si><t>Hermes XLSX probe</t></si></sst>`

  const sheet = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<cols><col min="1" max="1" width="20"/></cols>
<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row></sheetData>
</worksheet>`

  return zipSync({
    '[Content_Types].xml': strToU8(contentTypes),
    '_rels/.rels': strToU8(rels),
    'xl/workbook.xml': strToU8(workbook),
    'xl/_rels/workbook.xml.rels': strToU8(workbookRels),
    'xl/sharedStrings.xml': strToU8(shared),
    'xl/worksheets/sheet1.xml': strToU8(sheet)
  })
}

describe('spreadsheet preview', () => {
  it('reads an xlsx shared string and a number', () => {
    const [sheet] = parseXlsx(xlsxFixture())

    expect(sheet.name).toBe('Probe')
    expect(sheet.rows).toEqual([['Hermes XLSX probe', '42']])
    expect(sheet.widths?.[0]).toBe(160)
  })

  it('renders an Excel date serial instead of the raw number', () => {
    const styles = `<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs>
</styleSheet>`

    const sheet = `<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData><row r="1"><c r="A1" s="1"><v>44927</v></c></row></sheetData>
</worksheet>`

    const workbook = `<?xml version="1.0"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Dates" sheetId="1" r:id="rId1"/></sheets></workbook>`

    const bytes = zipSync({
      'xl/workbook.xml': strToU8(workbook),
      'xl/_rels/workbook.xml.rels': strToU8(
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
      ),
      'xl/styles.xml': strToU8(styles),
      'xl/worksheets/sheet1.xml': strToU8(sheet)
    })

    expect(parseXlsx(bytes)[0].rows).toEqual([['2023-01-01']])
  })

  it('keeps a quoted unit suffix as a number instead of a 1900 date', () => {
    const styles = `<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="#,##0&quot; m&quot;"/></numFmts>
<cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="164"/></cellXfs>
</styleSheet>`

    const sheet = `<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData><row r="1"><c r="A1" s="1"><v>42</v></c></row></sheetData>
</worksheet>`

    const workbook = `<?xml version="1.0"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Units" sheetId="1" r:id="rId1"/></sheets></workbook>`

    const bytes = zipSync({
      'xl/workbook.xml': strToU8(workbook),
      'xl/_rels/workbook.xml.rels': strToU8(
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
      ),
      'xl/styles.xml': strToU8(styles),
      'xl/worksheets/sheet1.xml': strToU8(sheet)
    })

    expect(parseXlsx(bytes)[0].rows).toEqual([['42']])
  })

  it('reads a BIFF8 label and number without treating them as binary text', () => {
    const name = [5, 0, ...'Probe'.split('').map(char => char.charCodeAt(0))]

    const sst = [
      1, 0, 0, 0,
      1, 0, 0, 0,
      2, 0, 0,
      ...'Hi'.split('').map(char => char.charCodeAt(0))
    ]

    const label = [...u16(0), ...u16(0), ...u16(0), 0, 0, 0, 0]
    const number = [...u16(1), ...u16(1), ...u16(0), 0, 0, 0, 0, 0, 0, 69, 64]

    const stream = concat([
      record(0x0809, [...u16(0x0600), ...u16(0x0005), 0, 0, 0, 0]),
      record(0x0085, [0, 0, 0, 0, 0, 0, ...name]),
      record(0x00fc, sst),
      record(0x0809, [...u16(0x0600), ...u16(0x0010), 0, 0, 0, 0]),
      record(0x00fd, label),
      record(0x0203, number)
    ])

    expect(parseBiff8(stream)).toEqual([
      { name: 'Probe', rows: [['Hi', ''], ['', '42']], truncated: false }
    ])
  })

  it('rejects a zip that is not a workbook', () => {
    expect(() => parseSpreadsheet(zipSync({ 'hello.txt': strToU8('no') }))).toThrow('Could not read this spreadsheet')
  })
})
