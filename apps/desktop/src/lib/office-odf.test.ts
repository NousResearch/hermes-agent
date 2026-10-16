import { strToU8, zipSync } from 'fflate'
import { describe, expect, it } from 'vitest'

import { odtPreviewDocument, parseOds } from './office-odf'
import { parseSpreadsheet } from './office-workbook'

const content = `<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
<office:body>
<office:text>
<text:h text:outline-level="1">Open notes</text:h>
<text:p>Hello from Writer</text:p>
</office:text>
<office:spreadsheet>
<table:table table:name="Probe">
<table:table-row>
<table:table-cell office:value-type="string"><text:p>Hermes ODS probe</text:p></table:table-cell>
<table:table-cell office:value-type="float" office:value="42"><text:p>42</text:p></table:table-cell>
<table:table-cell table:number-columns-repeated="1000"/>
</table:table-row>
</table:table>
</office:spreadsheet>
</office:body>
</office:document-content>`

describe('OpenDocument preview', () => {
  it('renders odt headings and paragraphs on the page frame', () => {
    const html = odtPreviewDocument(zipSync({ 'content.xml': strToU8(content) }))

    expect(html).toContain('<h1>Open notes</h1>')
    expect(html).toContain('Hello from Writer')
    expect(html).toContain('class="page"')
  })

  it('reads an ods sheet without expanding empty repeated columns', () => {
    const [sheet] = parseSpreadsheet(zipSync({ 'content.xml': strToU8(content) }), 'book.ods')

    expect(sheet.name).toBe('Probe')
    expect(sheet.rows).toEqual([['Hermes ODS probe', '42']])
  })

  it('keeps parseOds limited to spreadsheet tables', () => {
    expect(parseOds(zipSync({ 'content.xml': strToU8(content) }))).toHaveLength(1)
  })
})
