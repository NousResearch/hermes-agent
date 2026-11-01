import { strToU8, zipSync } from 'fflate'
import { describe, expect, it } from 'vitest'

import { parseCsv } from './office-csv'
import { parsePptx } from './office-pptx'
import { parseSpreadsheet } from './office-workbook'

describe('csv and powerpoint preview', () => {
  it('reads a quoted csv into a grid', () => {
    const [sheet] = parseSpreadsheet(strToU8('Name,Amount\n"Ada, Lovelace",42\n'), 'people.csv')

    expect(sheet.name).toBe('people')
    expect(sheet.rows).toEqual([
      ['Name', 'Amount'],
      ['Ada, Lovelace', '42']
    ])
  })

  it('reads slide text from a pptx', () => {
    const presentation = `<?xml version="1.0"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
</p:presentation>`

    const rels = `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>`

    const slide = `<?xml version="1.0"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Quarterly review</a:t></a:r></a:p><a:p><a:r><a:t>Revenue is up</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>`

    const slides = parsePptx(
      zipSync({
        'ppt/presentation.xml': strToU8(presentation),
        'ppt/_rels/presentation.xml.rels': strToU8(rels),
        'ppt/slides/slide1.xml': strToU8(slide)
      })
    )

    expect(slides).toEqual([{ paragraphs: ['Quarterly review', 'Revenue is up'], images: [] }])
  })

  it('keeps a quoted comma inside one csv cell', () => {
    expect(parseCsv(strToU8('"a,b",c'), 't.csv')[0].rows).toEqual([['a,b', 'c']])
  })
})
