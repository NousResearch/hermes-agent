import { deflateRawSync } from 'node:zlib'

import { describe, expect, it } from 'vitest'

import { parseOfficePreview } from './ooxml-preview'

function crc32(data: Uint8Array): number {
  let crc = ~0

  for (const byte of data) {
    crc ^= byte

    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
    }
  }

  return ~crc >>> 0
}

function u16(value: number) {
  const bytes = new Uint8Array(2)
  new DataView(bytes.buffer).setUint16(0, value, true)

  return bytes
}

function u32(value: number) {
  const bytes = new Uint8Array(4)
  new DataView(bytes.buffer).setUint32(0, value, true)

  return bytes
}

function concat(parts: Uint8Array[]) {
  const total = parts.reduce((sum, part) => sum + part.length, 0)
  const out = new Uint8Array(total)
  let offset = 0

  for (const part of parts) {
    out.set(part, offset)
    offset += part.length
  }

  return out
}

function zipFiles(files: Record<string, string>, method: 'store' | 'deflate' = 'store') {
  const encoder = new TextEncoder()
  const locals: Uint8Array[] = []
  const centrals: Uint8Array[] = []
  let offset = 0

  for (const [name, text] of Object.entries(files)) {
    const nameBytes = encoder.encode(name)
    const raw = encoder.encode(text)
    const payload = method === 'deflate' ? new Uint8Array(deflateRawSync(raw)) : raw
    const crc = crc32(raw)
    const zipMethod = method === 'deflate' ? 8 : 0

    const local = concat([
      encoder.encode('PK\u0003\u0004'),
      u16(20),
      u16(0),
      u16(zipMethod),
      u16(0),
      u16(0),
      u32(crc),
      u32(payload.length),
      u32(raw.length),
      u16(nameBytes.length),
      u16(0),
      nameBytes,
      payload
    ])

    const central = concat([
      encoder.encode('PK\u0001\u0002'),
      u16(20),
      u16(20),
      u16(0),
      u16(zipMethod),
      u16(0),
      u16(0),
      u32(crc),
      u32(payload.length),
      u32(raw.length),
      u16(nameBytes.length),
      u16(0),
      u16(0),
      u16(0),
      u16(0),
      u32(0),
      u32(offset),
      nameBytes
    ])

    locals.push(local)
    centrals.push(central)
    offset += local.length
  }

  const localBlob = concat(locals)
  const centralBlob = concat(centrals)

  const eocd = concat([
    encoder.encode('PK\u0005\u0006'),
    u16(0),
    u16(0),
    u16(locals.length),
    u16(locals.length),
    u32(centralBlob.length),
    u32(localBlob.length),
    u16(0)
  ])

  return concat([localBlob, centralBlob, eocd])
}

function zipDeflatedWithReportedSize(name: string, raw: Uint8Array, reportedUncompressed: number) {
  const nameBytes = new TextEncoder().encode(name)
  const payload = new Uint8Array(deflateRawSync(raw))
  const crc = crc32(raw)

  const local = concat([
    new Uint8Array([0x50, 0x4b, 0x03, 0x04]),
    u16(20),
    u16(0),
    u16(8),
    u16(0),
    u16(0),
    u32(crc),
    u32(payload.length),
    u32(reportedUncompressed),
    u16(nameBytes.length),
    u16(0),
    nameBytes,
    payload
  ])

  const central = concat([
    new Uint8Array([0x50, 0x4b, 0x01, 0x02]),
    u16(20),
    u16(20),
    u16(0),
    u16(8),
    u16(0),
    u16(0),
    u32(crc),
    u32(payload.length),
    u32(reportedUncompressed),
    u16(nameBytes.length),
    u16(0),
    u16(0),
    u16(0),
    u16(0),
    u32(0),
    u32(0),
    nameBytes
  ])

  const eocd = concat([
    new Uint8Array([0x50, 0x4b, 0x05, 0x06]),
    u16(0),
    u16(0),
    u16(1),
    u16(1),
    u32(central.length),
    u32(local.length),
    u16(0)
  ])

  return concat([local, central, eocd])
}

const NS_MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
const NS_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
const NS_PKG_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
const NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
const NS_P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
const NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'

function xlsxFiles(extraSheets?: Record<string, string>) {
  return {
    'xl/workbook.xml': `<?xml version="1.0"?>
<workbook xmlns="${NS_MAIN}" xmlns:r="${NS_REL}">
  <sheets>
    <sheet name="Revenue" sheetId="1" r:id="rId1"/>
    <sheet name="Notes" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>`,
    'xl/_rels/workbook.xml.rels': `<?xml version="1.0"?>
<Relationships xmlns="${NS_PKG_REL}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>`,
    'xl/sharedStrings.xml': `<?xml version="1.0"?>
<sst xmlns="${NS_MAIN}" count="3" uniqueCount="3">
  <si><t>Name</t></si>
  <si><t>Ada</t></si>
  <si><r><rPr/><t xml:space="preserve">Q1 </t></r><r><t>total</t></r></si>
</sst>`,
    'xl/worksheets/sheet1.xml': `<?xml version="1.0"?>
<worksheet xmlns="${NS_MAIN}">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c><c r="C1"><f>B1+1</f><v>43</v></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="b"><v>1</v></c><c r="C2" t="e"><v>#DIV/0!</v></c></row>
    <row r="3"><c r="A3" t="str"><v>ok</v></c></row>
    <row r="4"><c r="A4" t="s"><v>2</v></c></row>
  </sheetData>
</worksheet>`,
    'xl/worksheets/sheet2.xml': `<?xml version="1.0"?>
<worksheet xmlns="${NS_MAIN}">
  <sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>hello</t></is></c></row>
  </sheetData>
</worksheet>`,
    ...extraSheets
  }
}

describe('parseOfficePreview xlsx', () => {
  it('renders shared strings, numbers, booleans, and sparse rows as a grid', async () => {
    const preview = await parseOfficePreview(zipFiles(xlsxFiles()), '.xlsx')

    expect(preview).toMatchObject({ kind: 'spreadsheet' })

    if (preview?.kind !== 'spreadsheet') {
      return
    }

    expect(preview.sheets.map(sheet => sheet.name)).toEqual(['Revenue', 'Notes'])
    expect(preview.sheets[0]?.rows.map(row => row.map(cell => cell.value))).toEqual([
      ['Name', '42', '43'],
      ['Ada', 'TRUE', '#DIV/0!'],
      ['ok', '', ''],
      ['Q1 total', '', '']
    ])
    expect(preview.sheets[0]?.rows[0]?.[2]).toMatchObject({ formula: 'B1+1', value: '43' })
    expect(preview.sheets[1]?.rows.map(row => row.map(cell => cell.value))).toEqual([['hello']])
  })

  it('inflates DEFLATE-compressed OOXML parts', async () => {
    const preview = await parseOfficePreview(zipFiles(xlsxFiles(), 'deflate'), '.xlsx')

    expect(preview?.kind).toBe('spreadsheet')

    if (preview?.kind === 'spreadsheet') {
      expect(preview.sheets[0]?.rows[0]?.map(cell => cell.value)).toEqual(['Name', '42', '43'])
    }
  })

  it('formats dates, fills, and font styles from styles.xml', async () => {
    const preview = await parseOfficePreview(
      zipFiles({
        ...xlsxFiles({
          'xl/worksheets/sheet1.xml': `<?xml version="1.0"?>
<worksheet xmlns="${NS_MAIN}">
  <sheetData>
    <row r="1">
      <c r="A1" s="1"><v>44927</v></c>
      <c r="B1" s="2"><v>0.15</v></c>
      <c r="C1" s="3" t="s"><v>1</v></c>
    </row>
  </sheetData>
</worksheet>`
        }),
        'xl/styles.xml': `<?xml version="1.0"?>
<styleSheet xmlns="${NS_MAIN}">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0%"/></numFmts>
  <fonts count="2">
    <font/>
    <font><b/><color rgb="FFFF0000"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/></patternFill></fill>
  </fills>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0"/>
    <xf numFmtId="14" fontId="0" fillId="0" applyNumberFormat="1"/>
    <xf numFmtId="164" fontId="0" fillId="2" applyNumberFormat="1" applyFill="1"/>
    <xf numFmtId="0" fontId="1" fillId="0" applyFont="1"/>
  </cellXfs>
</styleSheet>`
      }),
      '.xlsx'
    )

    expect(preview?.kind).toBe('spreadsheet')

    if (preview?.kind !== 'spreadsheet') {
      return
    }

    expect(preview.sheets[0]?.rows[0]?.[0]).toMatchObject({ value: '1/1/2023' })
    expect(preview.sheets[0]?.rows[0]?.[1]).toMatchObject({ fill: '#C6EFCE', value: '15%' })
    expect(preview.sheets[0]?.rows[0]?.[2]).toMatchObject({ bold: true, color: '#FF0000', value: 'Ada' })
  })

  it('evaluates formulas when the cached value is empty', async () => {
    const preview = await parseOfficePreview(
      zipFiles({
        'xl/workbook.xml': `<?xml version="1.0"?>
<workbook xmlns="${NS_MAIN}" xmlns:r="${NS_REL}">
  <sheets>
    <sheet name="Transactions" sheetId="1" r:id="rId1"/>
    <sheet name="Pivot" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>`,
        'xl/_rels/workbook.xml.rels': `<?xml version="1.0"?>
<Relationships xmlns="${NS_PKG_REL}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>`,
        'xl/worksheets/sheet1.xml': `<?xml version="1.0"?>
<worksheet xmlns="${NS_MAIN}">
  <sheetData>
    <row r="1"><c r="A1"><v>2</v></c><c r="B1"><v>3</v></c><c r="C1"><f>A1+B1</f><v/></c></row>
    <row r="2"><c r="D2"><f>IF(G2="",0,G2)-IF(F2="",0,F2)</f><v/></c><c r="F2"/><c r="G2" t="n"/><c r="H2"><v>10</v></c><c r="I2" t="inlineStr"><is><t>Cat</t></is></c><c r="K2" t="inlineStr"><is><t>Inflow</t></is></c></row>
    <row r="3"><c r="H3"><v>4</v></c><c r="I3" t="inlineStr"><is><t>Cat</t></is></c><c r="K3" t="inlineStr"><is><t>Outflow</t></is></c></row>
  </sheetData>
</worksheet>`,
        'xl/worksheets/sheet2.xml': `<?xml version="1.0"?>
<worksheet xmlns="${NS_MAIN}">
  <sheetData>
    <row r="6"><c r="A6" t="inlineStr"><is><t>Cat</t></is></c><c r="B6"><f>SUMIFS(Transactions!$H$2:$H$3,Transactions!$I$2:$I$3,$A6,Transactions!$K$2:$K$3,"Inflow")</f><v/></c></row>
  </sheetData>
</worksheet>`
      }),
      '.xlsx'
    )

    expect(preview?.kind).toBe('spreadsheet')

    if (preview?.kind !== 'spreadsheet') {
      return
    }

    const transactions = preview.sheets.find(sheet => sheet.name === 'Transactions')
    const pivot = preview.sheets.find(sheet => sheet.name === 'Pivot')

    expect(transactions?.rows[0]?.[2]).toMatchObject({ formula: 'A1+B1', value: '5' })
    expect(transactions?.rows[1]?.[3]).toMatchObject({ value: '0' })
    expect(pivot?.rows[5]?.[1]).toMatchObject({ value: '10' })
  })
})

describe('parseOfficePreview docx', () => {
  it('renders paragraphs, bold, and tables as HTML', async () => {
    const preview = await parseOfficePreview(
      zipFiles({
        'word/document.xml': `<?xml version="1.0"?>
<w:document xmlns:w="${NS_W}">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Title</w:t></w:r></w:p>
    <w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>Ada</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
  </w:body>
</w:document>`
      }),
      '.docx'
    )

    expect(preview?.kind).toBe('document')

    if (preview?.kind !== 'document') {
      return
    }

    expect(preview.blocks).toEqual([
      { heading: 1, runs: [{ text: 'Title' }], type: 'paragraph' },
      { runs: [{ text: 'Hello ' }, { bold: true, text: 'Ada' }], type: 'paragraph' },
      { rows: [['A', 'B']], type: 'table' }
    ])
  })

  it('escapes HTML injected into document text', async () => {
    const preview = await parseOfficePreview(
      zipFiles({
        'word/document.xml': `<?xml version="1.0"?>
<w:document xmlns:w="${NS_W}">
  <w:body>
    <w:p><w:r><w:t>&lt;img src=x onerror=alert(1)&gt;</w:t></w:r></w:p>
  </w:body>
</w:document>`
      }),
      '.docx'
    )

    expect(preview?.kind).toBe('document')

    if (preview?.kind === 'document') {
      expect(preview.blocks).toEqual([
        { runs: [{ text: '<img src=x onerror=alert(1)>' }], type: 'paragraph' }
      ])
    }
  })
})

describe('parseOfficePreview pptx', () => {
  it('renders each slide as HTML in numeric order', async () => {
    const preview = await parseOfficePreview(
      zipFiles({
        'ppt/slides/slide2.xml': `<?xml version="1.0"?>
<p:sld xmlns:p="${NS_P}" xmlns:a="${NS_A}"><p:cSld><p:spTree>
  <p:sp><p:txBody><a:p><a:r><a:t>Second</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>`,
        'ppt/slides/slide1.xml': `<?xml version="1.0"?>
<p:sld xmlns:p="${NS_P}" xmlns:a="${NS_A}"><p:cSld><p:spTree>
  <p:sp><p:txBody><a:p><a:r><a:t>First</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>`
      }),
      '.pptx'
    )

    expect(preview?.kind).toBe('slides')

    if (preview?.kind === 'slides') {
      expect(preview.slides.map(slide => slide.lines)).toEqual([['First'], ['Second']])
    }
  })
})

describe('parseOfficePreview guards', () => {
  it('returns null for non-ZIP bytes', async () => {
    await expect(parseOfficePreview(new TextEncoder().encode('not a zip'), '.xlsx')).resolves.toBeNull()
  })

  it('returns null for unsupported extensions', async () => {
    await expect(parseOfficePreview(zipFiles({ 'xl/workbook.xml': '<workbook/>' }), '.txt')).resolves.toBeNull()
  })

  it('rejects a deflate bomb whose true size exceeds the part cap', async () => {
    const raw = new Uint8Array(9 * 1024 * 1024)
    const archive = zipDeflatedWithReportedSize('xl/worksheets/sheet1.xml', raw, 100)

    await expect(parseOfficePreview(archive, '.xlsx')).resolves.toBeNull()
  })
})
