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
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="b"><v>1</v></c></row>
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
    expect(preview.sheets[0]?.rows).toEqual([
      ['Name', '42'],
      ['Ada', 'TRUE'],
      ['', ''],
      ['Q1 total', '']
    ])
    expect(preview.sheets[1]?.rows).toEqual([['hello']])
  })

  it('inflates DEFLATE-compressed OOXML parts', async () => {
    const preview = await parseOfficePreview(zipFiles(xlsxFiles(), 'deflate'), '.xlsx')

    expect(preview?.kind).toBe('spreadsheet')

    if (preview?.kind === 'spreadsheet') {
      expect(preview.sheets[0]?.rows[0]).toEqual(['Name', '42'])
    }
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

    expect(preview.html).toContain('<h1>')
    expect(preview.html).toContain('Title')
    expect(preview.html).toContain('<strong>Ada</strong>')
    expect(preview.html).toContain('<td>A</td>')
    expect(preview.html).toContain('<td>B</td>')
    expect(preview.html).not.toContain('<script')
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
      expect(preview.html).toContain('&lt;img src=x onerror=alert(1)&gt;')
      expect(preview.html).not.toContain('<img')
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
      expect(preview.slides.map(slide => slide.html)).toEqual([
        expect.stringContaining('First'),
        expect.stringContaining('Second')
      ])
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
})
