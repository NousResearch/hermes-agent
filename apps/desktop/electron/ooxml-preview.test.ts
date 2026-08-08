import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import zlib from 'node:zlib'

import { test } from 'vitest'

import { extractOoxmlPreviewText, extractOoxmlPreviewTextFromBuffer } from './ooxml-preview'

// Matches MAX_OOXML_SOURCE_BYTES in ooxml-preview.ts — kept as a literal so
// this test still exercises the cap if that module fails to export it.
const OVER_SOURCE_CAP_BYTES = 20 * 1024 * 1024 + 1

// Minimal ZIP writer (store + deflate) — just enough to exercise the reader
// in ooxml-preview.ts against a real archive, without a third-party ZIP lib.
function buildZip(parts: Record<string, string>, options: { store?: boolean } = {}): Buffer {
  const localChunks: Buffer[] = []
  const centralChunks: Buffer[] = []
  let offset = 0

  for (const [name, content] of Object.entries(parts)) {
    const nameBuf = Buffer.from(name, 'utf8')
    const contentBuf = Buffer.from(content, 'utf8')
    const compressed = options.store ? contentBuf : zlib.deflateRawSync(contentBuf)
    const method = options.store ? 0 : 8

    const localHeader = Buffer.alloc(30)
    localHeader.writeUInt32LE(0x04034b50, 0)
    localHeader.writeUInt16LE(20, 4)
    localHeader.writeUInt16LE(0, 6)
    localHeader.writeUInt16LE(method, 8)
    localHeader.writeUInt16LE(0, 10)
    localHeader.writeUInt16LE(0, 12)
    localHeader.writeUInt32LE(0, 14)
    localHeader.writeUInt32LE(compressed.length, 18)
    localHeader.writeUInt32LE(contentBuf.length, 22)
    localHeader.writeUInt16LE(nameBuf.length, 26)
    localHeader.writeUInt16LE(0, 28)

    const localHeaderOffset = offset
    localChunks.push(localHeader, nameBuf, compressed)
    offset += localHeader.length + nameBuf.length + compressed.length

    const centralHeader = Buffer.alloc(46)
    centralHeader.writeUInt32LE(0x02014b50, 0)
    centralHeader.writeUInt16LE(20, 4)
    centralHeader.writeUInt16LE(20, 6)
    centralHeader.writeUInt16LE(0, 8)
    centralHeader.writeUInt16LE(method, 10)
    centralHeader.writeUInt16LE(0, 12)
    centralHeader.writeUInt16LE(0, 14)
    centralHeader.writeUInt32LE(0, 16)
    centralHeader.writeUInt32LE(compressed.length, 20)
    centralHeader.writeUInt32LE(contentBuf.length, 24)
    centralHeader.writeUInt16LE(nameBuf.length, 28)
    centralHeader.writeUInt16LE(0, 30)
    centralHeader.writeUInt16LE(0, 32)
    centralHeader.writeUInt16LE(0, 34)
    centralHeader.writeUInt16LE(0, 36)
    centralHeader.writeUInt32LE(0, 38)
    centralHeader.writeUInt32LE(localHeaderOffset, 42)

    centralChunks.push(centralHeader, nameBuf)
  }

  const centralDirStart = offset
  const centralDir = Buffer.concat(centralChunks)
  offset += centralDir.length

  const eocd = Buffer.alloc(22)
  eocd.writeUInt32LE(0x06054b50, 0)
  eocd.writeUInt16LE(0, 4)
  eocd.writeUInt16LE(0, 6)
  eocd.writeUInt16LE(Object.keys(parts).length, 8)
  eocd.writeUInt16LE(Object.keys(parts).length, 10)
  eocd.writeUInt32LE(centralDir.length, 12)
  eocd.writeUInt32LE(centralDirStart, 16)
  eocd.writeUInt16LE(0, 20)

  return Buffer.concat([...localChunks, centralDir, eocd])
}

const DOCX_DOCUMENT_XML = `<?xml version="1.0"?>
<w:document><w:body>
<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> world</w:t></w:r></w:p>
<w:p><w:r><w:t>Second paragraph &amp; more</w:t></w:r></w:p>
</w:body></w:document>`

const PPTX_SLIDE_1 = `<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide one title</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>`
const PPTX_SLIDE_2 = `<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide two</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>`

const XLSX_SHARED_STRINGS = `<sst><si><t>Name</t></si><si><t>Score</t></si><si><t>Ada</t></si></sst>`

const XLSX_SHEET_1 = `<worksheet><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>42</v></c></row>
</sheetData></worksheet>`

test('extracts paragraph text from a docx, joined with entity decoding', () => {
  const zip = buildZip({ 'word/document.xml': DOCX_DOCUMENT_XML })
  const text = extractOoxmlPreviewTextFromBuffer(zip, '.docx')

  assert.equal(text, 'Hello world\nSecond paragraph & more')
})

test('extracts slide text from a pptx in numeric slide order', () => {
  const zip = buildZip({
    'ppt/slides/slide1.xml': PPTX_SLIDE_1,
    'ppt/slides/slide2.xml': PPTX_SLIDE_2
  })

  const text = extractOoxmlPreviewTextFromBuffer(zip, '.pptx')

  assert.equal(text, '--- Slide 1 ---\nSlide one title\n\n--- Slide 2 ---\nSlide two')
})

test('extracts a worksheet as tab-separated rows, resolving shared strings', () => {
  const zip = buildZip({
    'xl/sharedStrings.xml': XLSX_SHARED_STRINGS,
    'xl/worksheets/sheet1.xml': XLSX_SHEET_1
  })

  const text = extractOoxmlPreviewTextFromBuffer(zip, '.xlsx')

  assert.equal(text, '--- Sheet 1 ---\nName\tScore\nAda\t42')
})

test('works with stored (uncompressed) ZIP entries too', () => {
  const zip = buildZip({ 'word/document.xml': DOCX_DOCUMENT_XML }, { store: true })
  const text = extractOoxmlPreviewTextFromBuffer(zip, '.docx')

  assert.equal(text, 'Hello world\nSecond paragraph & more')
})

test('returns null for an unsupported extension', () => {
  const zip = buildZip({ 'word/document.xml': DOCX_DOCUMENT_XML })

  assert.equal(extractOoxmlPreviewTextFromBuffer(zip, '.doc'), null)
})

test('returns null (not a throw) for a docx missing document.xml', () => {
  const zip = buildZip({ 'word/other.xml': DOCX_DOCUMENT_XML })

  assert.equal(extractOoxmlPreviewTextFromBuffer(zip, '.docx'), null)
})

test('throws on a buffer that is not a ZIP archive at all', () => {
  assert.throws(() => extractOoxmlPreviewTextFromBuffer(Buffer.from('not a zip file'), '.docx'))
})

test('skips reading the file once the known source size exceeds the cap', async () => {
  const zip = buildZip({ 'word/document.xml': DOCX_DOCUMENT_XML })
  const tmpFile = path.join(os.tmpdir(), `ooxml-preview-source-cap-${process.pid}.docx`)
  fs.writeFileSync(tmpFile, zip)

  try {
    const withinCap = await extractOoxmlPreviewText(tmpFile, '.docx', zip.length)

    assert.equal(withinCap, 'Hello world\nSecond paragraph & more')

    // The file on disk is tiny — only the *declared* size claims it's huge.
    // A correct implementation must reject on that declared size alone,
    // without ever reading (let alone successfully extracting from) the file.
    const overCap = await extractOoxmlPreviewText(tmpFile, '.docx', OVER_SOURCE_CAP_BYTES)

    assert.equal(overCap, null)
  } finally {
    fs.rmSync(tmpFile, { force: true })
  }
})
