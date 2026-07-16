import { afterEach, expect, it } from 'vitest'

import { loadPdfRuntime, pdfWorkerUrl } from './pdf-runtime'

const mapPrototype = Map.prototype as Map<unknown, unknown> & {
  getOrInsertComputed?: (key: unknown, callback: (key: unknown) => unknown) => unknown
}

const originalGetOrInsertComputed = Object.getOwnPropertyDescriptor(mapPrototype, 'getOrInsertComputed')

afterEach(() => {
  if (originalGetOrInsertComputed) {
    Object.defineProperty(mapPrototype, 'getOrInsertComputed', originalGetOrInsertComputed)
  } else {
    delete mapPrototype.getOrInsertComputed
  }
})

function minimalPdf() {
  const encoder = new TextEncoder()
  const stream = 'BT /F1 12 Tf 72 72 Td (Hermes PDF) Tj ET'

  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>',
    `<< /Length ${encoder.encode(stream).length} >>\nstream\n${stream}\nendstream`,
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'
  ]

  let pdf = '%PDF-1.4\n'
  const offsets = [0]

  objects.forEach((object, index) => {
    offsets.push(encoder.encode(pdf).length)
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`
  })
  const xref = encoder.encode(pdf).length

  const rows = offsets
    .slice(1)
    .map(offset => `${String(offset).padStart(10, '0')} 00000 n `)
    .join('\n')

  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${rows}\ntrailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`

  return encoder.encode(pdf)
}

it('loads the compatibility runtime and parses a PDF without native Map emplace methods', async () => {
  delete mapPrototype.getOrInsertComputed
  const runtime = await loadPdfRuntime()

  expect(typeof mapPrototype.getOrInsertComputed).toBe('function')
  expect(pdfWorkerUrl).toContain('pdf.worker.min')
  const document = await runtime.getDocument({ data: minimalPdf() }).promise
  const page = await document.getPage(1)
  const content = await page.getTextContent()

  expect(document.numPages).toBe(1)
  expect(content.items.map(item => ('str' in item ? item.str : '')).join('')).toBe('Hermes PDF')
  await document.destroy()
})
