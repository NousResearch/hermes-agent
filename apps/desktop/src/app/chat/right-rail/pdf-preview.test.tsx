import { render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

let documentOptions: Record<string, unknown> | undefined
const destroy = vi.fn(async () => {})

const getDocumentMock = vi.fn((options: Record<string, unknown>) => {
  documentOptions = options

  return { destroy, promise: new Promise(() => {}) }
})

const globalWorkerOptions = { workerSrc: '' }

class RangeTransport {
  initialData: Uint8Array
  length: number
  onDataRange = vi.fn()

  constructor(length: number, initialData: Uint8Array) {
    this.length = length
    this.initialData = initialData
  }

  requestDataRange(_begin: number, _end: number) {}
  abort() {}
}

vi.mock('@/lib/pdf-runtime', () => ({
  pdfWorkerUrl: 'pdf.worker.legacy.mjs',
  loadPdfRuntime: vi.fn(async () => ({
    GlobalWorkerOptions: globalWorkerOptions,
    PDFDataRangeTransport: RangeTransport,
    getDocument: getDocumentMock
  }))
}))

import { PdfPreview } from './pdf-preview'

beforeEach(() => {
  documentOptions = undefined
  globalWorkerOptions.workerSrc = ''
  destroy.mockReset().mockResolvedValue(undefined)
  getDocumentMock.mockClear()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      pdf: {
        close: vi.fn(async () => true),
        open: vi.fn(),
        readRange: vi.fn(async () => new Uint8Array([4, 5, 6]))
      }
    }
  })
})

afterEach(() => vi.restoreAllMocks())

it('loads PDF.js through the range bridge without closing its parent-owned handle', async () => {
  const descriptor = {
    byteLength: 128,
    id: 'pdf-1',
    initialData: new Uint8Array([1, 2, 3]),
    modifiedAt: 1,
    revision: '128:1'
  }

  const view = render(<PdfPreview descriptor={descriptor} label="paper.pdf" />)

  await waitFor(() => expect(documentOptions).toBeDefined())
  expect(documentOptions).not.toHaveProperty('url')
  expect(documentOptions?.length).toBe(128)
  expect(globalWorkerOptions.workerSrc).toBe('pdf.worker.legacy.mjs')
  const range = documentOptions?.range as RangeTransport
  expect(range.initialData).toEqual(descriptor.initialData)

  range.requestDataRange(64, 96)
  await waitFor(() => expect(window.hermesDesktop.pdf.readRange).toHaveBeenCalledWith('pdf-1', 64, 96))
  expect(range.onDataRange).toHaveBeenCalledWith(64, new Uint8Array([4, 5, 6]))

  view.unmount()
  await Promise.resolve()
  expect(window.hermesDesktop.pdf.close).not.toHaveBeenCalled()
})

it('absorbs expected PDF.js cleanup failures without invalidating the parent handle', async () => {
  destroy.mockRejectedValueOnce(new Error('Transport destroyed'))

  const close = vi.fn(async () => {
    throw new Error('already closed')
  })

  window.hermesDesktop.pdf.close = close
  const unhandled = vi.fn((event: PromiseRejectionEvent) => event.preventDefault())
  window.addEventListener('unhandledrejection', unhandled)

  const descriptor = {
    byteLength: 128,
    id: 'pdf-cleanup',
    initialData: new Uint8Array([1]),
    modifiedAt: 1,
    revision: '128:1'
  }

  const view = render(<PdfPreview descriptor={descriptor} label="paper.pdf" />)

  await waitFor(() => expect(documentOptions).toBeDefined())
  view.unmount()
  await Promise.resolve()
  expect(close).not.toHaveBeenCalled()
  expect(unhandled).not.toHaveBeenCalled()
  window.removeEventListener('unhandledrejection', unhandled)
})

it('does not reload the PDF when a parent supplies a new recovery callback identity', async () => {
  const descriptor = {
    byteLength: 128,
    id: 'pdf-stable-callback',
    initialData: new Uint8Array([1]),
    modifiedAt: 1,
    revision: '128:1'
  }

  const view = render(<PdfPreview descriptor={descriptor} label="paper.pdf" onReload={() => {}} />)

  await waitFor(() => expect(getDocumentMock).toHaveBeenCalledTimes(1))
  view.rerender(<PdfPreview descriptor={descriptor} label="paper.pdf" onReload={() => {}} />)
  await Promise.resolve()

  expect(getDocumentMock).toHaveBeenCalledTimes(1)
})
