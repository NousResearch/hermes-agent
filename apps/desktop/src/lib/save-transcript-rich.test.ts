import { strFromU8, unzipSync } from 'fflate'
import { afterEach, expect, it, vi } from 'vitest'

import { saveTranscript } from './save-transcript'
import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'

const doc: TranscriptDocument = {
  version: 1,
  session_id: 'row-a',
  title: 'Sample conversation',
  exported_at: '2026-11-03T18:00:00Z',
  options: { ...DEFAULT_TRANSCRIPT_OPTIONS },
  warnings: [],
  messages: [{ role: 'user', speaker: 'User', text: 'An actual exported message.' }]
}

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

function setup(binary = true, document = doc) {
  const save = vi.fn(async (_request: { content: string }) => ({ saved: true, path: 'C:/sample/chosen.docx' }))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      saveTextFileAs: save,
      ...(binary ? { saveTextFileEncodings: ['base64'] } : {})
    }
  })

  const requestGateway = vi.fn(async () => ({
    content: JSON.stringify(document),
    format: 'json',
    filename: 'hermes_session_row-a.json',
    transcript_version: 1
  }))

  return { save, requestGateway }
}

it('requests a filtered owning-session snapshot and saves a real Word package with the right extension', async () => {
  const { save, requestGateway } = setup()
  expect(
    await saveTranscript({
      format: 'docx',
      options: doc.options,
      redact: false,
      requestGateway,
      sessionId: 'row-a',
      title: 'Export'
    })
  ).toEqual({ kind: 'saved', path: 'C:/sample/chosen.docx' })
  expect(requestGateway).toHaveBeenCalledWith('session.export', {
    session_id: 'row-a',
    format: 'json',
    transcript_options: doc.options
  })

  const request = save.mock.calls[0][0] as {
    content: string
    encoding: string
    defaultPath: string
    filters: Array<{ extensions: string[] }>
  }

  expect(request.encoding).toBe('base64')
  expect(request.defaultPath).toBe('hermes_session_row-a.docx')
  expect(request.filters[0].extensions).toEqual(['docx'])
  const entries = unzipSync(Uint8Array.from(atob(request.content), c => c.charCodeAt(0)))
  expect(strFromU8(entries['word/document.xml'])).toContain('An actual exported message.')
})
it('refuses binary output on an older native bridge instead of saving base64 as text', async () => {
  const { save, requestGateway } = setup(false)
  expect(
    await saveTranscript({
      format: 'docx',
      options: doc.options,
      redact: false,
      requestGateway,
      sessionId: 'row-a',
      title: 'Export'
    })
  ).toEqual({ kind: 'unsupported' })
  expect(save).not.toHaveBeenCalled()
})

it('refuses a backend response whose inclusion choices differ from the user request', async () => {
  const { save, requestGateway } = setup(true, { ...doc, options: { ...doc.options, include_reasoning: true } })
  expect(
    await saveTranscript({
      format: 'json',
      options: doc.options,
      redact: false,
      requestGateway,
      sessionId: 'row-a',
      title: 'Export'
    })
  ).toEqual({ kind: 'unsupported' })
  expect(save).not.toHaveBeenCalled()
})
