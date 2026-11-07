import { afterEach, describe, expect, it, vi } from 'vitest'

import { saveTranscript } from './save-transcript'

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('transcript file writer', () => {
  it('never saves an unfiltered older-backend response for a filtered transcript request', async () => {
    const save = vi.fn(async () => ({ saved: true, path: 'C:/sample/conversation.json' }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ content: '{}', filename: 'conversation.json', format: 'json' }))

    const options = { include_tools: false, include_reasoning: false, include_images: true, include_videos: true,
      include_files: true, include_links: true, include_asides: true, collapse_details: true }

    expect(await saveTranscript({ format: 'json', options, redact: false, requestGateway, sessionId: 'row-a', title: 'Save' }))
      .toEqual({ kind: 'unsupported' })
    expect(requestGateway).toHaveBeenCalledWith('session.export', { session_id: 'row-a', format: 'json', transcript_options: options })
    expect(save).not.toHaveBeenCalled()
  })

  it('does not mistake an unconfirmed file write for a canceled picker', async () => {
    const save = vi.fn(async () => ({ saved: true }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ content: '{}', filename: 'sample.json', format: 'json' }))

    const outcome = await saveTranscript({
      format: 'json',
      redact: false,
      requestGateway,
      sessionId: 'sample-runtime',
      title: 'Save conversation'
    })

    expect(outcome).toEqual({ kind: 'error', error: expect.any(Error) })
  })

  it('refuses a backend response in a different format instead of relabeling it', async () => {
    const save = vi.fn(async () => ({ saved: true, path: 'C:/sample/wrong.html' }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ content: '{}', filename: 'wrong.json', format: 'json' }))

    const outcome = await saveTranscript({
      format: 'html',
      redact: true,
      requestGateway,
      sessionId: 'sample-runtime',
      title: 'Save conversation'
    })

    expect(outcome).toEqual({ kind: 'unsupported' })
    expect(save).not.toHaveBeenCalled()
  })

  it.each(['json', 'md', 'html'] as const)('writes %s content verbatim with the matching extension', async format => {
    const content = `sample ${format} transcript`
    const save = vi.fn(async () => ({ saved: true, path: `C:/sample/conversation.${format}` }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ content, filename: `conversation.${format}`, format }))

    expect(
      await saveTranscript({ format, redact: false, requestGateway, sessionId: 'sample-runtime', title: 'Save' })
    ).toEqual({ kind: 'saved', path: `C:/sample/conversation.${format}` })
    expect(requestGateway).toHaveBeenCalledWith('session.export', { session_id: 'sample-runtime', format })
    expect(save).toHaveBeenCalledWith(
      expect.objectContaining({ content, filters: [expect.objectContaining({ extensions: [format] })] })
    )
  })

  it('keeps an explicitly requested filename', async () => {
    const save = vi.fn(async () => ({ saved: true, path: 'C:/sample/notes.md' }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ content: '# sample', filename: 'backend.md', format: 'md' }))
    await saveTranscript({
      filename: 'notes.md',
      format: 'md',
      redact: true,
      requestGateway,
      sessionId: 'runtime',
      title: 'Save'
    })
    expect(save).toHaveBeenCalledWith(expect.objectContaining({ defaultPath: 'notes.md' }))
  })

  it('reports an explicitly canceled picker without claiming a save', async () => {
    const save = vi.fn(async () => ({ saved: false, canceled: true }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ content: '{}', filename: 'sample.json', format: 'json' }))
    expect(
      await saveTranscript({ format: 'json', redact: false, requestGateway, sessionId: 'runtime', title: 'Save' })
    ).toEqual({ kind: 'canceled' })
  })

  it('reports a missing client file writer honestly', async () => {
    const requestGateway = vi.fn(async () => ({ content: '{}', filename: 'sample.json', format: 'json' }))
    expect(
      await saveTranscript({ format: 'json', redact: false, requestGateway, sessionId: 'runtime', title: 'Save' })
    ).toEqual({ kind: 'unsupported' })
  })

  it('preserves a file write error instead of announcing success', async () => {
    const failure = new Error('sample write refused')

    const save = vi.fn(async () => {
      throw failure
    })

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ content: '{}', filename: 'sample.json', format: 'json' }))
    expect(
      await saveTranscript({ format: 'json', redact: false, requestGateway, sessionId: 'runtime', title: 'Save' })
    ).toEqual({ kind: 'error', error: failure })
  })

  it('never writes a response without transcript content', async () => {
    const save = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: save } })
    const requestGateway = vi.fn(async () => ({ filename: 'sample.json', format: 'json' }))
    expect(
      await saveTranscript({ format: 'json', redact: false, requestGateway, sessionId: 'runtime', title: 'Save' })
    ).toEqual({ kind: 'unsupported' })
    expect(save).not.toHaveBeenCalled()
  })
})
