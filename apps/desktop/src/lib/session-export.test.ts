import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import * as exports from './session-export'

const mocks = vi.hoisted(() => ({
  download: vi.fn(),
  getAllSessionMessages: vi.fn(),
  request: vi.fn(),
  ownedRequest: vi.fn(),
  runtime: vi.fn(),
  save: vi.fn()
}))

vi.mock('@/hermes', () => ({ getAllSessionMessages: mocks.getAllSessionMessages }))
vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/i18n/runtime', () => ({
  runtimeTranslations: () => ({ promptNotices: { desktopCommands: { saveDialogTitle: 'Save conversation' } } })
}))
vi.mock('@/store/gateway', () => ({ activeGateway: () => ({ request: mocks.request }) }))
vi.mock('@/store/session-gone-latch', () => ({ ambientRequestFor: (gateway: { request: unknown }) => gateway.request }))
vi.mock('@/store/session-runtime-binding', () => ({ runtimeIdForStoredSession: mocks.runtime }))
vi.mock('@/store/session-states', () => ({ requestForOwnedSession: mocks.ownedRequest }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

beforeEach(() => {
  vi.clearAllMocks()
  mocks.runtime.mockReturnValue('runtime-row-a')
  mocks.ownedRequest.mockImplementation((_id, _ambient, method, params) => mocks.request(method, params))
  mocks.request.mockResolvedValue({ content: '# Sample conversation', filename: 'conversation.md', format: 'md' })
  mocks.save.mockResolvedValue({ path: 'C:/sample/conversation.md', saved: true })
  mocks.getAllSessionMessages.mockResolvedValue({ messages: [{ role: 'user', content: 'Sample message' }] })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveTextFileAs: mocks.save } })
})

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('conversation file export', () => {
  it('never substitutes a raw JSON download when transcript inclusion controls were requested', async () => {
    mocks.runtime.mockReturnValue(null)

    const options = { include_tools: false, include_reasoning: false, include_images: true, include_videos: true,
      include_files: true, include_links: true, include_asides: true, collapse_details: true }

    expect(await exports.exportSessionFile({ format: 'json', options, redact: false, sessionId: 'stored-row' }))
      .toEqual({ kind: 'needs-live-backend' })
    expect(mocks.getAllSessionMessages).not.toHaveBeenCalled()
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('keeps the existing plain JSON download when the captured row has no live runtime', async () => {
    mocks.runtime.mockReturnValue(null)
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    vi.stubGlobal(
      'URL',
      class extends URL {
        static createObjectURL = vi.fn(() => 'blob:sample')
        static revokeObjectURL = vi.fn()
      }
    )

    const outcome = await exports.exportSessionFile?.({
      format: 'json',
      profile: 'sample-bot',
      redact: false,
      sessionId: 'stored-row',
      title: 'Sample conversation'
    })

    expect(outcome).toEqual({ kind: 'json-download' })
    expect(mocks.getAllSessionMessages).toHaveBeenCalledWith('stored-row', 'sample-bot')
    expect(click).toHaveBeenCalledTimes(1)
    expect(mocks.request).not.toHaveBeenCalled()
    expect(mocks.save).not.toHaveBeenCalled()
    click.mockRestore()
    vi.unstubAllGlobals()
  })

  it('saves a redacted Markdown export through the captured row owner', async () => {
    const outcome = await exports.exportSessionFile?.({
      format: 'md',
      profile: 'sample-bot',
      redact: true,
      sessionId: 'row-a',
      title: 'Sample conversation'
    })

    expect(outcome).toEqual({ kind: 'saved', path: 'C:/sample/conversation.md' })
    expect(mocks.runtime).toHaveBeenCalledWith('row-a')
    expect(mocks.ownedRequest).toHaveBeenCalledWith('runtime-row-a', mocks.request, 'session.export', {
      session_id: 'runtime-row-a',
      format: 'md',
      redact: true
    })
    expect(mocks.save).toHaveBeenCalledWith({
      content: '# Sample conversation',
      defaultPath: 'conversation.md',
      filters: [{ name: 'Markdown', extensions: ['md'] }],
      title: 'Save conversation'
    })
    expect(mocks.getAllSessionMessages).not.toHaveBeenCalled()
  })

  it.each([
    { format: 'json', redact: true },
    { format: 'md', redact: false },
    { format: 'html', redact: false }
  ] as const)('never falls back to unredacted JSON for $format with redact=$redact', async ({ format, redact }) => {
    mocks.runtime.mockReturnValue(null)
    expect(await exports.exportSessionFile({ format, redact, sessionId: 'stored-row' })).toEqual({
      kind: 'needs-live-backend'
    })
    expect(mocks.getAllSessionMessages).not.toHaveBeenCalled()
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('refuses a redacted export on an older backend without downloading the original', async () => {
    mocks.request.mockRejectedValue(Object.assign(new Error('Method not found'), { code: -32601 }))
    expect(await exports.exportSessionFile({ format: 'json', redact: true, sessionId: 'row-a' })).toEqual({
      kind: 'needs-live-backend'
    })
    expect(mocks.getAllSessionMessages).not.toHaveBeenCalled()
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('preserves a backend or owner-routing failure instead of trying another conversation', async () => {
    const failure = Object.assign(new Error('sample owner unavailable'), { code: -32000 })
    mocks.ownedRequest.mockRejectedValue(failure)
    expect(await exports.exportSessionFile({ format: 'json', redact: false, sessionId: 'row-a' })).toEqual({
      kind: 'failed',
      error: failure
    })
    expect(mocks.getAllSessionMessages).not.toHaveBeenCalled()
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('refuses an empty conversation id without producing a false JSON download', async () => {
    expect(await exports.exportSessionFile({ format: 'json', redact: false, sessionId: '' })).toEqual({
      kind: 'failed',
      error: expect.any(Error)
    })
    expect(mocks.getAllSessionMessages).not.toHaveBeenCalled()
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it('leaves direct command-center JSON export working with its original filename', async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const createObjectURL = vi.fn(() => 'blob:sample')
    vi.stubGlobal(
      'URL',
      class extends URL {
        static createObjectURL = createObjectURL
        static revokeObjectURL = vi.fn()
      }
    )
    await exports.exportSession('stored-row', { profile: 'sample-bot', title: 'Sample conversation' })
    expect(mocks.getAllSessionMessages).toHaveBeenCalledWith('stored-row', 'sample-bot')
    expect((click.mock.instances[0] as HTMLAnchorElement).download).toBe('sample-conversation-stored-r.json')
    expect(createObjectURL).toHaveBeenCalledTimes(1)
    click.mockRestore()
    vi.unstubAllGlobals()
  })
})
