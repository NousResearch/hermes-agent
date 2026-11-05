import { beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { setRuntimeI18nLocale } from '@/i18n/runtime'

const notify = vi.hoisted(() => vi.fn())
const notifyError = vi.hoisted(() => vi.fn())
const writeClipboardText = vi.hoisted(() => vi.fn())

vi.mock('@/store/notifications', () => ({ notify, notifyError }))
vi.mock('@/lib/clipboard', () => ({ writeClipboardText }))
vi.mock('@/hermes', () => ({ getAllSessionMessages: vi.fn() }))

import { copyConciseTranscript } from './session-export'

const d = en.desktop

beforeEach(() => {
  vi.clearAllMocks()
  setRuntimeI18nLocale('en')
})

describe('copyConciseTranscript', () => {
  it('asks the owner for the concise Markdown and writes it to THIS client clipboard', async () => {
    writeClipboardText.mockResolvedValue(undefined)
    const request = vi.fn(async () => ({ content: '# Conversation: x\n\n## User\n\nhi\n' }))

    await copyConciseTranscript('runtime-1', request as never)

    expect(request).toHaveBeenCalledWith('session.export', { concise: true, format: 'md', session_id: 'runtime-1' })
    expect(writeClipboardText).toHaveBeenCalledWith('# Conversation: x\n\n## User\n\nhi\n')
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'success', message: d.conciseTranscriptCopied })
    )
  })

  it('does not claim success when the clipboard write fails', async () => {
    writeClipboardText.mockRejectedValue(new Error('Clipboard write is unavailable'))

    await copyConciseTranscript('runtime-1', (async () => ({ content: 'text' })) as never)

    expect(notify).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
    expect(notifyError).toHaveBeenCalledOnce()
  })

  it.each([[{ content: '' }], [{ content: '  \n' }], [{}], [null]])(
    'empty owner reply %j copies nothing',
    async reply => {
      await copyConciseTranscript('runtime-1', (async () => reply) as never)

      expect(writeClipboardText).not.toHaveBeenCalled()
      expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', message: d.conciseTranscriptEmpty }))
    }
  )

  it('reports a gateway that predates the option instead of copying UI rows', async () => {
    await copyConciseTranscript('runtime-1', (async () => {
      throw new Error('invalid_params')
    }) as never)

    expect(writeClipboardText).not.toHaveBeenCalled()
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ message: d.conciseTranscriptOutdated }))
  })

  it('needs a live session and never calls the gateway without one', async () => {
    const request = vi.fn()

    await copyConciseTranscript(null, request as never)

    expect(request).not.toHaveBeenCalled()
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ message: d.conciseTranscriptNeedsOpenSession }))
  })
})
