import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { ComposerAttachment } from '@/store/composer'
import { setBusy } from '@/store/session'

import { attachmentContextRef } from './attachment-context'
import { useSubmitPrompt } from './submit'

import { uploadComposerAttachment } from '.'

afterEach(() => {
  cleanup()
  setBusy(false)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

it('sends the original positional tokens with a mapping to the renamed gateway attachments', async () => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { readFileDataUrl: vi.fn(async () => 'data:image/png;base64,aW1hZ2U=') }
  })

  const original: ComposerAttachment = {
    id: 'image:before.png',
    kind: 'image',
    label: 'before.png',
    referenceName: 'before.png',
    path: '/local/before.png'
  }

  const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>) => {
    if (method === 'image.attach_bytes') {
      return { attached: true, path: '/gateway/upload_123.png' } as never
    }

    return { accepted: true } as never
  })

  let state = createClientSessionState('references-runtime')

  const { result } = renderHook(() =>
    useSubmitPrompt({
      activeSessionIdRef: { current: 'references-runtime' },
      busyRef: { current: false },
      copy: en.desktop,
      createBackendSessionForSend: async () => 'references-runtime',
      getRoutedStoredSessionId: () => 'references-runtime',
      getRuntimeIdForStoredSession: () => 'references-runtime',
      getRouteToken: () => 'references',
      requestGateway,
      runtimeIdByStoredSessionIdRef: { current: new Map([['references-runtime', 'references-runtime']]) },
      resumeStoredSession: () => {},
      selectedStoredSessionIdRef: { current: 'references-runtime' },
      syncAttachmentsForSubmit: async (sessionId, attachments) => ({
        sessionId,
        attachments: await Promise.all(
          attachments.map(attachment =>
            uploadComposerAttachment(attachment, {
              remote: true,
              requestGateway,
              sessionId
            })
          )
        )
      }),
      updateSessionState: (_session, update) => {
        state = update(state)

        return state
      }
    })
  )

  await act(async () => {
    expect(await result.current('Explain this failure [ BEFORE.PNG ]', { attachments: [original] })).toBe(true)
  })
  expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', {
    session_id: 'references-runtime',
    content_base64: 'aW1hZ2U=',
    filename: 'before.png'
  })
  const submit = requestGateway.mock.calls.find(([method]) => method === 'prompt.submit')
  expect(submit?.[1]).toMatchObject({
    text: '[before.png] = "/gateway/upload_123.png"\n\nExplain this failure [ BEFORE.PNG ]'
  })
  expect(original.path).toBe('/local/before.png')
})

it('maps staged file references while keeping links and unreferenced attachments on the existing path', () => {
  const attachment: ComposerAttachment = {
    id: 'file:report.csv',
    kind: 'file',
    label: 'report.csv',
    referenceName: 'report.csv',
    refText: '@file:attachments/report-2.csv'
  }

  expect(attachmentContextRef(attachment, 'Analyze [report.csv]')).toBe('[report.csv] = @file:attachments/report-2.csv')
  expect(attachmentContextRef(attachment, '[report.csv](https://example.com)')).toBe(attachment.refText)
  expect(attachmentContextRef(attachment, 'Analyze the attachment')).toBe(attachment.refText)
})
