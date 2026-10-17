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

it.each(['before.png', '@terminal:log.png'])(
  'sends the literal %s token with a mapping to its renamed attachment',
  async filename => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:image/png;base64,aW1hZ2U=') }
    })

    const original: ComposerAttachment = {
      id: `image:${filename}`,
      kind: 'image',
      label: filename,
      referenceName: filename,
      path: `/local/${filename}`
    }

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'image.attach_bytes') {
        return { attached: true, path: '/gateway/upload_123.png' } as never
      }

      return { admission_id: params?.submission_id, status: 'started' } as never
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
      expect(await result.current(`Explain this failure [ ${filename} ]`, { attachments: [original] })).toBe(true)
    })
    expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', {
      session_id: 'references-runtime',
      content_base64: 'aW1hZ2U=',
      filename: filename
    })
    const submit = requestGateway.mock.calls.find(([method]) => method === 'prompt.submit')
    expect(submit?.[1]).toMatchObject({
      text: `[${filename}] = "/gateway/upload_123.png"\n\nExplain this failure [ ${filename} ]`
    })
    expect(original.path).toBe(`/local/${filename}`)
  }
)

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
