import { JsonRpcGatewayError } from '@hermes/shared'
import { cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { errorRecoveryPlan, type ErrorSurface } from '@/lib/error-surface'
import { $composerAttachments } from '@/store/composer'
import { clearNotifications } from '@/store/notifications'
import { $connection, $currentCwd } from '@/store/session'

import { actRender, Harness, type HarnessHandle } from './index-test-utils'

// RT-37 (#140385): a prompt on a chat whose gateway owner is mid-turn.
describe('usePromptActions gateway-owner queue (#140385)', () => {
  afterEach(() => {
    cleanup()
    clearNotifications()
    $composerAttachments.set([])
  })

  it('re-uploads the image of a draft restored after a SESSION_OWNER_BUSY refusal: serve dropped it (#140968)', async () => {
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('/root')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:image/jpeg;base64,aGVsbG8=') }
    })
    $composerAttachments.set([
      {
        detail: '/home/alice/photo.jpg',
        id: 'image:photo.jpg',
        kind: 'image',
        label: 'photo.jpg',
        occurrenceId: 'occurrence-photo',
        path: '/home/alice/photo.jpg'
      }
    ])

    const calls: string[] = []
    let submits = 0

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      if (method === 'image.attach_bytes' || method === 'image.attach') {
        return { attached: true, path: '/root/tmp/photo.jpg' } as never
      }

      if (method === 'prompt.submit' && ++submits === 1) {
        throw new JsonRpcGatewayError(
          'The gateway is running a turn in this chat and an attachment cannot join its queue. The image was removed. Nothing was sent.',
          { code: 4090, data: { reason: 'SESSION_OWNER_BUSY' } }
        )
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.submitText('describe this')).toBe(false)
    expect($composerAttachments.get()).toHaveLength(1)

    await handle!.submitText('describe this')

    const uploads = (m: string) => m === 'image.attach_bytes' || m === 'image.attach'
    const firstRefusal = calls.indexOf('prompt.submit')
    const resend = calls.lastIndexOf('prompt.submit')
    expect(resend).toBeGreaterThan(firstRefusal)
    expect(calls.slice(firstRefusal + 1, resend).some(uploads)).toBe(true)
  })

  it('turns a SESSION_OWNER_BUSY refusal into a retryable card, not "Start new session" (#140385)', async () => {
    let latest: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'prompt.submit') {
        throw new JsonRpcGatewayError(
          'The gateway is running a turn in this chat and a rewind or attachment cannot join its queue. Nothing was sent. Send it again when the current turn ends.',
          { code: 4090, data: { reason: 'SESSION_OWNER_BUSY' } }
        )
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={next => {
          latest = next
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    expect(await handle!.submitText('continue here')).toBe(false)
    const bubble = (latest?.messages as { errorSurface?: Record<string, unknown> }[]).at(-1)
    expect(bubble?.errorSurface).toEqual({ layer: 'gateway', code: 'SESSION_OWNER_BUSY', retryable: true })
  })

  it('turns an unconfirmed forward into a card without Retry: the owner may already hold it', async () => {
    let latest: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'prompt.submit') {
        throw new JsonRpcGatewayError(
          'The gateway is running a turn in this chat and did not confirm your message in time. It may already be queued there: check the chat before sending it again.',
          { code: 4090, data: { reason: 'SESSION_OWNER_UNCONFIRMED' } }
        )
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={next => {
          latest = next
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    expect(await handle!.submitText('continue here')).toBe(false)
    const bubble = (latest?.messages as { errorSurface?: ErrorSurface }[]).at(-1)
    expect(bubble?.errorSurface).toEqual({ layer: 'gateway', code: 'SESSION_OWNER_UNCONFIRMED', retryable: false })
    expect(errorRecoveryPlan(bubble?.errorSurface).retry).toBe(false)
  })

  it('shows a prompt the legacy backend forwarded to the gateway owner as queued, not as a failed turn (#140385)', async () => {
    let latest: Record<string, unknown> | undefined

    // The legacy backend refuses `submission_id` (4000, pre-admission), then answers the identityless
    // retry with the owner's queued receipt: the prompt waits in the gateway's FIFO.
    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit' && params?.submission_id) {
        throw new JsonRpcGatewayError("unknown field 'submission_id'", { code: 4000 })
      }

      if (method === 'prompt.submit') {
        return { status: 'queued', admission_id: 'owner-admission', forwarded: true } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={next => {
          latest = next
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    expect(await handle!.submitText('my own question')).toBe(true)
    const messages = (latest?.messages ?? []) as { error?: string; role?: string }[]
    // No failed-turn card and no optimistic bubble: the queued-row card stands for the send.
    expect(messages.some(message => message.error)).toBe(false)
    expect(messages.some(message => message.role === 'user')).toBe(false)
    expect(latest?.busy).toBe(false)
    expect(latest?.awaitingResponse).toBe(false)
  })
})
