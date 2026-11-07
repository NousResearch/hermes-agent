// A canonical gateway answers `image.attach_bytes` by STAGING the file and queuing nothing
// (`gateway/session_images.py`), so the reply's `path`/`mime` are the only carrier of the image:
// the client has to declare them on `prompt.submit`. A Desktop that reaches `hermes serve` by URL
// (no `native_dial`, wire not yet known) takes this byte-staging branch even though the peer is
// canonical, so dropping the reply's `mime` dropped the image from the turn (#148682).
//
// The gate stays the REPLY, not the dial: a legacy peer's attach RPC queues the image inside the
// session and returns no `mime`, and its `prompt.submit` contract refuses an `attachments` key
// (extra=forbid, #144186). No mime in the reply therefore keeps that key off the wire.
import { cleanup } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { $composerAttachments, type ComposerAttachment } from '@/store/composer'
import { $connection, $currentCwd } from '@/store/session'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: RUNTIME_SESSION_ID })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

// The composer chip as `attachImagePath` creates it: a local path, no mime (the file has not been
// staged yet).
const HOST_IMAGE = 'C:\\Users\\alice\\Pictures\\screenshot.png'

function composerImage(): ComposerAttachment {
  return { detail: HOST_IMAGE, id: 'image:screenshot.png', kind: 'image', label: 'screenshot.png', path: HOST_IMAGE }
}

interface Call {
  method: string
  params: Record<string, unknown>
}

async function submitWithImage(attachReply: Record<string, unknown>) {
  $connection.set({ connectionId: 'remote', mode: 'remote', wsUrl: 'ws://gateway.test:9119/api/ws' } as never)
  $currentCwd.set('/root')

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { readFileDataUrl: vi.fn(async () => 'data:image/png;base64,aGVsbG8=') }
  })

  const calls: Call[] = []

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    calls.push({ method, params: params ?? {} })

    if (method === 'image.attach_bytes') {
      return attachReply as never
    }

    return {} as never
  })

  $composerAttachments.set([composerImage()])

  let handle: null | HarnessHandle = null
  await actRender(
    <Harness
      onReady={value => (handle = value)}
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
    />
  )

  expect(await handle!.submitText('what does this say?')).toBe(true)
  expect(calls.some(call => call.method === 'image.attach_bytes')).toBe(true)

  return calls.filter(call => call.method === 'prompt.submit').at(-1)!
}

afterEach(() => {
  cleanup()
  $connection.set(null)
  $composerAttachments.set([])
  vi.clearAllMocks()
})

test('a canonically staged image reaches prompt.submit with the reply mime', async () => {
  const submit = await submitWithImage({
    attached: true,
    mime: 'image/png',
    path: '/gateway/cache/images/upload_ab12.png'
  })

  expect(submit.params.attachments).toEqual([{ mime: 'image/png', path: '/gateway/cache/images/upload_ab12.png' }])
})

test('a legacy attach reply without mime keeps attachments off submit', async () => {
  const submit = await submitWithImage({ attached: true, path: '/root/.hermes/images/upload_cd34.png' })

  expect(submit.params).not.toHaveProperty('attachments')
})
