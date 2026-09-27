import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { uploadComposerAttachment } from '.'

const path = '/Users/client/Application Support/Hermes/composer-images/shot.png'
const attachment = { id: 'shot', kind: 'image' as const, label: 'shot.png', path }
const missing = () => new JsonRpcGatewayError(`image not found: ${path}`, { code: 4016 })

afterEach(() => vi.unstubAllGlobals())

it('recovers a missing gateway path using client bytes without changing sessions', async () => {
  const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,aGVsbG8=')
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'image.attach') throw missing()
    return { attached: true, path: '/gateway/shot.png' } as never
  })

  await expect(
    uploadComposerAttachment(attachment, {
      remote: false,
      sessionId: 'existing',
      requestGateway
    })
  ).resolves.toMatchObject({ path: '/gateway/shot.png', attachedSessionId: 'existing' })
  expect(readFileDataUrl).toHaveBeenCalledExactlyOnceWith(path)
  expect(requestGateway.mock.calls.map(([method]) => method)).toEqual(['image.attach', 'image.attach_bytes'])
  expect(requestGateway).toHaveBeenLastCalledWith('image.attach_bytes', {
    session_id: 'existing',
    content_base64: 'aGVsbG8=',
    filename: 'shot.png'
  })
})

it('reuses original bytes across session recovery and ignores a thumbnail preview', async () => {
  const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,b3JpZ2luYWw=')
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  const onRecovered = vi.fn()
  const onSessionRecovered = vi.fn()
  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'image.attach') throw missing()
    if (method === 'session.resume') return { session_id: 'recovered' } as never
    if (params?.session_id === 'existing') throw new JsonRpcGatewayError('session not found', { code: 4001 })
    expect(onRecovered).toHaveBeenCalledWith('recovered')
    return { attached: true, path: '/gateway/shot.png' } as never
  })
  await expect(
    uploadComposerAttachment(
      { ...attachment, previewUrl: 'data:image/png;base64,dGh1bWI=' },
      {
        remote: false,
        sessionId: 'existing',
        storedSessionId: 'fallback-stored',
        requestGateway,
        onRecovered,
        onSessionRecovered
      }
    )
  ).resolves.toMatchObject({ attachedSessionId: 'recovered' })
  expect(readFileDataUrl).toHaveBeenCalledTimes(1)
  expect(requestGateway.mock.calls.map(([method]) => method)).toEqual([
    'image.attach',
    'image.attach_bytes',
    'session.resume',
    'image.attach_bytes'
  ])
  expect(requestGateway).toHaveBeenLastCalledWith('image.attach_bytes', {
    session_id: 'recovered',
    content_base64: 'b3JpZ2luYWw=',
    filename: 'shot.png'
  })
  expect(onSessionRecovered).toHaveBeenCalledWith('recovered')
})

it('keeps successful gateway-native paths without reading client disk', async () => {
  const readFileDataUrl = vi.fn()
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  const requestGateway = vi.fn(async () => ({ attached: true, path: '/gateway/native.png' }) as never)
  await expect(
    uploadComposerAttachment(attachment, {
      remote: false,
      sessionId: 'existing',
      requestGateway
    })
  ).resolves.toMatchObject({ path: '/gateway/native.png' })
  expect(readFileDataUrl).not.toHaveBeenCalled()
  expect(requestGateway).toHaveBeenCalledTimes(1)
})

it.each([
  new JsonRpcGatewayError('unsupported image: shot.png', { code: 4016 }),
  new JsonRpcGatewayError('image not found: path', { code: 403 }),
  new Error('image not found: path')
])('does not retry other failures (%s)', async error => {
  const readFileDataUrl = vi.fn()
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  const requestGateway = vi.fn(async () => {
    throw error
  })
  await expect(
    uploadComposerAttachment(attachment, {
      remote: false,
      sessionId: 'existing',
      requestGateway
    })
  ).rejects.toBe(error)
  expect(readFileDataUrl).not.toHaveBeenCalled()
  expect(requestGateway).toHaveBeenCalledTimes(1)
})

it.each(['empty', 'throws'])('preserves the path failure if local bytes are unavailable: %s', async mode => {
  const error = missing()
  const readFileDataUrl = vi.fn(async () => {
    if (mode === 'throws') throw new Error('disk unreadable')
    return ''
  })
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  const requestGateway = vi.fn(async () => {
    throw error
  })
  await expect(
    uploadComposerAttachment(attachment, {
      remote: false,
      sessionId: 'existing',
      requestGateway
    })
  ).rejects.toBe(error)
  expect(readFileDataUrl).toHaveBeenCalledTimes(1)
  expect(requestGateway).toHaveBeenCalledTimes(1)
})

it('surfaces a failed bytes upload without looping', async () => {
  vi.stubGlobal('hermesDesktop', { readFileDataUrl: vi.fn(async () => 'data:image/png;base64,aGVsbG8=') })
  const error = new Error('upload failed')
  const requestGateway = vi.fn(async (method: string) => {
    throw method === 'image.attach' ? missing() : error
  })
  await expect(
    uploadComposerAttachment(attachment, {
      remote: false,
      sessionId: 'existing',
      requestGateway
    })
  ).rejects.toBe(error)
  expect(requestGateway).toHaveBeenCalledTimes(2)
})
