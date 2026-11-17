import { JsonRpcGatewayError } from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

import {
  appGuidanceFromError,
  isAppBlocked,
  officialAppDownloadUrl,
  PRESENCE_TIMEOUT_MS,
  presenceForCatalogPlugin
} from './plugin-app-guidance'

const missing = {
  reason: 'missing_app',
  app_title: 'Godot 4',
  download_url: 'https://godotengine.org/download/macos/'
} as const

describe('catalog app preflight', () => {
  it('asks the backend for only the selected catalog name and target profile', async () => {
    const request = vi.fn(async () => ({ presence: [{ name: 'game-editor', ...missing }] }))
    expect(await presenceForCatalogPlugin(request as never, 'game-editor', 'research')).toMatchObject(missing)
    expect(request).toHaveBeenCalledWith(
      'plugins.manage',
      {
        action: 'presence',
        names: ['game-editor'],
        profile: 'research'
      },
      PRESENCE_TIMEOUT_MS
    )
  })

  it('bounds the read so a stalled backend leaves Install possible instead of disabled', async () => {
    const request = vi.fn(async () => {
      throw new Error('request timed out after 30s: plugins.manage')
    })

    expect(PRESENCE_TIMEOUT_MS).toBeLessThan(120_000)
    expect(await presenceForCatalogPlugin(request as never, 'game-editor', 'default')).toEqual({
      reason: 'uncertain',
      app_title: 'game-editor'
    })
    expect(isAppBlocked({ reason: 'uncertain', app_title: 'game-editor' })).toBe(false)
  })

  it('treats older backend method errors as unsupported and missing rows as uncertain', async () => {
    expect(
      await presenceForCatalogPlugin(
        vi.fn(async () => {
          throw new JsonRpcGatewayError('unknown action', { code: 4019 })
        }) as never,
        'game-editor',
        'default'
      )
    ).toBeNull()
    expect(
      await presenceForCatalogPlugin(vi.fn(async () => ({ presence: [] })) as never, 'game-editor', 'default')
    ).toEqual({ reason: 'uncertain', app_title: 'game-editor' })
    expect(
      await presenceForCatalogPlugin(
        vi.fn(async () => {
          throw new Error('network unavailable')
        }) as never,
        'game-editor',
        'default'
      )
    ).toEqual({ reason: 'uncertain', app_title: 'game-editor' })
  })

  it('accepts typed install refusal only from the gateway code and backend host', () => {
    expect(appGuidanceFromError(new JsonRpcGatewayError('refused', { code: 5026, data: missing }))).toMatchObject(
      missing
    )
    expect(appGuidanceFromError(new JsonRpcGatewayError('refused', { code: 5000, data: missing }))).toBeNull()
    expect(
      appGuidanceFromError(
        new JsonRpcGatewayError('refused', { code: 5026, data: { ...missing, reason: 'not-a-reason' } })
      )
    ).toBeNull()
  })

  it('offers safe HTTPS vendor links only for actionable blockers', () => {
    expect(officialAppDownloadUrl(missing)).toBe('https://godotengine.org/download/macos/')
    expect(
      officialAppDownloadUrl({ ...missing, download_url: 'https://godotengine.org@evil.test/download/macos/' })
    ).toBeNull()
    expect(officialAppDownloadUrl({ ...missing, download_url: 'http://godotengine.org/download/macos/' })).toBeNull()
    expect(officialAppDownloadUrl({ ...missing, reason: 'unsupported_os' })).toBeNull()
  })

  it('allows a reviewed backend HTTPS vendor link without a core title map', () => {
    expect(
      officialAppDownloadUrl({ ...missing, app_title: 'Paint Studio', download_url: 'https://paint.example/download' })
    ).toBe('https://paint.example/download')
    expect(officialAppDownloadUrl({ ...missing, download_url: 'https://user@evil.example/download' })).toBeNull()
    expect(officialAppDownloadUrl({ ...missing, download_url: 'https://127.0.0.1/download' })).toBeNull()
  })

  it.each([
    'https://127.1/',
    'https://0x7f.0.0.1/',
    'https://0177.0.0.1/',
    'https://10.1/x',
    'https://printer.local/',
    'https://nas.home.arpa/',
    'https://api.internal/',
    'https://www.nvidia.com:443/en-us/software/nvidia-app/'
  ])('rejects a numeric or special-use vendor host or an explicit port: %s', url => {
    expect(officialAppDownloadUrl({ ...missing, download_url: url })).toBeNull()
  })

  it('holds an unreadable required app version without an install retry loop', () => {
    expect(isAppBlocked({ ...missing, reason: 'uncertain', required_version: '11.0', found_version: null })).toBe(true)
    expect(isAppBlocked({ ...missing, reason: 'uncertain', required_version: null })).toBe(false)
  })
})
