// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $connection } from '@/store/session'

import { $cloudOwnsUpdates, cloudOwnsUpdates, isHermesCloudUrl } from './cloud-updates'

const remote = (baseUrl: string, remoteKind?: HermesConnection['remoteKind']): HermesConnection =>
  ({ baseUrl, mode: 'remote', remoteKind }) as HermesConnection

afterEach(() => {
  $connection.set(null)
  vi.unstubAllGlobals()
  document.documentElement.removeAttribute('data-hermes-desktop-host')
})

describe('Hermes Cloud owns updates', () => {
  it('recognises production and staging agent hosts, nothing else', () => {
    expect(isHermesCloudUrl('https://ares-3009.agents.nousresearch.com')).toBe(true)
    expect(isHermesCloudUrl('https://ares--app.agents.staging-nousresearch.com/')).toBe(true)
    expect(isHermesCloudUrl('https://agents.nousresearch.com.evil.example')).toBe(false)
    expect(isHermesCloudUrl('https://my-box.tailnet.ts.net')).toBe(false)
    expect(isHermesCloudUrl('not a url')).toBe(false)
  })

  it('native Desktop connected to a Cloud agent: only the backend is Cloud-owned', () => {
    $connection.set(remote('https://lab.agents.nousresearch.com', 'cloud'))
    expect($cloudOwnsUpdates.get()).toEqual({ backend: true, client: false })

    $connection.set(remote('https://my-box.example.com', 'url'))
    expect($cloudOwnsUpdates.get()).toEqual({ backend: false, client: false })
  })

  it('a Webapp tab served by a Cloud agent hides every update target', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.stubGlobal('location', new URL('https://lab--app.agents.nousresearch.com/'))
    expect(cloudOwnsUpdates('client', null)).toBe(true)
    expect(cloudOwnsUpdates('backend', null)).toBe(true)

    vi.stubGlobal('location', new URL('http://127.0.0.1:9120/'))
    expect(cloudOwnsUpdates('client', null)).toBe(false)
  })
})
