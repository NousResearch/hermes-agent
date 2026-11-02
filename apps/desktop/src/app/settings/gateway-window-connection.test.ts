import { describe, expect, it } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

import { normalizeGatewaySettingsState } from './gateway-settings'
import { windowConnectionSettings } from './gateway-window-connection'

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
    {
      id: 'homelab',
      kind: 'remote',
      label: 'Homelab',
      url: 'http://homelab:9119',
      authMode: 'token',
      tokenPreview: '...abc',
      tokenSet: true
    },
    {
      id: 'box',
      kind: 'ssh',
      label: 'Box',
      host: 'box.lan',
      user: 'me',
      port: 2222,
      tokenPreview: null,
      tokenSet: false
    }
  ],
  primary: 'local',
  secureTokenStorage: true,
  version: 2
}

const savedLocal = normalizeGatewaySettingsState({ mode: 'local' })

describe('Settings > Gateways > This window', () => {
  it('shows the gateway the window is on, not the primary it booted', () => {
    const state = windowConnectionSettings(savedLocal, registry, 'homelab')

    expect(state.mode).toBe('remote')
    expect(state.remoteUrl).toBe('http://homelab:9119')

    const ssh = windowConnectionSettings(savedLocal, registry, 'box')

    expect(ssh).toMatchObject({ mode: 'ssh', sshHost: 'box.lan', sshUser: 'me', sshPort: 2222 })
  })

  it('never offers the saved token for another gateway URL', () => {
    const savedRemote = normalizeGatewaySettingsState({
      mode: 'remote',
      remoteUrl: 'http://other:9119',
      remoteTokenSet: true,
      remoteTokenPreview: '...zzz'
    })

    expect(windowConnectionSettings(savedRemote, registry, 'homelab')).toMatchObject({
      remoteTokenSet: false,
      remoteTokenPreview: null
    })

    const sameUrl = { ...savedRemote, remoteUrl: 'http://homelab:9119/' }

    expect(windowConnectionSettings(sameUrl, registry, 'homelab').remoteTokenSet).toBe(true)
  })

  it('keeps the saved config when the window is on the primary or pinned by env', () => {
    expect(windowConnectionSettings(savedLocal, registry, 'local')).toBe(savedLocal)
    expect(windowConnectionSettings(savedLocal, registry, null)).toBe(savedLocal)

    const pinned = { ...savedLocal, envOverride: true }

    expect(windowConnectionSettings(pinned, registry, 'homelab')).toBe(pinned)
  })
})
