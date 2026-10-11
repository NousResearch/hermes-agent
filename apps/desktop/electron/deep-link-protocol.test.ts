import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { describe, expect, it, vi } from 'vitest'

import { registerDeepLinkProtocol } from './deep-link-protocol'

const desktopPackage = JSON.parse(
  readFileSync(fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8')
) as { desktopName?: string }

function context(
  overrides: Partial<Parameters<typeof registerDeepLinkProtocol>[1]> = {}
): Parameters<typeof registerDeepLinkProtocol>[1] {
  return {
    protocol: 'hermes',
    defaultApp: false,
    argv: ['/usr/bin/electron', '.'],
    execPath: '/usr/bin/electron',
    resolve: p => p,
    ...overrides
  }
}

describe('deep-link protocol registration', () => {
  it('reports the registrar boolean and never assumes success', () => {
    const yes = { setAsDefaultProtocolClient: vi.fn(() => true) }

    expect(registerDeepLinkProtocol(yes, context())).toBe(true)
    expect(yes.setAsDefaultProtocolClient).toHaveBeenCalledTimes(1)
    expect(yes.setAsDefaultProtocolClient).toHaveBeenCalledWith('hermes')

    const no = { setAsDefaultProtocolClient: vi.fn(() => false) }

    expect(registerDeepLinkProtocol(no, context())).toBe(false)
  })

  it('registers the dev scheme with execPath + the resolved entry script', () => {
    const app = { setAsDefaultProtocolClient: vi.fn(() => true) }

    registerDeepLinkProtocol(
      app,
      context({ protocol: 'hermes-dev', defaultApp: true, resolve: p => `/app/${p}` })
    )

    expect(app.setAsDefaultProtocolClient).toHaveBeenCalledWith('hermes-dev', '/usr/bin/electron', [
      '/app/.'
    ])
  })

  it('pins the Linux desktop-file identity to the launcher entry via package.json', () => {
    // Electron 40 has no app.setDesktopName: the desktopName field in
    // package.json is the documented surface the installed hermes.desktop
    // entry (written by `hermes desktop`) relies on.
    expect(desktopPackage.desktopName).toBe('hermes.desktop')
  })
})
