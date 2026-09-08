import { describe, expect, it, vi } from 'vitest'

import {
  buildTailscaleRoutePlan,
  probeMobileCompanionRoute,
  refreshMobileCompanionRoute,
  tailscaleBinaryCandidates
} from './mobile-companion-route'

describe('mobile companion Tailscale route', () => {
  it('maps one managed HTTPS port to the existing loopback backend', () => {
    expect(buildTailscaleRoutePlan('https://desktop.example.ts.net:9443', 'http://127.0.0.1:62600')).toEqual({
      localTarget: 'http://127.0.0.1:62600',
      publicHostname: 'desktop.example.ts.net',
      publicPort: 9443
    })
  })

  it.each([
    'http://desktop.example.ts.net:9443',
    'https://desktop.example.com:9443',
    'https://user:password@desktop.example.ts.net:9443',
    'https://desktop.example.ts.net:9443/private',
    'not-a-url'
  ])('rejects an unmanaged public target: %s', publicUrl => {
    expect(() => buildTailscaleRoutePlan(publicUrl, 'http://127.0.0.1:62600')).toThrow('unsupported-public-url')
  })

  it.each([
    'http://192.168.1.2:62600',
    'https://gateway.example.com:62600',
    'http://127.0.0.1:62600/other',
    'not-a-url'
  ])(
    'rejects a non-loopback backend: %s',
    backend => {
      expect(() => buildTailscaleRoutePlan('https://desktop.example.ts.net:9443', backend)).toThrow(
        'unsupported-backend'
      )
    }
  )

  it('uses OS-specific binary candidates before the PATH fallback', () => {
    expect(
      tailscaleBinaryCandidates('win32', {
        ProgramFiles: 'C:\\Program Files',
        LOCALAPPDATA: 'C:\\Users\\test\\AppData\\Local'
      })
    ).toEqual([
      'C:\\Program Files\\Tailscale\\tailscale.exe',
      'C:\\Users\\test\\AppData\\Local\\Tailscale\\tailscale.exe',
      'tailscale'
    ])
    expect(tailscaleBinaryCandidates('darwin', {})).toContain('/Applications/Tailscale.app/Contents/MacOS/Tailscale')
  })

  it('recognizes a stale managed route without blocking unmanaged HTTPS proxies', async () => {
    await expect(
      probeMobileCompanionRoute('https://desktop.example.ts.net:9443', {
        probe: async () => ({ status: 502 })
      })
    ).resolves.toEqual({ error: 'route-unreachable', managed: true, ok: false })

    await expect(
      probeMobileCompanionRoute('https://gateway.example.com', {
        probe: async () => {
          throw new Error('must not probe unmanaged routes')
        }
      })
    ).resolves.toEqual({ managed: false, ok: true })
  })

  it('updates only the selected HTTPS port and waits for that route to answer', async () => {
    const calls: Array<{ args: string[]; command: string }> = []
    const probe = vi.fn().mockResolvedValueOnce({ status: 502 }).mockResolvedValue({ status: 200 })

    const result = await refreshMobileCompanionRoute('https://desktop.example.ts.net:9443', 'http://127.0.0.1:62600', {
      env: { ProgramFiles: 'C:\\Program Files' },
      platform: 'win32',
      probe,
      run: async (command, args) => {
        calls.push({ args, command })

        return args[0] === 'status'
          ? JSON.stringify({ BackendState: 'Running', Self: { DNSName: 'desktop.example.ts.net.', Online: true } })
          : ''
      }
    })

    expect(result).toEqual({ managed: true, ok: true })
    expect(calls).toEqual([
      {
        args: ['status', '--json'],
        command: 'C:\\Program Files\\Tailscale\\tailscale.exe'
      },
      {
        args: ['serve', '--bg', '--yes', '--https=9443', 'http://127.0.0.1:62600'],
        command: 'C:\\Program Files\\Tailscale\\tailscale.exe'
      }
    ])
    expect(probe).toHaveBeenCalledTimes(2)
  })

  it('falls through missing binaries but does not hide a real command failure', async () => {
    const missing = Object.assign(new Error('missing'), { code: 'ENOENT' })
    const run = vi.fn().mockRejectedValueOnce(missing).mockRejectedValueOnce(new Error('permission denied'))

    await expect(
      refreshMobileCompanionRoute('https://desktop.example.ts.net:9443', 'http://127.0.0.1:62600', {
        env: { ProgramFiles: 'C:\\Program Files' },
        platform: 'win32',
        run
      })
    ).resolves.toEqual({ error: 'tailscale-failed', managed: true, ok: false })
    expect(run).toHaveBeenCalledTimes(2)
  })

  it('refuses to mutate a route for another Tailscale node', async () => {
    const run = vi
      .fn()
      .mockResolvedValue(
        JSON.stringify({ BackendState: 'Running', Self: { DNSName: 'other.example.ts.net.', Online: true } })
      )

    await expect(
      refreshMobileCompanionRoute('https://desktop.example.ts.net:9443', 'http://127.0.0.1:62600', {
        env: { ProgramFiles: 'C:\\Program Files' },
        platform: 'win32',
        run
      })
    ).resolves.toEqual({ error: 'tailscale-host-mismatch', managed: true, ok: false })
    expect(run).toHaveBeenCalledOnce()
    expect(run).toHaveBeenCalledWith('C:\\Program Files\\Tailscale\\tailscale.exe', ['status', '--json'])
  })
})
