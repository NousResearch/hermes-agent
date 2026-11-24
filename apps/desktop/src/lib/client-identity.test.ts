import { afterEach, describe, expect, it, vi } from 'vitest'

type Version = Record<string, string | undefined>

async function identityFrom(getVersion?: () => Promise<Version>) {
  vi.resetModules()
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = getVersion ? { getVersion } : undefined

  return (await import('./client-identity')).desktopClientIdentity
}

const identity = async (getVersion?: () => Promise<Version>) => (await identityFrom(getVersion))()

const DESKTOP = { kind: 'desktop', name: 'hermes-desktop' }

describe('desktopClientIdentity', () => {
  afterEach(() => {
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
    delete document.documentElement.dataset.hermesDesktopHost
    vi.useRealTimers()
  })

  it('names the Electron app with its version', async () => {
    await expect(identity(async () => ({ appVersion: '0.21.5+1913' }))).resolves.toEqual({
      ...DESKTOP,
      version: '0.21.5+1913'
    })
  })

  it("carries this machine's own version, not the display string or the backend's", async () => {
    // A release build: the display strings hold spaces, which a label's version may not.
    await expect(
      identity(async () => ({
        appVersion: '2026.11.3 (stable #12, abcdef12)',
        baseVersion: '2026.11.3',
        clientVersion: '2026.11.3 (stable #12, abcdef12)'
      }))
    ).resolves.toEqual({ ...DESKTOP, version: '2026.11.3' })

    // A source install on a remote backend: `appVersion` names the backend it is connected to.
    await expect(identity(async () => ({ appVersion: '0.21.5+1913', clientVersion: '0.22.0+7' }))).resolves.toEqual({
      ...DESKTOP,
      version: '0.22.0+7'
    })
  })

  it('names the Webapp behind the browser bridge', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'

    await expect(identity(async () => ({ appVersion: '0.21.5' }))).resolves.toEqual({
      kind: 'webapp',
      name: 'hermes-webapp',
      version: '0.21.5'
    })
  })

  it('leaves out a version the backend would refuse, a placeholder, or one the bridge cannot read', async () => {
    await expect(identity(async () => ({ appVersion: '0.21 dev' }))).resolves.toEqual(DESKTOP)

    for (const placeholder of ['unknown', '0.0.0', 'v0.0.0', '']) {
      await expect(identity(async () => ({ appVersion: placeholder }))).resolves.toEqual(DESKTOP)
    }

    await expect(identity(async () => Promise.reject(new Error('ipc closed')))).resolves.toEqual(DESKTOP)
    await expect(identity()).resolves.toEqual(DESKTOP)
  })

  it('does not hold the hello for a slow version read, and carries the version once it is known', async () => {
    vi.useFakeTimers()

    let answer: (version: Version) => void = () => undefined
    const read = vi.fn(() => new Promise<Version>(resolve => (answer = resolve)))
    const desktopClientIdentity = await identityFrom(read)
    const settled = vi.fn()

    void desktopClientIdentity().then(settled)
    await vi.advanceTimersByTimeAsync(999)
    expect(settled).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(1)
    expect(settled).toHaveBeenCalledExactlyOnceWith(DESKTOP)

    // The read finishes later: the next connection's hello carries it at once, with no second read.
    answer({ appVersion: '0.21.5+1913' })
    await vi.advanceTimersByTimeAsync(0)

    await expect(desktopClientIdentity()).resolves.toEqual({ ...DESKTOP, version: '0.21.5+1913' })
    expect(read).toHaveBeenCalledOnce()
  })
})
