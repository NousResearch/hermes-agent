import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { reachablePreviewUrl } from './preview-reach'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installBridge(reachPreviewUrl?: unknown) {
  desktopWindow.hermesDesktop = { reachPreviewUrl } as unknown as Window['hermesDesktop']
}

afterEach(() => {
  $connection.set(null)
  delete desktopWindow.hermesDesktop
})

describe('reachablePreviewUrl', () => {
  // On a local backend the agent's localhost IS this machine's localhost —
  // rewriting would break a URL that already works.
  it('leaves a local gateway alone without touching the bridge', async () => {
    const bridge = vi.fn()

    $connection.set({ mode: 'local' } as never)
    installBridge(bridge)

    expect(await reachablePreviewUrl('http://localhost:5173/')).toBe('http://localhost:5173/')
    expect(bridge).not.toHaveBeenCalled()
  })

  it('asks main to resolve the URL on a remote gateway', async () => {
    const bridge = vi.fn(async () => 'http://127.0.0.1:45173/')

    $connection.set({ mode: 'remote' } as never)
    installBridge(bridge)

    expect(await reachablePreviewUrl('http://localhost:5173/')).toBe('http://127.0.0.1:45173/')
    expect(bridge).toHaveBeenCalledWith('http://localhost:5173/')
  })

  // A url/cloud remote has no tunnel to borrow, so main hands the URL back
  // unchanged; the pane's own error explains it from there.
  it('passes the original through when main cannot reach it', async () => {
    $connection.set({ mode: 'remote' } as never)
    installBridge(vi.fn(async (url: string) => url))

    expect(await reachablePreviewUrl('http://localhost:5173/')).toBe('http://localhost:5173/')
  })

  it('falls back to the original when the bridge throws', async () => {
    $connection.set({ mode: 'remote' } as never)
    installBridge(
      vi.fn(async () => {
        throw new Error('no handler')
      })
    )

    expect(await reachablePreviewUrl('http://localhost:5173/')).toBe('http://localhost:5173/')
  })

  // An older Electron main predates the handler entirely.
  it('survives a bridge with no reach method', async () => {
    $connection.set({ mode: 'remote' } as never)
    installBridge(undefined)

    expect(await reachablePreviewUrl('http://localhost:5173/')).toBe('http://localhost:5173/')
  })

  it('ignores an empty url', async () => {
    $connection.set({ mode: 'remote' } as never)
    const bridge = vi.fn()

    installBridge(bridge)

    expect(await reachablePreviewUrl('')).toBe('')
    expect(bridge).not.toHaveBeenCalled()
  })
})
