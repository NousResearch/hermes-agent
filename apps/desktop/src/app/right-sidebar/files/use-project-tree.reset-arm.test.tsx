// @vitest-environment jsdom
// Regression: a connection re-home (wiring's `resetProjectTreeState` wipe) used
// to strand any mounted tree in "loading" forever — the wipe bumped the request
// id (dropping in-flight commits) without changing any effect dependency, so
// nothing ever reloaded the root. A fresh window that mounts with a preset cwd
// (the IDE's URL-seeded workspace) hit this on every boot. The reset nonce
// re-arms the load effect.
import { render, screen } from '@testing-library/react'
import { StrictMode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const readDirCalls: string[] = []
let cacheKey = 'local:'

vi.mock('@/lib/desktop-fs', () => ({
  desktopFsCacheKey: () => cacheKey,
  desktopGitRoot: async () => null,
  isDesktopFsRemoteMode: () => false,
  readDesktopDir: async (p: string) => {
    readDirCalls.push(p)
    await new Promise(resolve => setTimeout(resolve, 30))

    return {
      entries: [
        { isDirectory: true, name: 'sub', path: `${p}/sub` },
        { isDirectory: false, name: 'a.ts', path: `${p}/a.ts` }
      ]
    }
  },
  readDesktopFileDataUrl: async () => 'data:,'
}))

import { resetProjectTreeState, useProjectTree } from './use-project-tree'

// ipc.ts guards on the raw bridge before delegating to the (mocked) fs lib.
;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
  readDir: async () => ({ entries: [] })
}

function Probe({ root }: { root: string }) {
  const tree = useProjectTree(root)

  return (
    <div data-testid="probe">
      {JSON.stringify({ cwd: tree.effectiveCwd, data: tree.data.length, error: tree.rootError, loading: tree.rootLoading })}
    </div>
  )
}

const readState = () => JSON.parse(screen.getByTestId('probe').textContent || '{}')

beforeEach(() => {
  resetProjectTreeState()
  cacheKey = 'local:'
  readDirCalls.length = 0
})

describe('useProjectTree boot robustness', () => {
  it('loads on a plain mount', async () => {
    render(<Probe root="D:/probe-root" />)
    await new Promise(resolve => setTimeout(resolve, 120))

    const state = readState()

    expect(state.loading).toBe(false)
    expect(state.data).toBe(2)
    expect(state.error).toBe(null)
  })

  it('loads under StrictMode double-invoked effects', async () => {
    render(
      <StrictMode>
        <Probe root="D:/probe-root-2" />
      </StrictMode>
    )
    await new Promise(resolve => setTimeout(resolve, 120))

    expect(readState().loading).toBe(false)
    expect(readState().data).toBe(2)
  })

  it('loads across a connection-key transition mid-read', async () => {
    cacheKey = 'local:'

    const view = render(<Probe root="D:/probe-root-3" />)

    setTimeout(() => {
      cacheKey = 'connection:abc:default'
      view.rerender(<Probe root="D:/probe-root-3" />)
    }, 10)

    await new Promise(resolve => setTimeout(resolve, 250))

    expect(readState().loading).toBe(false)
    expect(readState().data).toBe(2)
  })

  it('re-arms and loads when a connection wipe lands mid-read', async () => {
    render(<Probe root="D:/probe-root-4" />)

    // The re-home wipe races the in-flight first read.
    setTimeout(() => resetProjectTreeState(), 5)

    await new Promise(resolve => setTimeout(resolve, 250))

    const state = readState()

    expect(state.loading).toBe(false)
    expect(state.data).toBe(2)
  })

  it('reloads the root after a wipe on a settled tree', async () => {
    render(<Probe root="D:/probe-root-5" />)
    await new Promise(resolve => setTimeout(resolve, 120))
    expect(readState().data).toBe(2)

    const readsBefore = readDirCalls.length
    resetProjectTreeState()

    await new Promise(resolve => setTimeout(resolve, 120))

    expect(readState().loading).toBe(false)
    expect(readState().data).toBe(2)
    expect(readDirCalls.length).toBeGreaterThan(readsBefore)
  })
})
