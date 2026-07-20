import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getStatus } from '@/hermes'
import { desktopFsCacheKey, isDesktopFsRemoteMode, readDesktopDir, readDesktopFileText } from '@/lib/desktop-fs'

import {
  $generatedViewProblems,
  $generatedViews,
  $openGeneratedViewIds,
  discoverGeneratedViews,
  openGeneratedView
} from './store'

vi.mock('@/hermes', () => ({ getStatus: vi.fn() }))
vi.mock('@/lib/desktop-fs', () => ({
  desktopFsCacheKey: vi.fn(() => 'local:'),
  isDesktopFsRemoteMode: vi.fn(() => true),
  readDesktopDir: vi.fn(),
  readDesktopFileText: vi.fn()
}))

const manifest = JSON.stringify({
  version: 1,
  id: 'usage-monitor',
  title: 'Usage Monitor',
  entry: 'index.html',
  capabilities: ['theme:read'],
  bindings: ['hermes:usage-30d']
})

function textResult(path: string, text: string) {
  return { byteSize: new TextEncoder().encode(text).byteLength, path, text }
}

describe('generated-view discovery', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $generatedViews.set([])
    $generatedViewProblems.set([])
    $openGeneratedViewIds.set([])
    vi.mocked(desktopFsCacheKey).mockReturnValue('local:')
    vi.mocked(isDesktopFsRemoteMode).mockReturnValue(false)
    vi.mocked(getStatus).mockResolvedValue({ hermes_home: '/home/me/.hermes' } as never)
    vi.mocked(readDesktopDir).mockResolvedValue({
      entries: [{ isDirectory: true, name: 'usage-monitor', path: '/home/me/.hermes/generated-views/usage-monitor' }]
    })
    vi.mocked(readDesktopFileText).mockImplementation(async path => {
      if (path.endsWith('view.json')) {
        return textResult('/home/me/.hermes/generated-views/usage-monitor/view.json', manifest)
      }

      return textResult('/home/me/.hermes/generated-views/usage-monitor/index.html', '<h1>Usage</h1>')
    })
  })

  it('loads a valid document through the filesystem facade and computes a digest', async () => {
    await discoverGeneratedViews()

    expect($generatedViews.get()).toHaveLength(1)
    expect($generatedViews.get()[0]).toMatchObject({
      connectionKey: 'local:',
      digest: expect.stringMatching(/^[a-f0-9]{64}$/),
      entryPath: '/home/me/.hermes/generated-views/usage-monitor/index.html',
      manifest: { id: 'usage-monitor', title: 'Usage Monitor' }
    })
    expect(readDesktopDir).toHaveBeenCalledWith('/home/me/.hermes/generated-views')
    expect($generatedViewProblems.get()).toEqual([])
  })

  it('fails a traversal-shaped canonical result closed', async () => {
    vi.mocked(readDesktopFileText).mockImplementation(async path =>
      path.endsWith('view.json')
        ? textResult('/home/me/.hermes/generated-views/usage-monitor/view.json', manifest)
        : textResult('/home/me/.hermes/escaped/index.html', '<h1>Escaped</h1>')
    )

    await discoverGeneratedViews()

    expect($generatedViews.get()).toEqual([])
    expect($generatedViewProblems.get()[0]?.message).toMatch(/outside/)
  })

  it('removes deleted documents and their open pane source on rescan', async () => {
    await discoverGeneratedViews()
    openGeneratedView('usage-monitor')
    expect($openGeneratedViewIds.get()).toEqual(['usage-monitor'])
    expect($generatedViews.get()[0]?.connectionKey).toBe('local:')

    vi.mocked(readDesktopDir).mockResolvedValueOnce({ entries: [] })
    await discoverGeneratedViews()

    expect($generatedViews.get()).toEqual([])
    expect($openGeneratedViewIds.get()).toEqual([])
  })

  it('restores local open views across a startup connection-key change', async () => {
    await discoverGeneratedViews()
    openGeneratedView('usage-monitor')
    expect($openGeneratedViewIds.get()).toEqual(['usage-monitor'])

    vi.mocked(desktopFsCacheKey).mockReturnValue('local::http://127.0.0.1:43123')
    $openGeneratedViewIds.set([])
    await discoverGeneratedViews()

    expect($openGeneratedViewIds.get()).toEqual(['usage-monitor'])
  })

  it('isolates remote open views by connection and Hermes home', async () => {
    vi.mocked(isDesktopFsRemoteMode).mockReturnValue(true)
    vi.mocked(desktopFsCacheKey).mockReturnValue('remote:alpha:https://alpha.example')
    await discoverGeneratedViews()
    openGeneratedView('usage-monitor')

    vi.mocked(desktopFsCacheKey).mockReturnValue('remote:beta:https://beta.example')
    await discoverGeneratedViews()

    expect($openGeneratedViewIds.get()).toEqual([])
  })

  it('surfaces malformed documents without exposing them as runnable views', async () => {
    vi.mocked(readDesktopFileText).mockResolvedValueOnce(textResult('/views/usage-monitor/view.json', '{'))

    await discoverGeneratedViews()

    expect($generatedViews.get()).toEqual([])
    expect($generatedViewProblems.get()[0]).toMatchObject({ id: 'usage-monitor', message: 'view.json is not valid JSON' })
  })
})
