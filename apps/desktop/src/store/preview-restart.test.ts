import { afterEach, describe, expect, it, vi } from 'vitest'

const TABS_KEY = 'hermes.desktop.previewTabs.v2'

function fileTab(path: string) {
  return {
    id: `file:file://${path}`,
    target: { kind: 'file' as const, label: path.split('/').at(-1) ?? path, path, source: path, url: `file://${path}` }
  }
}

function urlTab(id: string, url: string) {
  return { id: `url:${id}`, target: { kind: 'url' as const, label: url, source: url, url } }
}

async function restartWith(stored: unknown) {
  window.localStorage.setItem(TABS_KEY, JSON.stringify(stored))

  vi.resetModules()

  return import('./preview')
}

function tabIds(module: Awaited<ReturnType<typeof restartWith>>) {
  return module.$previewTabs.get().map(tab => tab.id)
}

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
  vi.resetModules()
})

describe('preview tabs restart recovery', () => {
  it('restores the default bucket in order and does not duplicate it when the default scope arrives', async () => {
    const stored = {
      default: [
        urlTab('browser-one', 'https://one.example'),
        fileTab('/work/restored.html'),
        urlTab('browser-two', 'https://two.example')
      ]
    }

    const preview = await restartWith(stored)

    expect(tabIds(preview)).toEqual(['url:browser-one', 'file:file:///work/restored.html', 'url:browser-two'])
    expect(preview.$previewTabs.get().map(tab => tab.target.url)).toEqual([
      'https://one.example',
      'file:///work/restored.html',
      'https://two.example'
    ])

    preview.setPreviewScope('default')

    expect(tabIds(preview)).toEqual(['url:browser-one', 'file:file:///work/restored.html', 'url:browser-two'])
  })

  it.each([
    ['profile buckets', { default: [fileTab('/work/default.html')] }],
    ['a legacy bare array', [fileTab('/work/legacy.html')]]
  ])('does not write the tab storage key while importing %s', async (_format, stored) => {
    window.localStorage.setItem(TABS_KEY, JSON.stringify(stored))
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    const removeItem = vi.spyOn(Storage.prototype, 'removeItem')

    vi.resetModules()
    await import('./preview')

    expect(setItem.mock.calls.filter(([key]) => key === TABS_KEY)).toEqual([])
    expect(removeItem.mock.calls.filter(([key]) => key === TABS_KEY)).toEqual([])
  })

  it('adopts a legacy bare array when default is the first scope, once only', async () => {
    const preview = await restartWith([fileTab('/work/legacy.html')])

    expect(tabIds(preview)).toEqual([])

    preview.setPreviewScope('default')
    expect(tabIds(preview)).toEqual(['file:file:///work/legacy.html'])

    preview.setPreviewScope('other')
    preview.setPreviewScope('default')
    expect(tabIds(preview)).toEqual(['file:file:///work/legacy.html'])
    expect(JSON.parse(window.localStorage.getItem(TABS_KEY) ?? '{}')).toEqual({ default: [fileTab('/work/legacy.html')] })
  })

  it('adopts a legacy bare array into the first non-default scope without leaking it after restart', async () => {
    const preview = await restartWith([fileTab('/work/legacy.html')])

    preview.setPreviewScope('tess')
    expect(tabIds(preview)).toEqual(['file:file:///work/legacy.html'])

    preview.setPreviewScope('default')
    expect(tabIds(preview)).toEqual([])

    vi.resetModules()
    const restarted = await import('./preview')
    restarted.setPreviewScope('tess')

    expect(tabIds(restarted)).toEqual(['file:file:///work/legacy.html'])
    restarted.setPreviewScope('default')
    expect(tabIds(restarted)).toEqual([])
  })

  it('continues to persist restored tabs through updates and removes the key after the final close', async () => {
    const preview = await restartWith({ default: [urlTab('browser-one', 'https://one.example')] })

    preview.openPreview(fileTab('/work/new.html').target)
    expect(window.localStorage.getItem(TABS_KEY)).toContain('/work/new.html')

    preview.commitBrowserTabLocation('url:browser-one', 'https://two.example', 'Two')
    expect(window.localStorage.getItem(TABS_KEY)).toContain('https://two.example')

    preview.closeRightRailTab('url:browser-one')
    expect(window.localStorage.getItem(TABS_KEY)).toContain('/work/new.html')

    preview.closeRightRailTab('file:file:///work/new.html')
    expect(window.localStorage.getItem(TABS_KEY)).toBeNull()
  })

  it('does not revive renamed or dropped profile buckets after a restart', async () => {
    const preview = await restartWith({
      tess: [fileTab('/work/tess.html')],
      other: [fileTab('/work/other.html')]
    })

    preview.setPreviewScope('tess')
    preview.migratePreviewTabsForProfile('tess', 'tess-renamed')
    preview.setPreviewScope('other')
    preview.dropPreviewTabsForProfile('tess-renamed')

    vi.resetModules()
    const restarted = await import('./preview')

    restarted.setPreviewScope('tess')
    expect(tabIds(restarted)).toEqual([])
    restarted.setPreviewScope('tess-renamed')
    expect(tabIds(restarted)).toEqual([])
    restarted.setPreviewScope('other')
    expect(tabIds(restarted)).toEqual(['file:file:///work/other.html'])
  })
})
