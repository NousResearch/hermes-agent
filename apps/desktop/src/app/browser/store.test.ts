import { beforeEach, describe, expect, it } from 'vitest'

import {
  BROWSER_MAX_TABS,
  BROWSER_QC_EVIDENCE_MAX_LENGTH,
  BROWSER_QC_NOTE_MAX_LENGTH,
  BROWSER_TAB_LIMIT_ERROR_CODE,
  BROWSER_TITLE_MAX_LENGTH,
  BROWSER_UNSUPPORTED_URL_ERROR_CODE,
  BROWSER_URL_MAX_LENGTH,
  browserStorageKey,
  BrowserTabLimitError,
  BrowserUnsupportedUrlError,
  browserWindowScope,
  createBrowserStore,
  sanitizeBrowserState,
  sanitizeBrowserUrl
} from './store'

describe('browser store', () => {
  beforeEach(() => window.localStorage.clear())

  it('caps persisted text and redacts data/blob URLs and capture bytes', () => {
    const inlineImage = `data:image/png;base64,${'a'.repeat(4_000)}`

    const state = sanitizeBrowserState({
      activeTabId: 'tab-1',
      capture: { dataUrl: inlineImage },
      tabs: [
        {
          id: 'tab-1',
          url: inlineImage,
          title: 't'.repeat(BROWSER_TITLE_MAX_LENGTH + 1),
          qc: {
            composition: {
              status: 'pass',
              note: 'n'.repeat(BROWSER_QC_NOTE_MAX_LENGTH + 1),
              evidence: 'e'.repeat(BROWSER_QC_EVIDENCE_MAX_LENGTH + 1)
            },
            color: { status: 'fail', note: '', evidence: 'proof=data:image/png;base64,capture' }
          }
        },
        {
          id: 'tab-2',
          url: `https://example.test/${'u'.repeat(BROWSER_URL_MAX_LENGTH + 1)}`,
          title: 'second'
        }
      ]
    })

    expect(state.tabs[0]).toMatchObject({ url: '', title: 't'.repeat(BROWSER_TITLE_MAX_LENGTH) })
    expect(state.tabs[0].qc.composition).toEqual({
      status: 'pass',
      note: 'n'.repeat(BROWSER_QC_NOTE_MAX_LENGTH),
      evidence: 'e'.repeat(BROWSER_QC_EVIDENCE_MAX_LENGTH)
    })
    expect(state.tabs[1].url).toHaveLength(BROWSER_URL_MAX_LENGTH)
    expect(JSON.stringify(state)).not.toContain('data:image')
    expect(JSON.stringify(state)).not.toContain('capture')
  })
  it('permits only Browser-partition URL protocols and transient raster image data URLs', () => {
    expect(sanitizeBrowserUrl('https://example.test')).toBe('https://example.test')
    expect(sanitizeBrowserUrl('file:///tmp/reference.png')).toBe('')
    expect(sanitizeBrowserUrl('file://fileserver/share/reference.png')).toBe('')
    expect(sanitizeBrowserUrl('\\\\fileserver\\share\\reference.png')).toBe('')
    expect(sanitizeBrowserUrl('about:blank')).toBe('about:blank')
    expect(sanitizeBrowserUrl('data:image/png;base64,bytes')).toBe('')
    expect(sanitizeBrowserUrl('blob:https://example.test/preview')).toBe('')

    const largeImage = `data:image/png;base64,${'a'.repeat(BROWSER_URL_MAX_LENGTH + 1)}`
    const store = createBrowserStore('runtime-url-policy')
    store.openBrowserSurface({ url: largeImage, pinned: true })
    expect(store.$browserState.get().tabs[0]?.url).toBe(largeImage)
    expect(store.$browserState.get().tabs[0]?.pinned).toBe(true)
    expect(window.localStorage.getItem(browserStorageKey('runtime-url-policy', 'state'))).not.toContain(largeImage)
    store.updateBrowserTab(store.$browserState.get().tabs[0]?.id ?? '', {
      url: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"/>'
    })
    expect(store.$browserState.get().tabs[0]?.url).toBe('')
    store.updateBrowserTab(store.$browserState.get().tabs[0]?.id ?? '', { url: 'data:text/html,not-allowed' })
    expect(store.$browserState.get().tabs[0]?.url).toBe('')
    store.updateBrowserTab(store.$browserState.get().tabs[0]?.id ?? '', { url: 'file:///tmp/reference.png' })
    expect(store.$browserState.get().tabs[0]?.url).toBe('')
  })
  it.each([
    'file:///tmp/reference.png',
    'blob:https://example.test/preview',
    'data:image/svg+xml,<svg/>',
    'data:text/html,not-allowed',
    'javascript:alert(1)',
    'not a URL'
  ])('rejects unsupported explicit surface and QC URLs without opening or focusing a blank tab: %s', url => {
    const store = createBrowserStore(`unsupported-url-${url}`)
    const tabId = store.addBrowserTab({ url: 'https://example.test/existing' })
    store.setBrowserOpen(false)
    const before = store.$browserState.get()

    expect(() => store.openBrowserSurface({ url })).toThrow(
      expect.objectContaining({ code: BROWSER_UNSUPPORTED_URL_ERROR_CODE, name: 'BrowserUnsupportedUrlError' })
    )
    expect(() => store.openBrowserQc({ url })).toThrow(BrowserUnsupportedUrlError)
    expect(store.$browserOpen.get()).toBe(false)
    expect(store.$browserRevealRequest.get()).toBe(0)
    expect(store.$browserQcRevealRequest.get()).toBe(0)
    expect(store.$browserState.get()).toEqual(before)
    expect(store.$browserState.get().activeTabId).toBe(tabId)
  })

  it('keeps blank New Tab behavior when the URL is absent or empty', () => {
    const store = createBrowserStore('blank-new-tab')

    expect(store.openBrowserSurface()).toBeTruthy()
    expect(store.openBrowserSurface({ url: '   ' })).toBeTruthy()
    expect(store.$browserState.get().tabs).toHaveLength(2)
    expect(store.$browserState.get().tabs.every(tab => tab.url === '')).toBe(true)
  })

  it('removes file and UNC URLs from persisted browser state', () => {
    const state = sanitizeBrowserState({
      activeTabId: 'file',
      tabs: [
        { id: 'file', url: 'file:///tmp/reference.png' },
        { id: 'unc', url: 'file://fileserver/share/reference.png' }
      ]
    })

    expect(state.tabs).toMatchObject([
      { id: 'file', url: '' },
      { id: 'unc', url: '' }
    ])
  })

  it('keeps presentation scoped to its renderer window key', () => {
    const primary = createBrowserStore('primary-test')
    const secondary = createBrowserStore('secondary-test')

    primary.openBrowserSurface({ url: 'https://primary.example' })
    secondary.openBrowserSurface({ url: 'https://secondary.example' })

    expect(primary.$browserState.get().tabs[0]?.url).toBe('https://primary.example')
    expect(secondary.$browserState.get().tabs[0]?.url).toBe('https://secondary.example')
    expect(browserStorageKey('primary-test', 'state')).not.toBe(browserStorageKey('secondary-test', 'state'))
    expect(browserWindowScope({ search: '?win=secondary', hash: '#/session-one' }, 'window-a')).toBe(
      'secondary.session.session-one'
    )
    expect(browserWindowScope({ search: '?win=secondary', hash: '#/session-two' }, 'window-b')).toBe(
      'secondary.session.session-two'
    )
    expect(browserWindowScope({ search: '?win=secondary&new=1', hash: '#/' }, 'window-a')).toBe(
      'secondary.new.window-a'
    )
    expect(browserWindowScope({ search: '?win=secondary&new=1', hash: '#/' }, 'window-b')).toBe(
      'secondary.new.window-b'
    )
    expect(browserWindowScope({ search: '', hash: '#/' }, 'window-a')).toBe('primary')
  })

  it('reuses an existing tab for the same normalized URL at the tab cap', () => {
    const store = createBrowserStore('tab-reuse')

    const ids = Array.from({ length: BROWSER_MAX_TABS }, (_, index) =>
      store.addBrowserTab({ url: `https://example.test/${index}` })
    )

    const reusedId = store.addBrowserTab({ url: ' https://example.test/3 ' })

    expect(reusedId).toBe(ids[3])
    expect(store.$browserState.get().activeTabId).toBe(ids[3])
    expect(store.$browserState.get().tabs).toHaveLength(BROWSER_MAX_TABS)
  })

  it('replaces the oldest unpinned tab at the tab cap while retaining pinned tabs', () => {
    const store = createBrowserStore('tab-eviction')
    const firstId = store.addBrowserTab({ title: 'first', url: 'https://example.test/first' })
    const pinnedId = store.addBrowserTab({ pinned: true, title: 'pinned', url: 'https://example.test/pinned' })

    Array.from({ length: BROWSER_MAX_TABS - 2 }, (_, index) =>
      store.addBrowserTab({ title: `later-${index}`, url: `https://example.test/later-${index}` })
    )
    const replacementId = store.addBrowserTab({ title: 'replacement', url: 'https://example.test/replacement' })
    const state = store.$browserState.get()

    expect(replacementId).not.toBe(firstId)
    expect(state.activeTabId).toBe(replacementId)
    expect(state.tabs).toHaveLength(BROWSER_MAX_TABS)
    expect(state.tabs.at(-1)).toMatchObject({ id: replacementId, title: 'replacement', pinned: false })
    expect(state.tabs.some(tab => tab.id === pinnedId && tab.pinned)).toBe(true)
  })

  it('fails explicitly without changing pinned tabs when every tab is pinned', () => {
    const store = createBrowserStore('all-pinned')

    Array.from({ length: BROWSER_MAX_TABS }, (_, index) =>
      store.addBrowserTab({ pinned: true, title: String(index), url: `https://example.test/${index}` })
    )
    const before = store.$browserState.get()

    expect(() => store.addBrowserTab({ title: 'blocked', url: 'https://example.test/blocked' })).toThrow(
      BrowserTabLimitError
    )
    expect(store.$browserState.get()).toEqual(before)
  })
  it('exposes a stable code for all-pinned tab-limit failures', () => {
    const store = createBrowserStore('all-pinned-error-code')

    Array.from({ length: BROWSER_MAX_TABS }, (_, index) =>
      store.addBrowserTab({ pinned: true, title: String(index), url: `https://example.test/${index}` })
    )

    expect(() => store.addBrowserTab({ url: 'https://example.test/blocked' })).toThrow(
      expect.objectContaining({ code: BROWSER_TAB_LIMIT_ERROR_CODE, name: 'BrowserTabLimitError' })
    )
  })

  it('initializes every QC dimension unchecked and persists targeted updates', () => {
    const store = createBrowserStore('qc')
    const tabId = store.openBrowserQc({ url: 'https://example.test' })
    expect(tabId).toBeTruthy()

    const tab = store.$browserState.get().tabs[0]
    expect(store.$browserState.get().qcOpen).toBe(true)
    expect(store.$browserQcRevealRequest.get()).toBe(1)
    expect(Object.values(tab.qc)).toEqual(
      Array.from({ length: 7 }, () => ({ status: 'unchecked', note: '', evidence: '' }))
    )

    store.updateBrowserQc(tab.id, 'contrast', { status: 'fail', note: 'Text is too faint', evidence: 'header label' })
    expect(store.$browserState.get().tabs[0]?.qc.contrast).toEqual({
      status: 'fail',
      note: 'Text is too faint',
      evidence: 'header label'
    })
    expect(store.$browserState.get().tabs[0]?.qc.color.status).toBe('unchecked')
  })
  it('preserves QC whitespace during runtime updates', () => {
    const store = createBrowserStore('qc-runtime-whitespace')
    const tabId = store.openBrowserQc({ url: 'https://example.test' })

    expect(tabId).toBeTruthy()
    store.updateBrowserQc(tabId!, 'composition', { note: ' Leading ' })
    store.updateBrowserQc(tabId!, 'composition', { evidence: ' Fixture proof ' })

    expect(store.$browserState.get().tabs[0]?.qc.composition).toMatchObject({
      note: ' Leading ',
      evidence: ' Fixture proof '
    })
  })
  it('bounds and redacts runtime QC text', () => {
    const store = createBrowserStore('qc-runtime-sanitizer')
    const tabId = store.openBrowserQc({ url: 'https://example.test' })

    store.updateBrowserQc(tabId!, 'composition', {
      note: 'n'.repeat(BROWSER_QC_NOTE_MAX_LENGTH + 1),
      evidence: 'proof=data:image/png;base64,bytes'
    })

    expect(store.$browserState.get().tabs[0]?.qc.composition).toMatchObject({
      note: 'n'.repeat(BROWSER_QC_NOTE_MAX_LENGTH),
      evidence: ''
    })
  })

  it('opens only through explicit browser actions and keeps captures transient', () => {
    const store = createBrowserStore('explicit-open')

    expect(store.$browserOpen.get()).toBe(false)
    store.$browserState.set({ activeTabId: null, qcOpen: false, tabs: [] })
    expect(store.$browserOpen.get()).toBe(false)

    expect(store.$browserRevealRequest.get()).toBe(0)
    store.openBrowserSurface({ url: 'data:image/png;base64,transient-image-bytes' })
    expect(store.$browserOpen.get()).toBe(true)
    expect(store.$browserRevealRequest.get()).toBe(1)
    expect(store.$browserState.get().tabs[0]?.url).toContain('transient-image-bytes')
    expect(window.localStorage.getItem(browserStorageKey('explicit-open', 'state'))).not.toContain(
      'transient-image-bytes'
    )
    store.openBrowserQc()
    expect(store.$browserRevealRequest.get()).toBe(2)
    expect(store.$browserQcRevealRequest.get()).toBe(1)
    store.setBrowserCapture({
      captureId: 'capture-1',
      createdAt: 123,
      dataUrl: 'data:image/png;base64,capture-bytes',
      height: 720,
      tabId: store.$browserState.get().activeTabId,
      width: 1280
    })
    expect(store.$browserCapture.get()?.dataUrl).toContain('capture-bytes')
    expect(window.localStorage.getItem(browserStorageKey('explicit-open', 'state'))).not.toContain('capture-bytes')
    expect(store.$browserCapture.get()).toMatchObject({
      captureId: 'capture-1',
      createdAt: 123,
      height: 720,
      width: 1280
    })
    store.clearBrowserCapture()
    expect(store.$browserCapture.get()).toBeNull()
  })
})
