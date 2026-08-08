import { describe, expect, it } from 'vitest'

import { forceLoneHeaderForPanes, hasCloseableMainTile, resolveZoneHeaderHidden } from './lone-header'

describe('forceLoneHeaderForPanes', () => {
  const chrome =
    (placement?: string, uncloseable = false) =>
    () => ({ placement, uncloseable })

  const noCollapse = () => false

  // Every mirrored tile (session / page / preview) is a closeable `main` pane, so
  // dragging one into a zone of its own must keep its tab — it used to strand a
  // preview headerless, with nothing to grab and no ✕.
  it('forces a header for closeable placement:main panes', () => {
    expect(forceLoneHeaderForPanes(['preview-tile:url:x'], chrome('main'), noCollapse)).toBe(true)
    expect(forceLoneHeaderForPanes(['session-tile:abc'], chrome('main'), noCollapse)).toBe(true)
  })

  it('forces a header for a lone collapse tool pane', () => {
    expect(
      forceLoneHeaderForPanes(
        ['terminal'],
        () => ({}),
        id => id === 'terminal'
      )
    ).toBe(true)
  })

  it('leaves a lone uncloseable workspace headerless', () => {
    expect(forceLoneHeaderForPanes(['workspace'], chrome('main', true), noCollapse)).toBe(false)
  })

  it('leaves standing side chrome (files / sessions) headerless', () => {
    expect(forceLoneHeaderForPanes(['files'], chrome('right'), noCollapse)).toBe(false)
  })
})

describe('resolveZoneHeaderHidden', () => {
  const mainChrome = () => ({ placement: 'main' as const, uncloseable: false })
  const workspaceChrome = () => ({ placement: 'main' as const, uncloseable: true })

  it('keeps the strip visible when a sticky hide meets a closeable preview', () => {
    const shown = ['workspace', 'preview-tile:url:x']
    const chromeOf = (id: string) => (id === 'workspace' ? workspaceChrome() : mainChrome())

    expect(
      resolveZoneHeaderHidden({
        forceLoneHeader: forceLoneHeaderForPanes(shown, chromeOf, () => false),
        hasCloseableMainTile: hasCloseableMainTile(shown, chromeOf),
        headerHiddenFlag: true,
        shownLength: shown.length
      })
    ).toBe(false)
  })

  it('keeps the strip visible for a lone Browser/preview tile even when sticky-hidden', () => {
    const shown = ['preview-tile:url:x']
    const chromeOf = mainChrome

    expect(
      resolveZoneHeaderHidden({
        forceLoneHeader: forceLoneHeaderForPanes(shown, chromeOf, () => false),
        hasCloseableMainTile: hasCloseableMainTile(shown, chromeOf),
        headerHiddenFlag: true,
        shownLength: shown.length
      })
    ).toBe(false)
  })

  it('still honors sticky hide for a tool-only zone', () => {
    const shown = ['terminal', 'logs']

    expect(
      resolveZoneHeaderHidden({
        forceLoneHeader: forceLoneHeaderForPanes(shown, () => ({}), id => id === 'terminal' || id === 'logs'),
        hasCloseableMainTile: false,
        headerHiddenFlag: true,
        shownLength: shown.length
      })
    ).toBe(true)
  })

  it('still auto-hides a lone uncloseable workspace', () => {
    const shown = ['workspace']
    const chromeOf = workspaceChrome

    expect(
      resolveZoneHeaderHidden({
        forceLoneHeader: forceLoneHeaderForPanes(shown, chromeOf, () => false),
        hasCloseableMainTile: hasCloseableMainTile(shown, chromeOf),
        shownLength: shown.length
      })
    ).toBe(true)
  })

  it('still honors headerVeto for full-page views', () => {
    const shown = ['preview-tile:url:x']
    const chromeOf = mainChrome

    expect(
      resolveZoneHeaderHidden({
        forceLoneHeader: true,
        hasCloseableMainTile: hasCloseableMainTile(shown, chromeOf),
        headerVeto: true,
        shownLength: shown.length
      })
    ).toBe(true)
  })
})
