import { describe, expect, it } from 'vitest'

import { forceLoneHeaderForPanes } from './lone-header'

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
