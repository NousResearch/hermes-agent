import { describe, expect, it } from 'vitest'

import { forceLoneHeaderForPanes } from './lone-header'

describe('forceLoneHeaderForPanes', () => {
  const chrome =
    (placement?: string, uncloseable = false) =>
    () => ({ placement, uncloseable })

  const noCollapse = () => false

  it('forces a header for session-tile ids even without registered chrome', () => {
    expect(forceLoneHeaderForPanes(['session-tile:abc'], () => ({}), noCollapse)).toBe(true)
  })

  it('forces a header for closeable placement:main panes', () => {
    expect(forceLoneHeaderForPanes(['workspace'], chrome('main', true), noCollapse)).toBe(false)
    expect(forceLoneHeaderForPanes(['some-page'], chrome('main', false), noCollapse)).toBe(true)
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
})
