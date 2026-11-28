import { describe, expect, it } from 'vitest'

import { physicalSidebarControls } from './titlebar-sidebar-placement'

describe('physical sidebar control placement', () => {
  it.each([false, true])('keeps the edge, shortcut and tour on the same physical side (RTL=%s)', rtl => {
    const logical = { left: { id: 'first' }, right: { id: 'last' } }
    const controls = physicalSidebarControls(logical, rtl)

    expect(controls.left).toBe(rtl ? logical.right : logical.left)
    expect(controls.right).toBe(rtl ? logical.left : logical.right)
    expect(controls.leftActionId).toBe(rtl ? 'view.toggleRightSidebar' : 'view.toggleSidebar')
    expect(controls.rightActionId).toBe(rtl ? 'view.toggleSidebar' : 'view.toggleRightSidebar')
    expect(controls.leftTour).toBe(rtl ? 'right-pane-toggle' : undefined)
    expect(controls.rightTour).toBe(rtl ? undefined : 'right-pane-toggle')
  })
})
