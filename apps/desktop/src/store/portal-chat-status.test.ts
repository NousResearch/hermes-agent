import { afterEach, describe, expect, it } from 'vitest'

import { setInterfaceMode } from './interface-mode'
import { $portalChatStatusShown, setPortalChatStatusEnabled } from './portal-chat-status'
import { setPortalPreferred } from './shell-variant'

describe('the redesign’s chat status line', () => {
  afterEach(() => {
    setPortalChatStatusEnabled(true)
    setInterfaceMode('advanced')
    setPortalPreferred(false)
  })

  it('shows only in the redesigned shell with developer chrome, until switched off', () => {
    setInterfaceMode('advanced')
    setPortalPreferred(false)
    expect($portalChatStatusShown.get()).toBe(false)

    setPortalPreferred(true)
    expect($portalChatStatusShown.get()).toBe(true)

    setInterfaceMode('simple')
    expect($portalChatStatusShown.get()).toBe(false)

    setInterfaceMode('advanced')
    setPortalChatStatusEnabled(false)
    expect($portalChatStatusShown.get()).toBe(false)
  })
})
