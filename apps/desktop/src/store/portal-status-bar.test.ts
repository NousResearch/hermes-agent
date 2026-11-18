import { afterEach, describe, expect, it } from 'vitest'

import { setInterfaceMode } from './interface-mode'
import { $portalStatusBarOpen, setPortalStatusBarOpen } from './portal-status-bar'
// setPortalPreferred arrives with the Settings toggle; until then these cases set the stored choice.
import { $shellChoice } from './shell-variant'

describe('the redesign’s status bar', () => {
  afterEach(() => {
    setInterfaceMode('advanced')
    setPortalStatusBarOpen(true)
    $shellChoice.set(false)
  })

  it('is open in the redesigned shell with developer chrome, until collapsed', () => {
    setInterfaceMode('advanced')
    $shellChoice.set(false)
    expect($portalStatusBarOpen.get()).toBe(false)

    $shellChoice.set(true)
    expect($portalStatusBarOpen.get()).toBe(true)

    setPortalStatusBarOpen(false)
    expect($portalStatusBarOpen.get()).toBe(false)
  })

  it('rests collapsed in Simple, where opening it never rewrites the advanced choice', () => {
    $shellChoice.set(true)
    setPortalStatusBarOpen(false)
    setInterfaceMode('simple')
    expect($portalStatusBarOpen.get()).toBe(false)

    setPortalStatusBarOpen(true)
    expect($portalStatusBarOpen.get()).toBe(true)

    setInterfaceMode('advanced')
    expect($portalStatusBarOpen.get()).toBe(false)
  })
})
