import { afterEach, expect, it } from 'vitest'

import { $widthClass } from '@/app/redesign/breakpoints'
import { $narrowViewport } from '@/components/pane-shell/tree/store'
import { $shellChoice } from '@/store/shell-variant'

import { host } from './index'

afterEach(() => {
  $shellChoice.set(null)
  $widthClass.set('wide')
})

// Plugins have read `host.state.viewport.narrow` as "below the 640px sidebar dock point" since the
// contract shipped. The redesign sheets its sidebar from 1024px down, but that is its own chrome.
it.each([
  ['classic', false],
  ['redesign', true]
])('reports narrow only below the dock point in %s', (_shell, portal) => {
  $shellChoice.set(portal)

  $widthClass.set('medium')
  expect(host.state.viewport.get().narrow).toBe(false)
  // The pane tree's own posture still differs by shell.
  expect($narrowViewport.get()).toBe(portal)

  $widthClass.set('compact')
  expect(host.state.viewport.get().narrow).toBe(true)

  $widthClass.set('wide')
  expect(host.state.viewport.get().narrow).toBe(false)
})
