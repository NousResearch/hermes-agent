import { beforeEach, describe, expect, it } from 'vitest'

import {
  $titlebarExternalButtons,
  setTitlebarExternalButtons,
  TITLEBAR_EXTERNAL_BUTTONS_DEFAULT,
  TITLEBAR_EXTERNAL_BUTTONS_MAX
} from './titlebar-external-buttons'

describe('$titlebarExternalButtons', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setTitlebarExternalButtons(TITLEBAR_EXTERNAL_BUTTONS_DEFAULT)
  })

  it('reserves nothing by default', () => {
    expect($titlebarExternalButtons.get()).toBe(0)
  })

  it('persists the configured count', () => {
    setTitlebarExternalButtons(2)
    expect($titlebarExternalButtons.get()).toBe(2)
    expect(window.localStorage.getItem('hermes.desktop.titlebarExternalButtons')).toBe('2')
  })

  it('clamps to the supported range', () => {
    setTitlebarExternalButtons(99)
    expect($titlebarExternalButtons.get()).toBe(TITLEBAR_EXTERNAL_BUTTONS_MAX)
    setTitlebarExternalButtons(-1)
    expect($titlebarExternalButtons.get()).toBe(0)
  })
})
