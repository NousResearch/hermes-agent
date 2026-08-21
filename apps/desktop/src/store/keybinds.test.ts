import { afterEach, describe, expect, it } from 'vitest'

import { $comboIndex, resetAllBindings, setBinding } from './keybinds'

describe('$comboIndex', () => {
  afterEach(() => resetAllBindings())

  it('leaves busy composer Enter out of the global index', () => {
    expect($comboIndex.get().get('enter')).toBe('composer.focus')
    expect($comboIndex.get().get('mod+enter')).not.toBe('composer.queue')
    expect($comboIndex.get().get('escape')).not.toBe('composer.cancel')
  })

  it('keeps rebound busy composer actions out of the global index', () => {
    setBinding('composer.steer', ['mod+alt+s'])

    expect($comboIndex.get().get('mod+alt+s')).not.toBe('composer.steer')
  })
})
