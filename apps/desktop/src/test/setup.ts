import { cleanup } from '@testing-library/react'
import cssEscape from 'css.escape'
import { afterEach } from 'vitest'

if (typeof globalThis.CSS === 'undefined') {
  Object.defineProperty(globalThis, 'CSS', {
    configurable: true,
    value: {}
  })
}

if (typeof globalThis.CSS.escape !== 'function') {
  Object.defineProperty(globalThis.CSS, 'escape', {
    configurable: true,
    value: cssEscape
  })
}

afterEach(cleanup)
