import { afterEach, describe, expect, it, vi } from 'vitest'

import { reactRootErrorOptions } from './react-root-error-logging'

describe('reactRootErrorOptions', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('logs the wrapped cause and component stack for recoverable errors', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const cause = new TypeError("Cannot read properties of undefined (reading 'refText')")
    const wrapper = new Error('Minified React error #520', { cause })

    reactRootErrorOptions().onRecoverableError?.(wrapper, { componentStack: '\n  at Thread' })

    expect(spy).toHaveBeenCalledWith('[react:recoverable]', wrapper, 'cause:', cause, '\n  at Thread')
  })

  it('logs caught and uncaught errors without a cause', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const error = new Error('boom')
    const options = reactRootErrorOptions()

    options.onCaughtError?.(error, { componentStack: '\n  at App' })
    options.onUncaughtError?.(error, {})

    expect(spy).toHaveBeenNthCalledWith(1, '[react:caught]', error, '\n  at App')
    expect(spy).toHaveBeenNthCalledWith(2, '[react:uncaught]', error, '')
  })
})
