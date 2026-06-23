import { describe, expect, it } from 'vitest'

import { formatBackendError } from './format-error'

describe('formatBackendError', () => {
  it('unwraps Electron IPC and FastAPI detail envelopes', () => {
    const error = new Error(
      'Error invoking remote method \'hermes:api\': Error: 400: {"detail":"Profile name \'test\' is reserved"}'
    )

    expect(formatBackendError(error, 'Failed')).toBe("Profile name 'test' is reserved")
  })

  it('unwraps plain HTTP JSON error messages', () => {
    expect(formatBackendError('400: {"detail":"No profile named coder"}', 'Failed')).toBe('No profile named coder')
  })

  it('keeps normal error messages readable', () => {
    expect(formatBackendError(new Error('Error: Something failed'), 'Failed')).toBe('Something failed')
  })

  it('falls back for non-string errors', () => {
    expect(formatBackendError({ error: 'hidden' }, 'Failed')).toBe('Failed')
  })
})
