/** Focused tests for errText()'s error-body parsing. */

import { describe, expect, it } from 'vitest'

import { errText } from './ui'

describe('errText', () => {
  it('should extract a plain string detail from a JSON error body', () => {
    const err = new Error('400: {"detail":"board not found"}')

    expect(errText(err)).toBe('board not found')
  })

  it('should flatten a structured validation-error list into one line', () => {
    const err = new Error(
      '422: {"detail":[{"loc":["body","workspace_path"],"msg":"workspace_path or default_workdir required","type":"value_error"}]}'
    )

    expect(errText(err)).toBe('workspace_path or default_workdir required')
  })

  it('should join multiple validation-error messages with a separator', () => {
    const err = new Error('422: {"detail":[{"msg":"first problem"},{"msg":"second problem"}]}')

    expect(errText(err)).toBe('first problem; second problem')
  })

  it('should fall back to the raw message when the body is not JSON', () => {
    const err = new Error('500: internal server error')

    expect(errText(err)).toBe('500: internal server error')
  })

  it('should fall back to the raw message when a non-Error value is thrown', () => {
    expect(errText('boom')).toBe('boom')
  })
})
