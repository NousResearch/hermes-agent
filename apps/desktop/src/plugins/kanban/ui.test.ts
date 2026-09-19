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

  it('should fall back to the raw message when detail is an object', () => {
    const err = new Error('422: {"detail":{"msg":"nested"}}')

    expect(errText(err)).toBe('422: {"detail":{"msg":"nested"}}')
  })

  it('should fall back to the raw message when detail is a number', () => {
    const err = new Error('500: {"detail":42}')

    expect(errText(err)).toBe('500: {"detail":42}')
  })

  it('should fall back to the raw message when detail is null', () => {
    const err = new Error('500: {"detail":null}')

    expect(errText(err)).toBe('500: {"detail":null}')
  })

  it('should keep only string msg entries from a mixed detail list', () => {
    const err = new Error('422: {"detail":[null,42,{"msg":"real problem"},{"msg":{"nested":true}}]}')

    expect(errText(err)).toBe('real problem')
  })

  it('should fall back to the raw message when no detail entry has a usable msg', () => {
    const err = new Error('422: {"detail":[null,42]}')

    expect(errText(err)).toBe('422: {"detail":[null,42]}')
  })
})
