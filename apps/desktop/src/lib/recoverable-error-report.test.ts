import type { ErrorInfo } from 'react'
import { describe, expect, it } from 'vitest'

import {
  buildRecoverableErrorReport,
  describeRecoverableError,
  RECOVERABLE_BOUNDARY
} from './recoverable-error-report'

const REACT_WRAPPER = 'There was an error during concurrent rendering but React was able to recover'

const lookupError = () => new Error('useClientLookup: Index 6 out of bounds (length: 2)')

describe('describeRecoverableError', () => {
  it('names the cause that actually threw, not just React recovery wrapper', () => {
    const message = describeRecoverableError(new Error(REACT_WRAPPER, { cause: lookupError() }))

    expect(message).toContain(REACT_WRAPPER)
    expect(message).toContain('← caused by: Error: useClientLookup: Index 6 out of bounds (length: 2)')
  })

  it('walks a nested cause chain in order', () => {
    const message = describeRecoverableError(
      new Error('outer', { cause: new Error('middle', { cause: new Error('inner') }) })
    )

    expect(message.indexOf('middle')).toBeLessThan(message.indexOf('inner'))
  })

  it('reports an error with no cause without inventing one', () => {
    expect(describeRecoverableError(new Error('lonely'))).toBe('Error: lonely')
  })

  it('survives a thrown non-Error value', () => {
    expect(describeRecoverableError('boom')).toBe('boom')
    expect(describeRecoverableError(undefined)).toBe('(undefined thrown)')
  })

  it('bounds the message even when the cause chain is long and chatty', () => {
    const message = describeRecoverableError(new Error('x'.repeat(5000), { cause: new Error('y'.repeat(5000)) }))

    expect(message.length).toBeLessThanOrEqual(2000)
  })
})

describe('buildRecoverableErrorReport', () => {
  it('marks the report as a concurrent-render recovery and keeps the component stack', () => {
    const report = buildRecoverableErrorReport(
      new Error(REACT_WRAPPER, { cause: lookupError() }),
      { componentStack: '\n    at Thread\n    at App\n' } as ErrorInfo,
      'main'
    )

    expect(report.boundary).toBe(RECOVERABLE_BOUNDARY)
    expect(report.label).toBe('main')
    expect(report.message).toContain('useClientLookup')
    expect(report.componentStack).toBe('at Thread\n    at App')
  })

  it('falls back to the main label and an empty stack when React supplies neither', () => {
    const report = buildRecoverableErrorReport(new Error('boom'), undefined, '')

    expect(report.label).toBe('main')
    expect(report.componentStack).toBe('')
  })

  it('clamps a hostile component stack', () => {
    const report = buildRecoverableErrorReport(new Error('boom'), { componentStack: 'a'.repeat(9000) } as ErrorInfo, 'main')

    expect(report.componentStack.length).toBeLessThanOrEqual(4000)
  })
})
