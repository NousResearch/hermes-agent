import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ApprovalResolutionCard, approvalResolutionFromResult } from './approval-resolution'

afterEach(cleanup)

describe('approvalResolutionFromResult', () => {
  it('accepts the durable live/history result shape', () => {
    expect(
      approvalResolutionFromResult(
        JSON.stringify({
          output: 'ok',
          approval: {
            status: 'approved',
            choice: 'once',
            actor: 'authenticated_user',
            resolved_at: 123.5,
            request_id: 'opaque-request'
          },
          execution: { status: 'succeeded' }
        })
      )
    ).toEqual({
      approval: {
        status: 'approved',
        choice: 'once',
        actor: 'authenticated_user',
        resolved_at: 123.5,
        request_id: 'opaque-request'
      },
      execution: { status: 'succeeded' }
    })
  })

  it('keeps rejection distinct from a technical approval failure', () => {
    expect(approvalResolutionFromResult({ approval: { status: 'rejected' }, execution: { status: 'not_attempted' } })).toEqual({
      approval: { status: 'rejected', choice: undefined, actor: undefined, resolved_at: undefined, request_id: undefined },
      execution: { status: 'not_attempted' }
    })
    expect(approvalResolutionFromResult({ approval: { status: 'unavailable' }, execution: { status: 'not_attempted' } })).not.toEqual(
      approvalResolutionFromResult({ approval: { status: 'rejected' }, execution: { status: 'not_attempted' } })
    )
  })

  it('ignores untrusted or incomplete result metadata', () => {
    expect(approvalResolutionFromResult({ approval: { status: 'unknown', actor: 'model' } })).toBeNull()
  })

  it.each([
    ['once', 'Approved once'],
    ['session', 'Approved for this session'],
    ['always', 'Approved always']
  ] as const)('renders the %s approval choice label without actionable controls', (choice, label) => {
    render(
      <ApprovalResolutionCard
        resolution={{
          approval: { status: 'approved', choice, actor: 'authenticated_user' },
          execution: { status: 'succeeded' }
        }}
      />
    )

    expect(screen.getByText(label)).toBeTruthy()
    if (choice === 'always') {
      expect(screen.queryByText('Approved once')).toBeNull()
    }
    expect(screen.getByText(/Execution succeeded/)).toBeTruthy()
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it.each([
    ['rejected', 'Rejected'],
    ['unavailable', 'Approval unavailable']
  ] as const)('keeps the %s approval label unchanged', (status, label) => {
    render(<ApprovalResolutionCard resolution={{ approval: { status } }} />)

    expect(screen.getByText(label)).toBeTruthy()
  })
})
