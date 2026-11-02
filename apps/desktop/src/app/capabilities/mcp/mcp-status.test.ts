import { describe, expect, it } from 'vitest'

import type { McpTestResult } from '@/hermes'

import { canAuthenticate, statusOf } from './mcp-status'

const failure = (error: string): McpTestResult => ({ ok: false, error, tools: [] })

describe('MCP probe status', () => {
  it('keeps a stdio REA timeout separate from a Learn authentication failure', () => {
    const rea = { command: 'rea', args: ['mcp'] }
    const learn = { url: 'https://learn.microsoft.com/api/mcp' }

    const reaStatus = statusOf(
      rea,
      failure(
        "MCP server 'rea' test timed out after 60s (bounded by connect_timeout; an OAuth login also by oauth.timeout)"
      )
    )

    const learnStatus = statusOf(learn, failure('OAuth authorization required'))

    expect(reaStatus).toBe('error')
    expect(learnStatus).toBe('needs-auth')
    expect(canAuthenticate(rea, reaStatus)).toBe(false)
    expect(canAuthenticate(learn, learnStatus)).toBe(true)
  })

  it('preserves disabled, pending, unknown, and successful statuses', () => {
    expect(statusOf({ enabled: false }, failure('HTTP 401 Unauthorized'))).toBe('off')
    expect(statusOf({}, 'probing')).toBe('probing')
    expect(statusOf({}, undefined)).toBe('unknown')
    expect(statusOf({}, { ok: true, tools: [] })).toBe('ok')
  })
})
