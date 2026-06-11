import { describe, expect, it } from 'vitest'

import {
  normalizeProcessNotificationMode,
  processCompletionToast
} from './process-notifications'

describe('normalizeProcessNotificationMode', () => {
  it('maps false to off (config allows boolean false)', () => {
    expect(normalizeProcessNotificationMode(false)).toBe('off')
  })

  it('accepts known modes case-insensitively', () => {
    expect(normalizeProcessNotificationMode('All')).toBe('all')
    expect(normalizeProcessNotificationMode(' result ')).toBe('result')
    expect(normalizeProcessNotificationMode('ERROR')).toBe('error')
    expect(normalizeProcessNotificationMode('off')).toBe('off')
  })

  it('defaults unknown values to all (mirrors gateway/run.py)', () => {
    expect(normalizeProcessNotificationMode(undefined)).toBe('all')
    expect(normalizeProcessNotificationMode('sometimes')).toBe('all')
    expect(normalizeProcessNotificationMode(42)).toBe('all')
    expect(normalizeProcessNotificationMode(true)).toBe('all')
  })
})

describe('processCompletionToast', () => {
  const completion = {
    kind: 'process',
    event_type: 'completion',
    command: 'npm run build',
    exit_code: 0,
    text: '[IMPORTANT: Background process proc_1 completed]'
  }

  it('builds a success toast for a zero exit code', () => {
    expect(processCompletionToast(completion, 'all')).toEqual({
      title: 'Background process finished',
      body: 'npm run build'
    })
  })

  it('builds a failure toast with the exit code', () => {
    expect(processCompletionToast({ ...completion, exit_code: 2 }, 'all')).toEqual({
      title: 'Background process failed (exit 2)',
      body: 'npm run build'
    })
  })

  it('treats a payload without event_type as a completion (older gateways)', () => {
    expect(processCompletionToast({ ...completion, event_type: undefined }, 'result')).not.toBeNull()
  })

  it('returns null when mode is off', () => {
    expect(processCompletionToast(completion, 'off')).toBeNull()
  })

  it('in error mode only notifies for non-zero exit codes', () => {
    expect(processCompletionToast(completion, 'error')).toBeNull()
    expect(processCompletionToast({ ...completion, exit_code: 1 }, 'error')).not.toBeNull()
    expect(processCompletionToast({ ...completion, exit_code: null }, 'error')).toBeNull()
  })

  it('never notifies for watch matches (they can fire many times)', () => {
    expect(processCompletionToast({ ...completion, event_type: 'watch_match' }, 'all')).toBeNull()
  })

  it('falls back to the formatted text when command is empty, truncated', () => {
    const toast = processCompletionToast({ ...completion, command: '', text: 'x'.repeat(300) }, 'all')

    expect(toast?.body).toHaveLength(140)
  })
})
