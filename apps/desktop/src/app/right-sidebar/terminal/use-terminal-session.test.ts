import { describe, expect, it } from 'vitest'

import { cleanReviveSnapshot, isLegacyDuplicatedIdleBuffer } from './use-terminal-session'

describe('cleanReviveSnapshot', () => {
  it('trims a trailing idle prompt following a blank-line-separated block', () => {
    const buffer = 'PS C:\\repo> echo test\r\ntest\r\n\r\nPS C:\\repo>'
    expect(cleanReviveSnapshot(buffer)).toBe('PS C:\\repo> echo test\r\ntest')
  })

  it('leaves a buffer untouched when there is no blank-line separator', () => {
    const buffer = 'PS C:\\repo> echo test\r\ntest\r\nPS C:\\repo>'
    expect(cleanReviveSnapshot(buffer)).toBe(buffer)
  })

  it('handles an empty buffer', () => {
    expect(cleanReviveSnapshot('')).toBe('')
  })
})

describe('isLegacyDuplicatedIdleBuffer', () => {
  it('flags a buffer that is nothing but one idle prompt', () => {
    expect(isLegacyDuplicatedIdleBuffer('PS C:\\Users\\Aleksandr>')).toBe(true)
  })

  it('flags a buffer with the same prompt duplicated across relaunches', () => {
    const buffer = 'PS C:\\Users\\Aleksandr>\r\nPS C:\\Users\\Aleksandr>\r\nPS C:\\Users\\Aleksandr>'
    expect(isLegacyDuplicatedIdleBuffer(buffer)).toBe(true)
  })

  it('flags an empty or purely blank buffer', () => {
    expect(isLegacyDuplicatedIdleBuffer('')).toBe(true)
    expect(isLegacyDuplicatedIdleBuffer('\r\n\r\n')).toBe(true)
  })

  it('does not flag a buffer that holds real command output', () => {
    const buffer = 'PS C:\\repo> echo test\r\ntest\r\nPS C:\\repo>'
    expect(isLegacyDuplicatedIdleBuffer(buffer)).toBe(false)
  })

  it('does not flag a buffer where the prompt changed (e.g. after cd)', () => {
    const buffer = 'PS C:\\Users\\Aleksandr> cd repo\r\nPS C:\\Users\\Aleksandr\\repo>'
    expect(isLegacyDuplicatedIdleBuffer(buffer)).toBe(false)
  })
})
