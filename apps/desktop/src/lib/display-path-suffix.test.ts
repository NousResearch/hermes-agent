import { describe, expect, it } from 'vitest'

import { displayPathSuffix } from './display-path'

describe('displayPathSuffix', () => {
  it.each([
    ['/home/person/.hermes/profiles/coder/cache/projects/example/.worktrees/checkout', '…/.worktrees/checkout'],
    ['C:\\Users\\person\\AppData\\Local\\cache\\projects\\example\\.worktrees\\checkout', '…/.worktrees/checkout'],
    ['/', '/'], ['C:\\', 'C:/'], ['/repo/task', '/repo/task'], ['C:\\src\\task', 'C:/src/task'],
    ['/home/person/repo', '~/repo'], ['', '']
  ])('keeps a useful suffix or intact short/root display for %s', (path, expected) => {
    expect(displayPathSuffix(path)).toBe(expected)
  })

  it('bounds a single long leaf from the leading side, not the checkout suffix', () => {
    const leaf = `${'long-'.repeat(30)}task-ui`
    const path = `/cache/${leaf}`
    const display = displayPathSuffix(path)
    expect(display.length).toBeLessThanOrEqual(34)
    expect(display.startsWith('…/')).toBe(true)
    expect(display.endsWith('task-ui')).toBe(true)
    expect(path.endsWith(display.slice(2))).toBe(true)
  })

  it('fits a compact menu without clipping a generated checkout identifier again', () => {
    const leaf = 'task-0123456789abcdef01234567'
    const display = displayPathSuffix(`/cache/projects/example/.worktrees/${leaf}`)
    expect(display).toBe(`…/${leaf}`)
    expect(display.length).toBeLessThanOrEqual(34)
  })

  it('also shortens medium-length paths that exceed the compact line budget', () => {
    const path = `/root/${'segment/'.repeat(4)}checkout`
    expect(displayPathSuffix(path)).toBe('…/segment/checkout')
  })
})
