import { describe, expect, it } from 'vitest'
import { isExcludedPath, ALWAYS_EXCLUDED } from '@/lib/excluded-paths'

describe('isExcludedPath', () => {
  it('flags node_modules paths', () => {
    expect(isExcludedPath('node_modules/.bin/foo')).toBe(true)
    expect(isExcludedPath('node_modules')).toBe(true)
  })

  it('flags .git paths', () => {
    expect(isExcludedPath('.git/config')).toBe(true)
    expect(isExcludedPath('.git')).toBe(true)
  })

  it('allows normal paths', () => {
    expect(isExcludedPath('src/index.ts')).toBe(false)
    expect(isExcludedPath('README.md')).toBe(false)
  })

  it('handles Windows separators', () => {
    expect(isExcludedPath('node_modules\\.bin\\foo')).toBe(true)
  })

  it('flags .DS_Store', () => {
    expect(isExcludedPath('.DS_Store')).toBe(true)
  })
})
