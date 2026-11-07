import { describe, expect, it } from 'vitest'

import { shortVersion } from './version-label'

describe('shortVersion', () => {
  it('keeps the distance past the release and drops the commit', (): void => {
    expect(shortVersion('0.21.5+1913.gf83a9e9')).toBe('0.21.5+1913')
    expect(shortVersion('v0.21.5+1913.gf83a9e9.dirty')).toBe('0.21.5+1913')
    expect(shortVersion('0.21.5+1913')).toBe('0.21.5+1913')
    expect(shortVersion('0.21.5')).toBe('0.21.5')
  })
})
