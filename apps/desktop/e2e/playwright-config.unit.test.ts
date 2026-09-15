import { describe, expect, it } from 'vitest'

import config from '../playwright.config'

describe('desktop Playwright configuration', () => {
  it('runs real Electron stacks serially', () => {
    expect(config.workers).toBe(1)
    expect(config.fullyParallel).toBe(false)
  })
})
