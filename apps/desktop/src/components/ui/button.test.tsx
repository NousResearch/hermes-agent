import { describe, expect, it } from 'vitest'

import { buttonVariants } from './button'

describe('buttonVariants', () => {
  it('keeps the hover lift for regular surface buttons', () => {
    expect(buttonVariants({ variant: 'ghost' })).toContain('hover:-translate-y-px')
  })

  it('can keep transform-positioned controls anchored', () => {
    expect(buttonVariants({ motion: 'none', variant: 'ghost' })).not.toContain('hover:-translate-y-px')
  })
})
