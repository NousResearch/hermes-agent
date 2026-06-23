import { describe, expect, it } from 'vitest'

import { sidebarSessionsSectionRootClassName } from './classes'

describe('sidebarSessionsSectionRootClassName', () => {
  it('keeps the flexing recents layout while expanded', () => {
    expect(
      sidebarSessionsSectionRootClassName(
        'min-h-32 flex-1 overflow-hidden p-0 compact:min-h-0 compact:flex-none compact:overflow-visible',
        true
      )
    ).toBe('min-h-32 flex-1 overflow-hidden p-0 compact:min-h-0 compact:flex-none compact:overflow-visible')
  })

  it('reclaims sidebar space when a flexing sessions section is collapsed', () => {
    const className = sidebarSessionsSectionRootClassName(
      'min-h-32 flex-1 overflow-hidden p-0 compact:min-h-0 compact:flex-none compact:overflow-visible',
      false
    )

    expect(className).toContain('min-h-0')
    expect(className).toContain('flex-none')
    expect(className).toContain('shrink-0')
    expect(className).toContain('overflow-visible')
    expect(className).toContain('p-0')
    expect(className).not.toContain('min-h-32')
    expect(className).not.toContain('flex-1')
    expect(className).not.toContain('overflow-hidden')
  })

  it('keeps non-flex collapsed sections header-sized', () => {
    const className = sidebarSessionsSectionRootClassName('shrink-0 p-0', false)

    expect(className).toContain('shrink-0')
    expect(className).toContain('p-0')
    expect(className).toContain('flex-none')
    expect(className).not.toContain('flex-1')
    expect(className).not.toContain('min-h-32')
  })
})
