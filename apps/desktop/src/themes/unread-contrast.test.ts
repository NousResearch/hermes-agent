import { describe, expect, it } from 'vitest'

import { contrastRatio } from './color'
import { getBaseColors, unreadBubbleColor, unreadStrokeColor } from './context'
import { BUILTIN_THEME_LIST } from './presets'

function sidebarBackground(colors: ReturnType<typeof getBaseColors>): string {
  return colors.sidebarBackground ?? colors.background
}

describe('unread session cue colors', () => {
  it('keeps the unread bubble as the preferred bright sky blue', () => {
    expect(unreadBubbleColor()).toBe('#38bdf8')
  })

  it.each(BUILTIN_THEME_LIST.flatMap(theme => [
    [`${theme.name} light`, getBaseColors(theme.name, 'light')] as const,
    [`${theme.name} dark`, getBaseColors(theme.name, 'dark')] as const
  ]))('chooses a visible stroke/ring color for %s', (_label, colors) => {
    const stroke = unreadStrokeColor(colors)

    expect(contrastRatio(stroke, sidebarBackground(colors))).toBeGreaterThanOrEqual(3)
  })
})
