import { beforeEach, describe, expect, it } from 'vitest'

import { setShowReasoningFromConfig } from '@/store/reasoning-disclosure'
import { $showToolActivity, setShowToolActivityFromConfig, toolProgressVisible } from '@/store/tool-activity'

describe('tool feed visibility follows display.tool_progress only', () => {
  beforeEach(() => {
    setShowReasoningFromConfig(true)
    setShowToolActivityFromConfig(undefined)
  })

  it('silences the feed only for false or "off"', () => {
    for (const off of [false, 'off', ' OFF ']) {
      expect(toolProgressVisible(off)).toBe(false)
    }

    for (const on of [true, 'all', 'new', 'verbose', 'unknown', undefined, null]) {
      expect(toolProgressVisible(on)).toBe(true)
    }
  })

  it('does not follow show_reasoning', () => {
    setShowReasoningFromConfig(false)
    setShowToolActivityFromConfig(undefined)
    expect($showToolActivity.get()).toBe(true)

    setShowReasoningFromConfig(true)
    setShowToolActivityFromConfig('off')
    expect($showToolActivity.get()).toBe(false)
  })
})
