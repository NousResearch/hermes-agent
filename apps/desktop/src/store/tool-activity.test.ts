import { beforeEach, describe, expect, it } from 'vitest'

import { setShowReasoningFromConfig } from '@/store/reasoning-disclosure'
import { $showToolActivity, setShowToolActivityFromConfig } from '@/store/tool-activity'

describe('tool feed visibility (display.tool_progress × display.show_reasoning)', () => {
  beforeEach(() => {
    setShowReasoningFromConfig(true)
    setShowToolActivityFromConfig(undefined)
  })

  it('shows the feed by default', () => {
    expect($showToolActivity.get()).toBe(true)
  })

  it('hides the feed under the answer-only default (no stated feed preference)', () => {
    setShowReasoningFromConfig(false)

    expect($showToolActivity.get()).toBe(false)
  })

  it('keeps the feed when the preference is stated, even with reasoning hidden', () => {
    setShowReasoningFromConfig(false)

    setShowToolActivityFromConfig('all')
    expect($showToolActivity.get()).toBe(true)

    setShowToolActivityFromConfig('verbose')
    expect($showToolActivity.get()).toBe(true)
  })

  it('treats a bare YAML off (false) and "off" as silenced', () => {
    setShowToolActivityFromConfig(false)
    expect($showToolActivity.get()).toBe(false)

    setShowToolActivityFromConfig(' OFF ')
    expect($showToolActivity.get()).toBe(false)
  })

  it('silences the feed on an explicit off even when reasoning blocks are on', () => {
    setShowToolActivityFromConfig('off')

    expect($showToolActivity.get()).toBe(false)
  })

  it('counts a present-but-null key as a stated preference, like the gateway', () => {
    // The gateway reads `tool_progress: <empty>` as written-but-all; the client
    // mirrors presence, not truthiness.
    setShowReasoningFromConfig(false)
    setShowToolActivityFromConfig(null)

    expect($showToolActivity.get()).toBe(true)
  })
})
