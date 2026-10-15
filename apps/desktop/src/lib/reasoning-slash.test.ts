import { describe, expect, it, vi } from 'vitest'

import { resolveDesktopCommand } from '@/lib/desktop-slash-commands'
import { applyReasoningSlashResult, reasoningSlashParams } from '@/lib/reasoning-slash'
import { $showReasoning, setShowReasoningFromConfig, syncShowReasoningForSession } from '@/store/reasoning-disclosure'

// `/reasoning hide` typed into the Desktop composer must gate Thinking blocks
// in the transcript immediately (#111761, #49664): the renderer mirrors the
// gateway's `config.set key=reasoning` answer instead of waiting for the next
// config refresh.
describe('/reasoning slash command', () => {
  it('runs as a desktop action instead of the slash worker', () => {
    expect(resolveDesktopCommand('/reasoning')?.surface.kind).toBe('action')
  })

  it('builds the gateway config.set payload, honoring the scope flags', () => {
    expect(reasoningSlashParams('hide', 's1')).toEqual({ key: 'reasoning', session_id: 's1', value: 'hide' })
    expect(reasoningSlashParams('high --global', 's1')).toEqual({
      key: 'reasoning',
      scope: 'global',
      session_id: 's1',
      value: 'high'
    })
    expect(reasoningSlashParams('   ', 's1')).toBeNull()
  })

  it('flips the Thinking gate on hide/show and leaves it alone for effort levels', () => {
    setShowReasoningFromConfig(true)
    applyReasoningSlashResult('hide')
    expect($showReasoning.get()).toBe(false)
    applyReasoningSlashResult('high')
    expect($showReasoning.get()).toBe(false)
    applyReasoningSlashResult('show')
    expect($showReasoning.get()).toBe(true)
  })

  it('syncs the display preference to an active gateway session', async () => {
    const request = vi.fn(async () => ({}))

    setShowReasoningFromConfig(false)
    await syncShowReasoningForSession(request, 'runtime-1')

    expect(request).toHaveBeenCalledWith('config.set', {
      key: 'show_reasoning',
      session_id: 'runtime-1',
      value: '0'
    })

    setShowReasoningFromConfig(true)
    await syncShowReasoningForSession(request, 'runtime-2')

    expect(request).toHaveBeenLastCalledWith('config.set', {
      key: 'show_reasoning',
      session_id: 'runtime-2',
      value: '1'
    })
  })
})
