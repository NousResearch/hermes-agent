import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useAutoSendIdle, type UseAutoSendIdleArgs } from './use-auto-send-idle'

interface RenderHarnessOverrides {
  canAutoSend?: () => boolean
  delayMs?: number
  enabled?: boolean
  onFire?: () => void
  readText?: () => string
  resetKey?: string
}

function renderAutoSendHook(overrides: RenderHarnessOverrides = {}) {
  const onFire = overrides.onFire ?? vi.fn()
  const canAutoSend = overrides.canAutoSend ?? (() => true)
  const readText = overrides.readText ?? (() => 'hello world')

  let props: UseAutoSendIdleArgs = {
    canAutoSend,
    delayMs: overrides.delayMs ?? 2000,
    enabled: overrides.enabled ?? true,
    onFire,
    readText,
    resetKey: overrides.resetKey ?? 'a'
  }

  const hook = renderHook((p: UseAutoSendIdleArgs) => useAutoSendIdle(p), { initialProps: props })

  const rerender = (newProps: Partial<UseAutoSendIdleArgs>) => {
    props = { ...props, ...newProps }
    hook.rerender(props)
  }

  return { canAutoSend, hook, onFire, readText, rerender }
}

describe('useAutoSendIdle', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('arms on a trusted insert and fires exactly once after the delay', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(onFire).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)

    // Ensure it never fires a second time from the same arm.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
  })

  it('does not fire before the delay elapses', async () => {
    const { hook, onFire } = renderAutoSendHook({ delayMs: 2000 })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1999)
    })
    expect(onFire).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
  })

  it('armedInSeconds is 2 right after arming with delayMs 2000, 1 after 1000 ms, null after firing', async () => {
    const { hook, onFire } = renderAutoSendHook({ delayMs: 2000 })

    expect(hook.result.current.armedInSeconds).toBeNull()

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(hook.result.current.armedInSeconds).toBe(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(hook.result.current.armedInSeconds).toBeNull()
    expect(onFire).toHaveBeenCalledTimes(1)
  })

  it('does NOT fire for an untrusted edit (noteEdit(false, "insertText"))', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteEdit(false, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })

  it('does NOT fire for inputType insertFromPaste or deleteContentBackward, and an armed timer is cancelled by them', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteEdit(true, 'insertFromPaste')
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()

    // Arm, then cancel with insertFromPaste.
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    act(() => {
      hook.result.current.noteEdit(true, 'insertFromPaste')
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()

    // Arm, then cancel with deleteContentBackward.
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    act(() => {
      hook.result.current.noteEdit(true, 'deleteContentBackward')
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })

  it('insertReplacementText DOES arm and fire (macOS dictation correction)', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteEdit(true, 'insertReplacementText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
  })

  it('a second trusted insert re-arms: no fire at the original deadline, one fire at the new one', async () => {
    const { hook, onFire } = renderAutoSendHook({ delayMs: 2000 })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(hook.result.current.armedInSeconds).toBe(1)

    // Re-arm at t = 1000 ms; resets window to full 2000 ms.
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    // Advance to t = 2000 ms (original deadline) -> should not fire.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(onFire).not.toHaveBeenCalled()
    expect(hook.result.current.armedInSeconds).toBe(1)

    // Advance to t = 3000 ms (new deadline) -> fires once.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('noteCommittedComposition() arms and fires', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteCommittedComposition()
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('fire-time canAutoSend() === false → no fire (the busy/disabled race)', async () => {
    let canSend = true

    const { hook, onFire } = renderAutoSendHook({
      canAutoSend: () => canSend
    })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    // Composer becomes busy or disabled during the idle window.
    canSend = false

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('fire-time empty/whitespace readText() → no fire', async () => {
    let text = 'hello world'

    const { hook, onFire } = renderAutoSendHook({
      readText: () => text
    })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    text = '   \t\n  '

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('fire-time text starting with "/" → no fire', async () => {
    let text = 'hello'

    const { hook, onFire } = renderAutoSendHook({
      readText: () => text
    })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    text = '   /help'

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('cancel() prevents the fire; armedInSeconds is null', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    act(() => {
      hook.result.current.cancel()
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })

  it('enabled: false → never fires', async () => {
    // Starts disabled: noteEdit does not arm.
    const { hook, onFire, rerender } = renderAutoSendHook({ enabled: false })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()

    // Enabled flips to false while armed: cancels immediately.
    rerender({ enabled: true })
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    rerender({ enabled: false })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })

  it('delayMs change while armed → no fire at the old deadline', async () => {
    const { hook, onFire, rerender } = renderAutoSendHook({ delayMs: 2000 })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    rerender({ delayMs: 3000 })
    expect(hook.result.current.armedInSeconds).toBeNull()

    // Advance past old deadline (2000 ms).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()

    // Settings change disarms immediately without auto-rearming.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })

  it('resetKey change while armed → no fire', async () => {
    const { hook, onFire, rerender } = renderAutoSendHook({ resetKey: 'session-1' })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    rerender({ resetKey: 'session-2' })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })

  it('unmount while armed → onFire never called', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    hook.unmount()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })

  it('noteEdit(true, "insertLineBreak") and noteEdit(true, "insertParagraph") each arm and fire after the delay', async () => {
    const { hook, onFire } = renderAutoSendHook()

    act(() => {
      hook.result.current.noteEdit(true, 'insertLineBreak')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()

    act(() => {
      hook.result.current.noteEdit(true, 'insertParagraph')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(2)
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('trusted edit with empty or omitted inputType arms and fires', async () => {
    const { hook, onFire } = renderAutoSendHook()

    // Empty inputType (Windows Voice Typing)
    act(() => {
      hook.result.current.noteEdit(true, '')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()

    // Omitted inputType
    act(() => {
      hook.result.current.noteEdit(true)
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(2)
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('noteCommittedComposition() followed immediately by trailing insertFromComposition still fires exactly once after the delay', async () => {
    const { hook, onFire } = renderAutoSendHook({ delayMs: 2000 })

    act(() => {
      hook.result.current.noteCommittedComposition()
      hook.result.current.noteEdit(true, 'insertFromComposition')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()

    // Ensure it fires only once.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
  })

  it('slash semantics: fires for path-like prose (/Users/me/notes.md is stale) but not for slash commands (/help)', async () => {
    let text = '/Users/me/notes.md is stale'

    const { hook, onFire } = renderAutoSendHook({
      readText: () => text
    })

    // Path-like prose is not a slash command — should fire
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()

    // Slash command — should not fire
    text = '/help'
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('slash semantics: a slash command embedded in prose still sends (it is a message)', async () => {
    // The guard is anchored: a message that merely CONTAINS a command is a
    // message, exactly as the submit engine treats it when Enter is pressed.
    const { hook, onFire } = renderAutoSendHook({
      readText: () => 'please /help me with the parser'
    })

    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onFire).toHaveBeenCalledTimes(1)
    expect(hook.result.current.armedInSeconds).toBeNull()
  })

  it('untrusted edits with an empty or omitted inputType cancel and never arm', async () => {
    const { hook, onFire } = renderAutoSendHook()

    // Arm with a trusted insert first, so the cancel path is what is exercised.
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    act(() => {
      hook.result.current.noteEdit(false, '')
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    // An omitted inputType is the same case: trust is what decides.
    act(() => {
      hook.result.current.noteEdit(true, 'insertText')
    })
    expect(hook.result.current.armedInSeconds).toBe(2)

    act(() => {
      hook.result.current.noteEdit(false)
    })
    expect(hook.result.current.armedInSeconds).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(onFire).not.toHaveBeenCalled()
  })
})
