import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type DictationHandle, startComposerDictation } from '@/lib/composer-dictation'

import { usePluginDictation } from './use-plugin-dictation'
import type { DictationDestination, useVoiceRecorder } from './use-voice-recorder'

vi.mock('../scope', () => ({ useComposerScope: () => ({ connectionId: 'local', profile: 'test' }) }))
afterEach(cleanup)

describe('plugin dictation routing', () => {
  it('claims exactly the addressed recorder and drops callbacks after a session switch', () => {
    let destination: DictationDestination | undefined

    const recorder = {
      voiceStatus: 'idle',
      start: vi.fn(async (value: DictationDestination) => {
        destination = value
      }),
      stop: vi.fn(async () => undefined),
      cancel: vi.fn()
    } as unknown as ReturnType<typeof useVoiceRecorder>

    const recipient = { onTranscript: vi.fn(), onState: vi.fn() }

    const hook = renderHook(
      ({ sessionId }) => usePluginDictation({ target: 'main', sessionId, disabled: false, recorder }),
      { initialProps: { sessionId: 'a' } }
    )

    expect(startComposerDictation('tile:other', recipient)).toBeNull()
    let handle: DictationHandle | null = null
    act(() => {
      handle = startComposerDictation('main', recipient)
    })
    expect(handle).not.toBeNull()
    expect(startComposerDictation('main', recipient)).toBeNull()
    act(() => {
      destination!.onTranscript('Comment only')
      handle!.stop()
    })
    expect(recipient.onTranscript).toHaveBeenCalledWith('Comment only')
    expect(recorder.stop).toHaveBeenCalledOnce()
    hook.rerender({ sessionId: 'b' })
    expect(recorder.cancel).toHaveBeenCalledOnce()
    destination!.onTranscript('Late result')
    handle!.stop()
    expect(recipient.onTranscript).toHaveBeenCalledTimes(1)
    expect(recorder.stop).toHaveBeenCalledOnce()
    hook.unmount()
    expect(startComposerDictation('main', recipient)).toBeNull()
  })

  it('rejects a disabled or occupied recorder and allows a new take after completion', () => {
    let destination: DictationDestination | undefined

    const recorder = {
      voiceStatus: 'recording',
      start: vi.fn(async (value: DictationDestination) => {
        destination = value
      }),
      stop: vi.fn(),
      cancel: vi.fn()
    } as unknown as ReturnType<typeof useVoiceRecorder>

    const recipient = { onTranscript: vi.fn(), onState: vi.fn() }

    const hook = renderHook(
      ({ disabled }) => usePluginDictation({ target: 'main', sessionId: 'a', disabled, recorder }),
      { initialProps: { disabled: false } }
    )

    expect(startComposerDictation('main', recipient)).toBeNull()
    recorder.voiceStatus = 'idle'
    hook.rerender({ disabled: true })
    expect(startComposerDictation('main', recipient)).toBeNull()
    hook.rerender({ disabled: false })
    expect(startComposerDictation('main', recipient)).not.toBeNull()
    act(() => destination!.onState?.('idle'))
    expect(startComposerDictation('main', recipient)).not.toBeNull()
  })
})
