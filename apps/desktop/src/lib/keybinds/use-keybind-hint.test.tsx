import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $composerEnterSends } from '@/store/composer-prefs'

import { formatCombo } from './combo'
import { useKeybindHint } from './use-keybind-hint'

afterEach(() => act(() => $composerEnterSends.set(true)))

describe('useKeybindHint composer Enter mode', () => {
  it('updates send and queue hints when multiline-first mode is enabled', () => {
    const send = renderHook(() => useKeybindHint('composer.send'))
    const queue = renderHook(() => useKeybindHint('composer.queue'))

    expect(send.result.current).toBe(formatCombo('enter'))
    expect(queue.result.current).toBe(formatCombo('mod+enter'))

    act(() => $composerEnterSends.set(false))

    expect(send.result.current).toBe(formatCombo('mod+enter'))
    expect(queue.result.current).toBe(formatCombo('mod+enter'))
  })
})
