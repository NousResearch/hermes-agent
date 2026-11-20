import { normalizeComposerSendPrefs } from '@hermes/shared'
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $composerSendPrefs } from '@/store/composer-prefs'

import { formatCombo } from './combo'
import { useKeybindHint } from './use-keybind-hint'

afterEach(() => act(() => $composerSendPrefs.set(normalizeComposerSendPrefs({}))))

describe('useKeybindHint composer Enter mode', () => {
  it('updates the send hint when the gate closes', () => {
    const send = renderHook(() => useKeybindHint('composer.send'))
    const queue = renderHook(() => useKeybindHint('composer.queue'))

    expect(send.result.current).toBe(formatCombo('enter'))
    expect(queue.result.current).toBe(formatCombo('mod+enter'))

    act(() => $composerSendPrefs.set(normalizeComposerSendPrefs({ enterSends: false })))

    expect(send.result.current).toBe(formatCombo('mod+enter'))
    expect(queue.result.current).toBe(formatCombo('mod+enter'))
  })

  it('keeps the generic send hint on the chord once a gesture owns the bare press', () => {
    act(() => $composerSendPrefs.set(normalizeComposerSendPrefs({ enterSends: false, sendOnDoubleTap: true })))

    const double = renderHook(() => useKeybindHint('composer.send.double'))
    const send = renderHook(() => useKeybindHint('composer.send'))

    // The gesture has its own row and spells both presses out; the generic row is
    // the chord that always works, because a lone Enter no longer commits.
    expect(double.result.current).toBe(`${formatCombo('enter')} ${formatCombo('enter')}`)
    expect(send.result.current).toBe(formatCombo('mod+enter'))
  })
})
