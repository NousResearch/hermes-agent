import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { useComposerInputDisabled } from '@/app/chat/composer/hooks/use-composer-input-disabled'

import { composerInputHeld, holdComposerInput } from './composer-holds'

describe('composer input holds', () => {
  it('holds one composer until its own token is released', () => {
    const release = holdComposerInput('tile:a')

    expect(composerInputHeld('tile:a')).toBe(true)
    expect(composerInputHeld('main')).toBe(false)

    release()
    expect(composerInputHeld('tile:a')).toBe(false)
  })

  it('keeps a composer held while any holder remains; a release is idempotent', () => {
    const first = holdComposerInput('main')
    const second = holdComposerInput('main')

    first()
    first()
    expect(composerInputHeld('main')).toBe(true)

    second()
    expect(composerInputHeld('main')).toBe(false)
  })
})

describe('useComposerInputDisabled', () => {
  it('disables only the held composer, and keeps the gateway rule', () => {
    const main = renderHook(() => useComposerInputDisabled(false, 'open', 'main'))
    const tile = renderHook(() => useComposerInputDisabled(false, 'open', 'tile:b'))

    let release = () => {}
    act(() => void (release = holdComposerInput('main')))

    expect(main.result.current).toBe(true)
    expect(tile.result.current).toBe(false)

    act(() => release())
    expect(main.result.current).toBe(false)

    expect(renderHook(() => useComposerInputDisabled(true, 'open', 'main')).result.current).toBe(true)
    expect(renderHook(() => useComposerInputDisabled(true, 'connecting', 'main')).result.current).toBe(false)
  })
})
