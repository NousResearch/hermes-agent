import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { getComposerSpellcheckDesired, requestComposerSpellcheck } from '../focus'

import { useComposerSpellcheck } from './use-composer-spellcheck'

/** Let the bus's deferred dispatch run. */
const tick = () => new Promise(resolve => setTimeout(resolve, 0))

describe('useComposerSpellcheck', () => {
  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('applies the setting to the editable when a request arrives', async () => {
    const editor = document.createElement('div')

    editor.spellcheck = false

    const { unmount } = renderHook(() => {
      const ref = { current: editor } as { current: HTMLElement | null }

      useComposerSpellcheck({ target: 'tile:apply', editorRef: ref })
    })

    await requestComposerSpellcheck(true, 'tile:apply')
    await tick()

    expect(editor.spellcheck).toBe(true)

    unmount()
  })

  it('ignores a request addressed to another surface', async () => {
    const editor = document.createElement('div')

    editor.spellcheck = false

    const { unmount } = renderHook(() => {
      const ref = { current: editor } as { current: HTMLElement | null }

      useComposerSpellcheck({ target: 'tile:mine', editorRef: ref })
    })

    await requestComposerSpellcheck(true, 'tile:other')
    await tick()

    expect(editor.spellcheck).toBe(false)

    unmount()
  })

  it('picks up a setting requested BEFORE the surface mounted', async () => {
    // The request lands while no surface is subscribed. The requested state
    // is recorded against the surface, so mounting later still applies it --
    // otherwise the requester would have been told "on" for a composer that
    // never received it.
    await requestComposerSpellcheck(true, 'tile:late')
    await tick()

    expect(getComposerSpellcheckDesired('tile:late')).toBe(true)

    const editor = document.createElement('div')

    editor.spellcheck = false

    const { unmount } = renderHook(() => {
      const ref = { current: editor } as { current: HTMLElement | null }

      useComposerSpellcheck({ target: 'tile:late', editorRef: ref })
    })

    await tick()

    expect(editor.spellcheck).toBe(true)

    unmount()
  })

  it('leaves the editable alone when nothing was ever requested', async () => {
    const editor = document.createElement('div')

    editor.spellcheck = false

    const { unmount } = renderHook(() => {
      const ref = { current: editor } as { current: HTMLElement | null }

      useComposerSpellcheck({ target: 'tile:never', editorRef: ref })
    })

    await tick()

    expect(editor.spellcheck).toBe(false)

    unmount()
  })
})
