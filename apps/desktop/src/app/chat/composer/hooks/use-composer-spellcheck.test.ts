import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { requestComposerSpellcheck } from '../focus'

import { useComposerSpellcheck } from './use-composer-spellcheck'

/** Let the bus's deferred dispatch run. */
const tick = () => new Promise(resolve => setTimeout(resolve, 0))

/** Mount the hook for `target` with `editor` already attached (or not). */
const mountFor = (target: string, editor: HTMLElement | null) =>
  renderHook(() => {
    const ref = { current: editor } as { current: HTMLElement | null }

    useComposerSpellcheck({ target, editorRef: ref })
  })

describe('useComposerSpellcheck', () => {
  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('applies the setting to the editable and reports it applied', async () => {
    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'false')

    const { unmount } = mountFor('tile:apply', editor)

    await expect(requestComposerSpellcheck(true, 'tile:apply')).resolves.toBe(true)
    await tick()

    expect(editor.getAttribute('spellcheck')).toBe('true')

    unmount()
  })

  it('turns a previously enabled editable back off', async () => {
    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'true')

    const { unmount } = mountFor('tile:off', editor)

    await expect(requestComposerSpellcheck(false, 'tile:off')).resolves.toBe(true)
    await tick()

    expect(editor.getAttribute('spellcheck')).toBe('false')

    unmount()
  })

  it('ignores a request addressed to another surface', async () => {
    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'false')

    const { unmount } = mountFor('tile:mine', editor)

    await expect(requestComposerSpellcheck(true, 'tile:other')).resolves.toBe(false)
    await tick()

    expect(editor.getAttribute('spellcheck')).toBe('false')

    unmount()
  })

  it('reports false when there is no editable to apply the setting to', async () => {
    // The surface is mounted (so it claims the request) but its editable is not
    // attached yet: nothing was applied, so the ack must say so. Anything else
    // tells the plugin "on" for a composer that changed nothing.
    const { unmount } = mountFor('tile:dead', null)

    await expect(requestComposerSpellcheck(true, 'tile:dead')).resolves.toBe(false)

    unmount()
  })

  it('does not apply a request that arrived before this surface mounted', async () => {
    // The fail-closed contract: a request no mounted surface claimed is
    // DROPPED, not remembered. A late mount starts from the app's default
    // (off) and the requester has already been told false, so the two agree.
    await expect(requestComposerSpellcheck(true, 'tile:late')).resolves.toBe(false)
    await tick()

    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'false')

    const { unmount } = mountFor('tile:late', editor)

    await tick()

    expect(editor.getAttribute('spellcheck')).toBe('false')

    unmount()
  })

  it('leaves the editable alone when nothing was ever requested', async () => {
    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'false')

    const { unmount } = mountFor('tile:never', editor)

    await tick()

    expect(editor.getAttribute('spellcheck')).toBe('false')

    unmount()
  })
})
