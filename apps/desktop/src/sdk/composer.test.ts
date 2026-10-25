// @vitest-environment jsdom
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { useComposerSpellcheck } from '@/app/chat/composer/hooks/use-composer-spellcheck'

import { composerHost } from './composer'

/** Mount a composer surface for `target`, like ChatBar does. */
const mountSurface = (target: string, editor: HTMLElement) =>
  renderHook(() => {
    const ref = { current: editor } as { current: HTMLElement | null }

    useComposerSpellcheck({ target, editorRef: ref })
  })

describe('host.composer.setSpellcheck', () => {
  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('resolves true when a mounted surface applies the setting', async () => {
    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'false')

    const { unmount } = mountSurface('tile:sess-live', editor)

    await expect(composerHost.setSpellcheck('sess-live', true)).resolves.toBe(true)
    expect(editor.getAttribute('spellcheck')).toBe('true')

    await expect(composerHost.setSpellcheck('sess-live', false)).resolves.toBe(true)
    expect(editor.getAttribute('spellcheck')).toBe('false')

    unmount()
  })

  it('resolves false when no mounted surface claims the request', async () => {
    // The address resolves to a surface with no subscriber, so the request is
    // dropped. The verb must report that, rather than claiming the setting was
    // applied.
    await expect(composerHost.setSpellcheck(null, true)).resolves.toBe(false)
  })

  it('resolves false for an address that resolves to no surface at all', async () => {
    // A session id the primary composer is not showing resolves to a tile
    // target; with no tile mounted, nothing claims it.
    await expect(composerHost.setSpellcheck('sess-does-not-exist', true)).resolves.toBe(false)
  })

  it('records nothing for an address that no surface claimed', async () => {
    // Same fail-closed contract end to end: the plugin was told false, so the
    // setting must not turn up later on a surface that mounts afterwards.
    await expect(composerHost.setSpellcheck('sess-ghost', true)).resolves.toBe(false)

    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'false')

    const { unmount } = mountSurface('tile:sess-ghost', editor)

    await new Promise(resolve => setTimeout(resolve, 0))

    expect(editor.getAttribute('spellcheck')).toBe('false')

    unmount()
  })

  it('rejects a non-boolean `enabled` instead of coercing it', async () => {
    // Argument validation matches `setDraft`, which rejects a non-string text:
    // a bad argument is a failed call, never a truthy spelling mode. The
    // surface is mounted on purpose — without the guard `'yes'` would reach it
    // as a truthy value and be applied.
    const editor = document.createElement('div')

    editor.setAttribute('spellcheck', 'false')

    const { unmount } = mountSurface('tile:sess-arg', editor)

    await expect(composerHost.setSpellcheck('sess-arg', 'yes' as unknown as boolean)).resolves.toBe(false)
    await expect(composerHost.setSpellcheck('sess-arg', undefined as unknown as boolean)).resolves.toBe(false)

    expect(editor.getAttribute('spellcheck')).toBe('false')

    unmount()
  })
})
