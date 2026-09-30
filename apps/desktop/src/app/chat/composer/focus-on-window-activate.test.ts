import { afterEach, describe, expect, it } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { setAutoFocusComposer } from '@/store/auto-focus-composer'

import { onComposerFocusRequest } from './focus'
import { handleWindowActivate } from './focus-on-window-activate'

/** The bus defers dispatch by one macrotask. This helper flushes it. */
const flushBus = () => new Promise(resolve => setTimeout(resolve, 1))

/** Focus requests that reached the bus during one window activation. */
async function focusRequests(): Promise<string[]> {
  const targets: string[] = []
  const off = onComposerFocusRequest(({ target }) => targets.push(target))

  handleWindowActivate()
  await flushBus()
  off()

  return targets
}

afterEach(() => {
  setAutoFocusComposer(false)
  $workspaceIsPage.set(false)
  document.body.replaceChildren()
})

describe('handleWindowActivate', () => {
  it('focuses the composer when the window regains focus and the pref is on', async () => {
    setAutoFocusComposer(true)

    expect(await focusRequests()).toEqual(['main'])
  })

  it('does nothing while the pref is off', async () => {
    setAutoFocusComposer(false)

    expect(await focusRequests()).toEqual([])
  })

  it('leaves the caret with another input — keyboard ownership follows focus', async () => {
    setAutoFocusComposer(true)

    const input = document.createElement('input')
    document.body.append(input)
    input.focus()

    expect(await focusRequests()).toEqual([])
    expect(document.activeElement).toBe(input)
  })

  it('stands down behind a full workspace page', async () => {
    setAutoFocusComposer(true)
    $workspaceIsPage.set(true)

    expect(await focusRequests()).toEqual([])
  })

  it('stands down while an overlay surface is open', async () => {
    setAutoFocusComposer(true)

    const dialog = document.createElement('div')
    dialog.setAttribute('role', 'dialog')
    document.body.append(dialog)

    expect(await focusRequests()).toEqual([])
  })
})
