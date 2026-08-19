/**
 * The regression that motivates this file: cmdk calls the root
 * `onValueChange` only in controlled mode (a set `value` prop). The palette
 * is uncontrolled, so a highlight preview wired to that prop never fires.
 * The watcher must read the cmdk store instead, which works in both modes.
 */

import { render } from '@testing-library/react'
import { Command } from 'cmdk'
import { describe, expect, it, vi } from 'vitest'

import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { HighlightWatcher } from './highlight-watcher'

stubResizeObserver()
stubMenuDomApis()

const palette = (onValue: (value: string) => void, onRootValueChange?: (value: string) => void) => (
  <Command onValueChange={onRootValueChange}>
    <HighlightWatcher onValue={onValue} />
    <Command.List>
      <Command.Item value="alpha">Alpha</Command.Item>
      <Command.Item value="beta">Beta</Command.Item>
    </Command.List>
  </Command>
)

describe('HighlightWatcher', () => {
  it('reports the highlight from the cmdk store in uncontrolled mode', () => {
    const onValue = vi.fn()

    render(palette(onValue))

    // cmdk auto-highlights the first item on mount.
    expect(onValue).toHaveBeenCalledWith('alpha')
  })

  it('covers the gap: the root onValueChange stays silent in uncontrolled mode', () => {
    const onValue = vi.fn()
    const onRootValueChange = vi.fn()

    render(palette(onValue, onRootValueChange))

    // The store-based watcher fires. The prop the first implementation used
    // does not. If cmdk starts to fire it in uncontrolled mode, the watcher
    // becomes removable — this assertion is the signal.
    expect(onValue).toHaveBeenCalledWith('alpha')
    expect(onRootValueChange).not.toHaveBeenCalled()
  })
})
