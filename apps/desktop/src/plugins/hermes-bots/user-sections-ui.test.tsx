/**
 * `SectionNameDialog`'s Enter-to-submit handling must go through the shared
 * IME-aware predicate (`isSubmitEnter`), not a hand-rolled `isComposing`-only
 * check — a bare check misses the legacy `keyCode === 229` commit-Enter some
 * IMEs (macOS Chinese) still fire after `compositionend`, letting a CJK user's
 * in-flight composition submit/rename with truncated text. See group-chat-parts'
 * identical guard (#93528) for the same class in a sibling hermes-bots surface.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// Radix (Dialog) calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    // The plugin bundle normally lands via `ctx.i18n.register` at load, so
    // without this every localized label renders empty.
    usePluginI18n: () => translateBots
  }
})

const { SectionNameDialog } = await import('./user-sections-ui')

afterEach(() => {
  cleanup()
})

describe('SectionNameDialog Enter handling', () => {
  function mount() {
    const onSubmit = vi.fn()
    render(<SectionNameDialog initialName="" mode="create" onOpenChange={() => undefined} onSubmit={onSubmit} open />)

    return { input: screen.getByRole('textbox'), onSubmit }
  }

  it('submits on a real Enter', () => {
    const { input, onSubmit } = mount()

    fireEvent.change(input, { target: { value: 'Work' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledWith('Work')
  })

  it('does not submit on an in-flight IME composition Enter', () => {
    const { input, onSubmit } = mount()

    fireEvent.change(input, { target: { value: '中文' } })
    fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('does not submit on the legacy IME commit Enter (keyCode 229)', () => {
    const { input, onSubmit } = mount()

    fireEvent.change(input, { target: { value: '中文' } })
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})
