// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { KeybindSettings } from './keybind-settings'

afterEach(cleanup)

/**
 * The send settings are their own page in the settings nav, between Key
 * bindings and HUD gesture. They used to render at the top of the keybind
 * list, where a run of unlabelled rows read as part of the shortcut map
 * rather than as one decision with three parts.
 *
 * Both directions are asserted, because either half alone passes on a broken
 * tree: the shortcuts list showing the panel satisfies "the panel renders",
 * and a deleted route falls through to that same list. The search field is
 * what separates the two pages — the shortcut map owns it, this page does not.
 * Uses the real i18n table, so a label that stops resolving fails here instead
 * of silently rendering a key.
 */
describe('send behavior subpage', () => {
  const open = (subpage?: string) =>
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <KeybindSettings subpage={subpage} />
      </I18nProvider>
    )

  it('renders the send settings on their own page, not the shortcut map', () => {
    open('send-behavior')

    expect(screen.getByLabelText('Keep a bare Enter from sending')).toBeTruthy()
    expect(screen.queryByPlaceholderText('Search shortcuts…')).toBeNull()
  })

  it('no longer renders them inside the keybind list', () => {
    open()

    expect(screen.getByPlaceholderText('Search shortcuts…')).toBeTruthy()
    expect(screen.queryByLabelText('Keep a bare Enter from sending')).toBeNull()
  })
})
