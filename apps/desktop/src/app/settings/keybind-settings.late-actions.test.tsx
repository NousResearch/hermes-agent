/**
 * A keybind action contributed AFTER the settings tab mounts must appear in
 * the map. Same class as the late plugin-route bug (#109063): the component
 * subscribed to `useContributions(KEYBINDS_AREA)` but discarded the snapshot
 * and called the impure `allKeybindActions()` in render, so the React
 * Compiler memoized the list without the subscription as an input. Uses the
 * real registry + hook under the compiler-enabled vitest ui project.
 */
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'
import { KEYBINDS_AREA } from '@/lib/keybinds/actions'

import { KeybindSettings } from './keybind-settings'

afterEach(cleanup)

describe('KeybindSettings late-contributed actions', () => {
  it('lists an action whose keybind registers after mount', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <KeybindSettings />
      </I18nProvider>
    )
    expect(screen.queryByText('Late Plugin Action')).toBeNull()

    let dispose = () => {}
    act(() => {
      dispose = registry.register({
        area: KEYBINDS_AREA,
        id: 'late-plugin:action',
        data: { id: 'late-plugin:action', label: 'Late Plugin Action', category: 'view', run: () => {} }
      })
    })

    expect(screen.getByText('Late Plugin Action')).toBeTruthy()

    act(() => dispose())
  })
})
