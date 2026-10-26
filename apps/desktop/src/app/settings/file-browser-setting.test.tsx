// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { $interfaceMode } from '@/store/interface-mode'
import { $fileBrowserOpen, FILE_BROWSER_PANE_ID, setFileBrowserOpen, toggleFileBrowserOpen } from '@/store/layout'
import { $paneStates } from '@/store/panes'

import { AppearanceSettings } from './appearance-settings'

// #65173: the file browser's open/closed default was only reachable through
// ⌘J and a persistence chain nobody could see. Settings now states it, and it
// is the same state the titlebar toggle writes — two views, one answer.

afterEach(() => {
  cleanup()
  act(() => {
    $interfaceMode.set('advanced')
    setFileBrowserOpen(false)
  })
})

function fileBrowserSwitch() {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <AppearanceSettings subpage="window-layout" />
    </QueryClientProvider>
  )

  return screen.getByRole('switch', { name: 'File Browser' })
}

describe('File Browser setting', () => {
  it('turns the file browser on and off, and persists the choice', () => {
    const toggle = fileBrowserSwitch()

    expect(toggle.getAttribute('aria-checked')).toBe('false')

    fireEvent.click(toggle)

    expect($fileBrowserOpen.get()).toBe(true)
    expect($paneStates.get()[FILE_BROWSER_PANE_ID]?.open).toBe(true)

    fireEvent.click(toggle)

    expect($fileBrowserOpen.get()).toBe(false)
    expect($paneStates.get()[FILE_BROWSER_PANE_ID]?.open).toBe(false)
  })

  it('follows the titlebar toggle', () => {
    const toggle = fileBrowserSwitch()

    act(() => toggleFileBrowserOpen())

    expect(toggle.getAttribute('aria-checked')).toBe('true')
  })

  it('says a flip is session-only while Simple mode decides the value', () => {
    const note = en.interfaceMode.sessionNote

    act(() => $interfaceMode.set('simple'))
    const toggle = fileBrowserSwitch()

    expect(screen.getByText(content => content.includes(note))).toBeTruthy()

    // Simple shadows the preference: the click is a session reveal, not a new default.
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-checked')).toBe('true')

    act(() => $interfaceMode.set('advanced'))
    expect(screen.queryByText(content => content.includes(note))).toBeNull()
    expect(toggle.getAttribute('aria-checked')).toBe('false')
  })
})
