import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type React from 'react'
import { afterEach, describe, expect, it, test, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { publishMicLevel, resetMicLevel } from '@/store/voice-level'

import { VoiceControls } from './voice-controls'

/** The jarvis stylesheets sit next to these tests; vitest module URLs are not file URLs. */
function readJarvisCss(file: string): string {
  for (const from of ['src/app/jarvis', 'apps/desktop/src/app/jarvis']) {
    try {
      return readFileSync(resolve(process.cwd(), from, file), 'utf8')
    } catch {
      // Try the next root the runner may have been started from.
    }
  }

  throw new Error(`could not read ${file}`)
}

function controlsProps(overrides: Partial<React.ComponentProps<typeof VoiceControls>> = {}) {
  return {
    cancelTask: vi.fn(),
    disabled: false,
    listening: false,
    loading: false,
    startListening: vi.fn(),
    stopListening: vi.fn(),
    stopPlayback: vi.fn(),
    speaking: true,
    taskRunning: true,
    ...overrides
  }
}

function renderControls(overrides: Partial<React.ComponentProps<typeof VoiceControls>> = {}) {
  const props = controlsProps(overrides)

  render(
    <I18nProvider initialLocale="pl">
      <VoiceControls {...props} />
    </I18nProvider>
  )

  return props
}

afterEach(() => {
  cleanup()
  resetMicLevel()
})

describe('VoiceControls', () => {
  it('stops speaking without cancelling the active task', () => {
    const controls = renderControls()

    fireEvent.click(screen.getByRole('button', { name: 'Przestań mówić' }))

    expect(controls.stopPlayback).toHaveBeenCalledTimes(1)
    expect(controls.cancelTask).not.toHaveBeenCalled()
  })

  it('cancels the backend task exactly once', () => {
    const controls = renderControls()

    fireEvent.click(screen.getByRole('button', { name: 'Zatrzymaj zadanie' }))

    expect(controls.cancelTask).toHaveBeenCalledTimes(1)
    expect(controls.stopPlayback).not.toHaveBeenCalled()
  })

  it('renders a neutral microphone meter until the real recorder reports a level', () => {
    renderControls({ listening: true })
    const meter = screen.getByRole('meter', { name: 'Poziom mikrofonu' })

    expect(meter.getAttribute('aria-valuenow')).toBe('0')

    act(() => publishMicLevel(0.5))

    expect(meter.getAttribute('aria-valuenow')).toBe('50')
    expect(meter.style.getPropertyValue('--jarvis-audio-level')).toBe('0.5')
  })

  it('keeps the meter flat when the microphone is closed or muted', () => {
    act(() => publishMicLevel(0.9))

    const { rerender } = render(
      <I18nProvider initialLocale="pl">
        <VoiceControls {...controlsProps({ listening: false })} />
      </I18nProvider>
    )

    expect(screen.getByRole('meter', { name: 'Poziom mikrofonu' }).getAttribute('aria-valuenow')).toBe('0')

    rerender(
      <I18nProvider initialLocale="pl">
        <VoiceControls {...controlsProps({ listening: true, muted: true })} />
      </I18nProvider>
    )

    expect(screen.getByRole('meter', { name: 'Poziom mikrofonu' }).getAttribute('aria-valuenow')).toBe('0')
  })

  it('mutes the microphone without cancelling the running task', () => {
    const controls = renderControls({ listening: true, toggleMute: vi.fn() })

    fireEvent.click(screen.getByRole('button', { name: 'Wycisz mikrofon' }))

    expect(controls.toggleMute).toHaveBeenCalledTimes(1)
    expect(controls.cancelTask).not.toHaveBeenCalled()
    expect(controls.stopListening).not.toHaveBeenCalled()
  })

  it('offers no mute control when the conversation cannot be muted', () => {
    renderControls({ listening: true })

    expect(screen.queryByRole('button', { name: 'Wycisz mikrofon' })).toBeNull()
  })

  it('uses English labels from the locale contract', () => {
    render(
      <I18nProvider initialLocale="en">
        <VoiceControls {...controlsProps()} />
      </I18nProvider>
    )

    expect(screen.getByRole('button', { name: 'Start listening' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Stop speaking' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Stop task' })).toBeTruthy()
  })

  it('exposes disabled controls and error alerts accessibly', () => {
    renderControls({ disabled: true, error: 'Mikrofon jest niedostępny' })

    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zacznij słuchać' }).disabled).toBe(true)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Przestań mówić' }).disabled).toBe(true)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Zatrzymaj zadanie' }).disabled).toBe(true)
    expect(screen.getByRole('alert').textContent).toBe('Mikrofon jest niedostępny')
  })

  it('keeps native button keyboard affordances focusable', () => {
    renderControls()
    const button = screen.getByRole('button', { name: 'Zacznij słuchać' })

    button.focus()

    expect(button.ownerDocument.activeElement).toBe(button)
    expect(button.tagName).toBe('BUTTON')
    expect(button.getAttribute('type')).toBe('button')
  })

  test.each([
    ['idle', false],
    ['cancelling', false],
    ['cancelled', false],
    ['failed', false],
    ['verified', false],
    ['planning', true],
    ['running', true],
    ['approval', true]
  ])('maps task phase %s to cancel enabled=%s', (_phase, taskRunning) => {
    renderControls({ taskRunning })

    expect(screen.getByRole('button', { name: 'Zatrzymaj zadanie' })).toHaveProperty('disabled', !taskRunning)
  })

  it('keeps one microphone in the dock: the mute is never a second mic, and the row never wraps', () => {
    renderControls({ listening: true, toggleMute: vi.fn() })

    // The only control that opens or closes the microphone…
    expect(screen.getAllByRole('button', { name: 'Przestań słuchać' })).toHaveLength(1)
    // …the mute is its own, distinctly labelled control (slashed icon in both
    // states, so an idle dock cannot show two identical microphones)…
    const mute = screen.getByRole('button', { name: 'Wycisz mikrofon' })

    expect(mute.getAttribute('aria-pressed')).toBe('false')
    expect(screen.queryByRole('button', { name: 'Włącz mikrofon' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Zacznij słuchać' })).toBeNull()

    // …and every control wears the same quiet glass material.
    for (const name of ['Przestań słuchać', 'Wycisz mikrofon', 'Przestań mówić', 'Zatrzymaj zadanie']) {
      expect(screen.getByRole('button', { name }).className).toContain('jarvis-icon-btn')
    }

    const dock = screen.getByTestId('jarvis-voice-controls')

    expect(dock.className).toContain('jarvis-voice-dock')
    expect(dock.className).not.toContain('flex-wrap')
    expect(dock.firstElementChild?.getAttribute('data-testid')).toBe('jarvis-mic-meter')
    expect(readJarvisCss('glass.css')).toMatch(/\.jarvis-voice-dock\s*\{[^}]*flex-wrap:\s*nowrap/)
  })
})
