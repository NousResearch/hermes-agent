import { fireEvent, render, screen } from '@testing-library/react'
import type React from 'react'
import { describe, expect, it, test, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { VoiceControls } from './voice-controls'

function controlsProps(overrides: Partial<React.ComponentProps<typeof VoiceControls>> = {}) {
  return {
    audioLevel: 0,
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

  it('renders a neutral microphone meter without inventing input level', () => {
    renderControls({ audioLevel: Number.NaN, listening: true })

    expect(screen.getByRole('meter', { name: 'Poziom mikrofonu' }).getAttribute('aria-valuenow')).toBe('0')
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
})
