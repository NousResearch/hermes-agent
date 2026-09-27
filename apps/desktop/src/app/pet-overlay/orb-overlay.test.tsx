import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { moveOrb } from '@/app/jarvis/desktop-orb-state'
import { TRANSLATIONS } from '@/i18n'

import { OrbOverlay } from './orb-overlay'

vi.mock('@/app/jarvis/core', () => ({ JarvisCore: ({ voice }: { voice: string }) => <div data-testid="orb-voice">{voice}</div> }))
afterEach(() => vi.unstubAllGlobals())

test('orb mirrors the live conversation and sends controls to its owner without starting another session', () => {
  const control = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { petOverlay: { control, setIgnoreMouse: vi.fn() } } })
  const state = { active: false, connected: true, locale: 'pl' as const, voice: 'idle' as const, task: 'idle' as const }
  const view = render(<OrbOverlay state={state} />)
  fireEvent.click(screen.getByRole('button', { name: 'Rozpocznij rozmowę' }))
  expect(control).toHaveBeenLastCalledWith({ type: 'orb-toggle-voice' })
  view.rerender(<OrbOverlay state={{ ...state, active: true, voice: 'speaking' }} />)
  expect(screen.getByTestId('orb-voice').textContent).toBe('speaking')
  view.rerender(<OrbOverlay state={{ ...state, active: true, task: 'approval' }} />)
  expect(screen.getByText(TRANSLATIONS.pl.jarvisShell.dashboard.core.task.approval)).toBeTruthy()
  view.rerender(<OrbOverlay state={{ ...state, active: true, task: 'failed' }} />)
  expect(screen.getByText('Sprawdź rozmowę w aplikacji')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Zakończ rozmowę' }))
  expect(control).toHaveBeenLastCalledWith({ type: 'orb-toggle-voice' })
  fireEvent.click(screen.getByRole('button', { name: 'Schowaj kulę' }))
  expect(control).toHaveBeenLastCalledWith({ type: 'pop-in' })
  expect(moveOrb({ x: -400, y: 60 }, { x: -320, y: 120 }, { x: -190, y: 160 })).toEqual({ x: -270, y: 100 })
})
