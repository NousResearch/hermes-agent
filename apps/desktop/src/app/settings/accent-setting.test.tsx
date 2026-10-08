import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test } from 'vitest'

import { en } from '@/i18n/en'
import { accentPref } from '@/themes/accent-pref'
import { ThemeProvider } from '@/themes/context'

import { AccentSetting } from './accent-setting'

const a = en.settings.appearance

const mount = () =>
  render(
    <ThemeProvider>
      <AccentSetting id="accent" />
    </ThemeProvider>
  )

const pressed = (name: string) => screen.getByRole('button', { name }).getAttribute('aria-pressed')

beforeEach(() => window.localStorage.clear())

afterEach(cleanup)

test('a swatch pick persists for the profile and Theme default clears it', () => {
  mount()

  expect(pressed(a.accentThemeDefault)).toBe('true')
  expect(screen.queryByRole('button', { name: 'Nous blue' })).toBeNull()

  fireEvent.click(screen.getByRole('button', { name: 'Barbie pink' }))

  expect(accentPref.stored('default')).toBe('pink')
  expect(pressed('Barbie pink')).toBe('true')
  expect(pressed(a.accentThemeDefault)).toBe('false')

  fireEvent.click(screen.getByRole('button', { name: a.accentThemeDefault }))

  expect(accentPref.stored('default')).toBeNull()
})

test('the custom color stores a custom id', () => {
  mount()

  fireEvent.change(screen.getByLabelText(a.accentCustom), { target: { value: '#123456' } })

  expect(accentPref.stored('default')).toBe('custom:#123456')
})

test('a stored pick shows as selected after a remount', () => {
  accentPref.assign('default', 'mono')
  mount()

  expect(pressed('Mono')).toBe('true')
})
