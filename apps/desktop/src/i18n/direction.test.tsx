import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { Select, SelectTrigger, SelectValue } from '@/components/ui/select'

import { I18nProvider, useI18n } from './context'

function DirectionProbe() {
  const { setLocale } = useI18n()

  return (
    <>
      <button onClick={() => void setLocale('en')} type="button">
        English
      </button>
      <Select>
        <SelectTrigger aria-label="Choice">
          <SelectValue />
        </SelectTrigger>
      </Select>
      <DropdownMenu modal={false} open>
        <DropdownMenuContent>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>Options</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>Nested options</DropdownMenuSubContent>
          </DropdownMenuSub>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  )
}

afterEach(cleanup)

it.each([
  ['ar', 'rtl', 'ArrowLeft'],
  ['en', 'ltr', 'ArrowRight']
] as const)('shares %s direction with portaled menus, selects and submenu navigation', async (locale, dir, key) => {
  render(
    <I18nProvider configClient={null} initialLocale={locale}>
      <DirectionProbe />
    </I18nProvider>
  )

  expect(screen.getByRole('menu').getAttribute('dir')).toBe(dir)
  expect(screen.getByRole('combobox').getAttribute('dir')).toBe(dir)
  const trigger = screen.getByRole('menuitem', { name: 'Options' })
  trigger.focus()
  fireEvent.keyDown(trigger, { key })
  await waitFor(() => expect(screen.getAllByRole('menu')).toHaveLength(2))
  expect(screen.getAllByRole('menu').every(menu => menu.getAttribute('dir') === dir)).toBe(true)
})

it('updates existing Radix consumers when the active language changes', async () => {
  render(
    <I18nProvider configClient={null} initialLocale="ar">
      <DirectionProbe />
    </I18nProvider>
  )
  fireEvent.click(screen.getByRole('button', { name: 'English' }))
  await waitFor(() => expect(screen.getByRole('menu').getAttribute('dir')).toBe('ltr'))
  expect(screen.getByRole('combobox').getAttribute('dir')).toBe('ltr')
})
