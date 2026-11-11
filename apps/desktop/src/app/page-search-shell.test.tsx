import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it } from 'vitest'

import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { $shellChoice } from '@/store/shell-variant'

import { PageSearchShell } from './page-search-shell'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

function Page() {
  const [query, setQuery] = useState('')
  const [tab, setTab] = useState('skills')

  return (
    <PageSearchShell
      activeTab={tab}
      onSearchChange={setQuery}
      onTabChange={setTab}
      pageTitle="Capabilities"
      searchPlaceholder="Search capabilities"
      searchValue={query}
      tabs={[
        { id: 'skills', label: 'Skills' },
        { id: 'plugins', label: 'Plugins' }
      ]}
    >
      <output>{`${tab}: ${query}`}</output>
    </PageSearchShell>
  )
}

it('keeps search and tab selection when switching between V2 and classic headers', () => {
  $shellChoice.set(true)
  render(<Page />)
  expect(screen.getByRole('heading', { level: 1, name: 'Capabilities' })).toBeTruthy()
  fireEvent.change(screen.getByPlaceholderText('Search capabilities'), { target: { value: 'calendar' } })
  fireEvent.click(screen.getByRole('button', { name: 'Plugins' }))
  expect(screen.getByRole('status').textContent).toBe('plugins: calendar')

  act(() => $shellChoice.set(false))
  expect(screen.queryByRole('heading')).toBeNull()
  expect(screen.getByRole('status').textContent).toBe('plugins: calendar')
  expect(screen.getByPlaceholderText<HTMLInputElement>('Search capabilities').value).toBe('calendar')
})

it('keeps embedded surfaces compact and names an empty full-page destination', () => {
  $shellChoice.set(true)
  const { rerender } = render(<PageSearchShell searchHidden>Embedded capabilities</PageSearchShell>)
  expect(screen.queryByRole('heading')).toBeNull()
  rerender(
    <PageSearchShell pageTitle="Library" searchHidden>
      No artifacts
    </PageSearchShell>
  )
  expect(screen.getByRole('heading', { level: 1, name: 'Library' })).toBeTruthy()
  expect(screen.getByText('No artifacts')).toBeTruthy()
})

it('returns focus to a V2 page action after closing its task dialog', async () => {
  function SettingsPage() {
    const [open, setOpen] = useState(false)
    return (
      <PageSearchShell pageTitle="Capabilities" searchHidden>
        <button onClick={() => setOpen(true)}>Configure</button>
        <Dialog onOpenChange={setOpen} open={open}>
          <DialogContent>
            <DialogTitle>Configure capability</DialogTitle>
            <input aria-label="Name" autoFocus />
          </DialogContent>
        </Dialog>
      </PageSearchShell>
    )
  }
  $shellChoice.set(true)
  render(<SettingsPage />)
  const opener = screen.getByRole('button', { name: 'Configure' })
  opener.focus()
  fireEvent.click(opener)
  expect(document.activeElement).toBe(screen.getByRole('textbox'))
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' })
  await waitFor(() => expect(document.activeElement).toBe(opener))
})
