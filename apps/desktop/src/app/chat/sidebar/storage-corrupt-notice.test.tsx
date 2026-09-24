import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $corruptSessionStores, setCorruptSessionStores } from '@/store/session'

import { SidebarStorageCorruptNotice } from './section-states'

const openExternalLink = vi.fn()

vi.mock('@/lib/external-link', () => ({ openExternalLink: (href: string) => openExternalLink(href) }))

beforeEach(() => {
  $corruptSessionStores.set([])
  openExternalLink.mockClear()
})

afterEach(cleanup)

// A structurally corrupt state.db used to render as "No sessions yet" (#72046):
// the list went empty and nothing said the store was damaged.
describe('SidebarStorageCorruptNotice', () => {
  it('renders nothing while every store is healthy', () => {
    setCorruptSessionStores({})
    render(<SidebarStorageCorruptNotice />)

    expect(screen.queryByTestId('storage-corrupt-notice')).toBeNull()
  })

  it('names the damaged profile and gives the non-destructive recovery path', () => {
    setCorruptSessionStores({ default: 'corrupt' })
    render(<SidebarStorageCorruptNotice />)

    const notice = screen.getByTestId('storage-corrupt-notice')

    expect(notice.getAttribute('role')).toBe('alert')
    expect(notice.textContent).toContain('Session database is damaged')
    expect(notice.textContent).toContain('for default')
    expect(notice.textContent).toContain('were not deleted')
    expect(notice.textContent).toContain('hermes sessions recover --source <state.db> --inspect-only')
    // No blanket "run repair" advice: structural damage goes to inspect/restore first.
    expect(notice.textContent).not.toContain('sessions repair')

    screen.getByRole('button', { name: /recovery guide/i }).click()
    expect(openExternalLink).toHaveBeenCalledWith(expect.stringContaining('/user-guide/session-storage-recovery'))
  })

  it('keeps atom identity when a refresh reports the same stores', () => {
    setCorruptSessionStores({ work: 'corrupt', default: 'corrupt' })
    const first = $corruptSessionStores.get()

    setCorruptSessionStores({ default: 'corrupt', work: 'corrupt' })
    expect($corruptSessionStores.get()).toBe(first)
    expect(first).toEqual(['default', 'work'])

    setCorruptSessionStores(undefined)
    expect($corruptSessionStores.get()).toEqual([])
  })
})
