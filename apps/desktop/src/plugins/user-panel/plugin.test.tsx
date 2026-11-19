// @vitest-environment jsdom
import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// @ts-expect-error - the bundled plugin is plain JavaScript without declarations.
import plugin from './plugin.js'

const mocks = vi.hoisted(() => ({
  notifyError: vi.fn(),
  signIn: vi.fn(),
  signOut: vi.fn(),
  snapshot: null as unknown as { set: (value: HermesSdk.ClerkSessionSnapshot) => void }
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  mocks.snapshot = sdk.atom({
    activeOrganization: null,
    enabled: false,
    error: null,
    organizations: [],
    status: 'disabled',
    tokenSyncError: null,
    user: null
  })

  return {
    ...sdk,
    host: {
      ...sdk.host,
      clerk: {
        createOrganization: vi.fn(),
        refresh: vi.fn(),
        setActiveOrganization: vi.fn(),
        signIn: mocks.signIn,
        signOut: mocks.signOut,
        snapshot: mocks.snapshot
      },
      notifyError: mocks.notifyError
    }
  }
})

const user = {
  email: null,
  fullName: 'Ada Lovelace',
  id: 'user_1',
  imageUrl: null,
  username: 'ada'
}

const organization = {
  id: 'org_1',
  imageUrl: 'https://example.test/acme.png',
  membersCount: 3,
  name: 'Acme',
  role: 'admin',
  slug: 'acme'
}

const setSnapshot = (patch: Partial<HermesSdk.ClerkSessionSnapshot>) => {
  act(() => {
    mocks.snapshot.set({
      activeOrganization: null,
      enabled: true,
      error: null,
      organizations: [],
      status: 'signed-out',
      tokenSyncError: null,
      user: null,
      ...patch
    })
  })
}

let mount: () => ReactNode

beforeAll(() => {
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  Element.prototype.scrollIntoView = vi.fn()

  let renderPanel: (() => ReactNode) | undefined

  plugin.register({
    os: { openExternal: vi.fn() },
    register: (contribution: { render: () => ReactNode }) => {
      renderPanel = contribution.render
    }
  })

  mount = () => renderPanel!()
})

beforeEach(() => {
  mocks.signIn.mockReset().mockResolvedValue(undefined)
  mocks.signOut.mockReset().mockResolvedValue(undefined)
  mocks.notifyError.mockReset()
})

afterEach(cleanup)

const openMenu = (name: RegExp = /Open account menu/) => {
  const trigger = screen.getByRole('button', { name })

  expect(trigger.hasAttribute('disabled')).toBe(false)

  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)

  return trigger
}

const selectRow = (name: string) => {
  const row = screen.getByRole('menuitem', { name })

  fireEvent.pointerDown(row, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(row, { button: 0, pointerType: 'mouse' })
  fireEvent.click(row)

  return row
}

const menuLabels = () => screen.getAllByRole('menuitem').map(item => item.textContent?.trim())

const PUBLIC_ROWS = ['Settings', 'AI Credits', 'Buy credits', 'Pricing', 'Book a demo']

describe('user panel menu', () => {
  it('shows the menu while Clerk is still loading', () => {
    setSnapshot({ status: 'loading' })

    render(mount())

    expect(screen.queryByLabelText('Loading account')).toBeNull()

    openMenu()

    expect(menuLabels()).toEqual(['Log in', ...PUBLIC_ROWS])
  })

  it('shows the menu when Clerk is not configured at all', () => {
    setSnapshot({ enabled: false, status: 'disabled' })

    render(mount())
    openMenu()

    expect(menuLabels()).toEqual(['Log in', ...PUBLIC_ROWS])
  })

  it('opens Clerk sign-in from the signed-out menu', async () => {
    setSnapshot({ status: 'signed-out' })

    render(mount())
    openMenu()
    selectRow('Log in')

    await waitFor(() => expect(mocks.signIn).toHaveBeenCalledTimes(1))
  })

  it('shows the organization switcher and logs out from the bottom of the menu', async () => {
    setSnapshot({
      activeOrganization: organization,
      organizations: [organization],
      status: 'signed-in',
      user
    })

    render(mount())
    openMenu(/Open account menu for Acme/)

    expect(menuLabels()).toEqual(['Acme', ...PUBLIC_ROWS, 'Log out'])

    selectRow('Log out')

    await waitFor(() => expect(mocks.signOut).toHaveBeenCalledTimes(1))
  })
})
