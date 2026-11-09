// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeConnectionId } from '@/store/connections'
import { $previewTabs } from '@/store/preview'
import { $activeGatewayProfile } from '@/store/profile'
import { setSessions } from '@/store/session'
import type * as SessionRouterModule from '@/store/session-request-router'

import { PreviewVaultButton, resolveTabVaultOwner, vaultOwnerKey } from './preview-vault-picker'

// Radix menus need these jsdom polyfills to open.
beforeEach(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => {}
  Element.prototype.releasePointerCapture ??= () => {}
  Element.prototype.scrollIntoView ??= () => {}
})

const openMenu = (rendered: ReturnType<typeof render>) =>
  fireEvent.pointerDown(rendered.getByRole('button', { name: 'Fill login' }), { button: 0, ctrlKey: false })

// Capture every owner-scoped RPC so a test can assert WHICH owner each vault
// call was routed to (the fix routes by the tab's owner, never the ambient view).
const calls: Array<{ method: string; owner: unknown; params: Record<string, unknown> }> = []
let resolvers: Array<(value: unknown) => void> = []

vi.mock('@/store/session-request-router', async () => {
  const actual = await vi.importActual<typeof SessionRouterModule>('@/store/session-request-router')

  return {
    ...actual,
    requestForSessionProfile: (owner: unknown, _ambient: unknown, method: string, params: Record<string, unknown> = {}) =>
      new Promise(resolve => {
        calls.push({ method, owner, params })
        resolvers.push(resolve)
      })
  }
})

vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

vi.mock('./preview-vault', () => ({
  forgetPreviewVaultBinding: vi.fn(),
  openUserPreviewVaultBinding: vi.fn(() => 'bind-target')
}))

vi.mock('@/store/recovery-requests', () => ({ requestRoute: vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      preview: {
        web: {
          vaultEmpty: 'No logins',
          vaultFill: 'Fill login',
          vaultFillFailed: 'Fill failed',
          vaultFilled: (label: string) => `Filled ${label}`,
          vaultLoadFailed: 'Load failed',
          vaultLoading: 'Loading…',
          vaultOpenSettings: 'Open settings',
          vaultPageNotReady: 'Page not ready',
          vaultTitle: (host: string) => `Logins for ${host}`,
          vaultUnlock: (name: string) => `Unlock ${name}`
        }
      },
      settings: {
        vault: {
          sources: {
            masterPasswordPlaceholder: 'Master password',
            unlock: 'Unlock',
            unlockDescription: 'Enter your master password.',
            unlocked: (name: string) => `Unlocked ${name}`,
            unlocking: 'Unlocking…',
            unlockTitle: (name: string) => `Unlock ${name}`
          }
        }
      }
    }
  })
}))

const settle = async (value: unknown) => {
  const resolve = resolvers.shift()
  await act(async () => {
    resolve?.(value)
    await Promise.resolve()
  })
}

// Tab owned by profile A on connection conn-a, saved logins on both profiles.
function seedOwnership() {
  setSessions([
    { id: 'sess-a', profile: 'alpha', connection_id: 'conn-a' },
    { id: 'sess-b', profile: 'beta', connection_id: 'conn-b' }
  ] as never)
  $previewTabs.set([{ id: 'tab-a', target: { kind: 'url', url: 'https://x' }, sessionId: 'sess-a' }] as never)
}

beforeEach(() => {
  calls.length = 0
  resolvers = []
  seedOwnership()
  // Profile B is the ACTIVE view — the ambient the old code wrongly used.
  $activeConnectionId.get() // computed from $connection; irrelevant, see below
  $activeGatewayProfile.set('beta')
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $previewTabs.set([])
  setSessions([])
  $activeGatewayProfile.set('default')
})

describe('resolveTabVaultOwner', () => {
  it('resolves a tab to its OWNING session owner, not the active view', () => {
    const owner = resolveTabVaultOwner('tab-a')

    expect(vaultOwnerKey(owner)).toBe('conn-a::alpha')
  })

  it('falls back to the active view for a session-less tab', () => {
    $previewTabs.set([{ id: 'tab-draft', target: { kind: 'url', url: 'https://x' } }] as never)
    $activeGatewayProfile.set('beta')

    expect(vaultOwnerKey(resolveTabVaultOwner('tab-draft'))).toBe('::beta')
  })
})

describe('PreviewVaultButton owner routing (#150283 P1-1)', () => {
  it('lists and fills through the tab owner A while B is the active view', async () => {
    const rendered = render(<PreviewVaultButton tabId="tab-a" url="https://x.example" />)

    openMenu(rendered)

    await waitFor(() => expect(calls.some(c => c.method === 'vault.list')).toBe(true))

    // Every vault RPC went to A (conn-a::alpha), NEVER the active view beta.
    for (const call of calls) {
      expect(vaultOwnerKey(call.owner as never)).toBe('conn-a::alpha')
    }
  })

  it('drops a stale load reply whose owner changed before it landed', async () => {
    const rendered = render(<PreviewVaultButton tabId="tab-a" url="https://x.example" />)

    // Open starts load #1 while the tab is owned by A (conn-a::alpha).
    openMenu(rendered)
    await waitFor(() => expect(calls.filter(c => c.method === 'vault.list')).toHaveLength(1))

    // The tab's owner changes to B while load #1 is still in flight (the menu
    // stays open). load #1's reply is now for a superseded owner.
    await act(async () => {
      setSessions([{ id: 'sess-a', profile: 'beta', connection_id: 'conn-b' }] as never)
      $activeGatewayProfile.set('beta')
      await Promise.resolve()
    })

    const stale = { backend: 'hermes', id: 'STALE', kind: 'login', label: 'x', origin: 'https://x.example' }

    await act(async () => {
      resolvers[0]!({ items: [stale] })
      resolvers[1]!({ sources: [] })
      await Promise.resolve()
    })

    // The stale login (fetched under owner A) never populates the current list.
    // (Radix portals menu content to document.body, not the render container.)
    expect(document.body.querySelector('[data-vault-handle="STALE"]')).toBeNull()
  })
})

describe('PreviewVaultButton unlock owner fence (#150283 P1-2 SECURITY)', () => {
  // A locked source so the unlock dialog can open.
  const sources = [
    { display_name: 'BW', enabled: true, installed: true, name: 'bitwarden', needs_unlock: true, unlocked: false }
  ]

  async function openUnlockDialogUnderA(rendered: ReturnType<typeof render>) {
    openMenu(rendered)
    await waitFor(() => expect(calls.filter(c => c.method === 'vault.list')).toHaveLength(1))
    await settle({ items: [] }) // vault.list
    await settle({ sources }) // vault.sources
    fireEvent.click(await rendered.findByRole('menuitem', { name: 'Unlock BW' }))

    return rendered.findByPlaceholderText('Master password')
  }

  it('refuses to submit the master password after the owner switches A→B', async () => {
    const rendered = render(<PreviewVaultButton tabId="tab-a" url="https://x.example" />)
    const input = await openUnlockDialogUnderA(rendered)

    fireEvent.change(input, { target: { value: 'hunter2' } })
    calls.length = 0 // ignore the load calls; only care about unlock routing

    // The user switches the tab's owner to B (active view AND the tab now
    // resolves to beta) while the dialog is open.
    await act(async () => {
      setSessions([
        { id: 'sess-a', profile: 'beta', connection_id: 'conn-b' },
        { id: 'sess-b', profile: 'beta', connection_id: 'conn-b' }
      ] as never)
      $activeConnectionId.get()
      $activeGatewayProfile.set('beta')
      await Promise.resolve()
    })

    // Submit: nothing must be sent to B (or anyone). The password is wiped.
    const form = input.closest('form')!
    await act(async () => {
      fireEvent.submit(form)
      await Promise.resolve()
    })

    expect(calls.find(c => c.method === 'vault.unlock')).toBeUndefined()
  })

  it('sends the unlock to the captured owner A when the owner is unchanged', async () => {
    const rendered = render(<PreviewVaultButton tabId="tab-a" url="https://x.example" />)
    const input = await openUnlockDialogUnderA(rendered)

    fireEvent.change(input, { target: { value: 'hunter2' } })
    calls.length = 0

    const form = input.closest('form')!
    await act(async () => {
      fireEvent.submit(form)
      await Promise.resolve()
    })

    const unlock = calls.find(c => c.method === 'vault.unlock')

    expect(unlock).toBeDefined()
    expect(vaultOwnerKey(unlock!.owner as never)).toBe('conn-a::alpha')
    expect(unlock!.params.password).toBe('hunter2')
  })
})

