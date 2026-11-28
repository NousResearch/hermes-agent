import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as RemoteSetup from '@/components/remote-setup/use-remote-setup'
import type { DesktopConnectionsRegistry } from '@/global'
import { deferred } from '@/test/deferred'

import { ConnectionsRegistrySection } from './connections-registry'

const setups = vi.hoisted(() => ({ count: 0 }))

// Counts every render that sets up the credential draft (the editor's sign-in, probe and token).
vi.mock('@/components/remote-setup/use-remote-setup', async importOriginal => {
  const real = await importOriginal<typeof RemoteSetup>()

  return {
    ...real,
    useRemoteSetup: (options: RemoteSetup.RemoteSetupOptions) => {
      setups.count += 1

      return real.useRemoteSetup(options)
    }
  }
})

const registry: DesktopConnectionsRegistry = {
  connections: [
    {
      authMode: 'oauth',
      id: 'alpha',
      kind: 'cloud',
      label: 'Alpha',
      tokenPreview: null,
      tokenSet: true,
      url: 'https://alpha.agents.example'
    }
  ],
  primary: 'alpha',
  secureTokenStorage: true,
  version: 2
}

function mount(managedByHost: boolean) {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      connections: {
        list: vi.fn(async () => registry),
        managedByHost,
        remove: vi.fn(),
        save: vi.fn(),
        setLaunchMode: vi.fn(),
        setPrimary: vi.fn(),
        test: vi.fn()
      }
    }
  })
  render(<ConnectionsRegistrySection />)
}

beforeEach(() => {
  setups.count = 0
})

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('the connection editor', () => {
  it('is never set up on a list the host manages for the account', async () => {
    mount(true)

    expect(await screen.findByText('Alpha')).toBeTruthy()
    expect(setups.count).toBe(0)
  })

  it('is set up only once Add connection opens it on a list this page edits', async () => {
    mount(false)

    fireEvent.click(await screen.findByText('Add connection'))

    expect(await screen.findByPlaceholderText('Homelab')).toBeTruthy()
    expect(setups.count).toBeGreaterThan(0)
  })
  // A save still in flight when another row's Edit opens a new draft answers for its own draft only.
  it("closes only the draft that started a save, not one opened while it was in flight", async () => {
    const remote = (id: string, label: string) => ({
      authMode: 'token' as const,
      headerNames: [],
      id,
      kind: 'remote' as const,
      label,
      tokenPreview: '...abcd',
      tokenSet: true,
      url: `https://${id}.example`
    })

    const editable: DesktopConnectionsRegistry = {
      ...registry,
      connections: [remote('alpha', 'Alpha'), remote('beta', 'Beta')]
    }

    const saving = deferred<{ ok: true; registry: DesktopConnectionsRegistry }>()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        connections: {
          list: vi.fn(async () => editable),
          remove: vi.fn(),
          save: vi.fn(() => saving.promise),
          setLaunchMode: vi.fn(),
          setPrimary: vi.fn(),
          test: vi.fn()
        }
      }
    })
    render(<ConnectionsRegistrySection />)

    await screen.findByText('Alpha')
    const [editAlpha, editBeta] = screen.getAllByRole('button', { name: 'Edit' })
    fireEvent.click(editAlpha)
    fireEvent.click(await screen.findByRole('button', { name: 'Save connection' }))
    expect(window.hermesDesktop.connections!.save).toHaveBeenCalledTimes(1)

    fireEvent.click(editBeta)
    expect((screen.getByPlaceholderText('Homelab') as HTMLInputElement).value).toBe('Beta')

    await act(async () => {
      saving.resolve({ ok: true, registry: editable })
      await saving.promise
    })

    expect((screen.getByPlaceholderText('Homelab') as HTMLInputElement).value).toBe('Beta')
  })

  // Each save answers with the whole list: a second one started while the first runs could answer
  // first, and the older answer would then replace the newer list.
  it('starts no second save from another draft while one is in flight', async () => {
    const remote = (id: string, label: string) => ({
      authMode: 'token' as const,
      headerNames: [],
      id,
      kind: 'remote' as const,
      label,
      tokenPreview: '...abcd',
      tokenSet: true,
      url: `https://${id}.example`
    })

    const editable: DesktopConnectionsRegistry = {
      ...registry,
      connections: [remote('alpha', 'Alpha'), remote('beta', 'Beta')]
    }

    const first = deferred<{ ok: true; registry: DesktopConnectionsRegistry }>()
    const save = vi.fn(() => first.promise)

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        connections: {
          list: vi.fn(async () => editable),
          remove: vi.fn(),
          save,
          setLaunchMode: vi.fn(),
          setPrimary: vi.fn(),
          test: vi.fn()
        }
      }
    })
    render(<ConnectionsRegistrySection />)

    await screen.findByText('Alpha')
    const [editAlpha, editBeta] = screen.getAllByRole('button', { name: 'Edit' })
    fireEvent.click(editAlpha)
    fireEvent.click(await screen.findByRole('button', { name: 'Save connection' }))
    fireEvent.click(editBeta)
    expect((screen.getByPlaceholderText('Homelab') as HTMLInputElement).value).toBe('Beta')
    fireEvent.click(screen.getByRole('button', { name: /Sav/ }))

    expect(save).toHaveBeenCalledTimes(1)

    await act(async () => {
      first.resolve({ ok: true, registry: editable })
      await first.promise
    })

    // Once the first answers, the open draft saves.
    save.mockImplementation(async () => ({ ok: true, registry: editable }))
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Save connection' }))
    })
    expect(save).toHaveBeenCalledTimes(2)
  })
})
