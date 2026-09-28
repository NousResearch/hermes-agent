import { act, cleanup, fireEvent, render as renderComponent, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import type { DesktopConnectionsRegistry } from '@/global'
import { $findInPage } from '@/store/find-in-page'
import { $machine } from '@/store/machine'

import { ConnectionSwitcher } from './connection-switcher'

const render = (ui: ReactNode) => renderComponent(<SidebarProvider>{ui}</SidebarProvider>)

// Radix menus use pointer capture; jsdom does not implement it.
Element.prototype.hasPointerCapture ??= () => false
Element.prototype.setPointerCapture ??= () => undefined
Element.prototype.releasePointerCapture ??= () => undefined
Element.prototype.scrollIntoView ??= () => undefined
globalThis.ResizeObserver ??= class ResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom<null | string>('local'),
  $connectionsRegistry: atom<DesktopConnectionsRegistry | null>(null),
  $pendingConnectionId: atom<null | string>(null),
  selectConnection: vi.fn(async () => undefined)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      profiles: {
        switchConnectionFailed: (name: string) => `Could not connect to ${name}`,
        switchToConnection: (name: string) => `Switch to ${name}`,
        connectGateway: 'Manage gateways…'
      },
      settings: {
        connections: {
          noSearchResults: 'No gateways match your search.',
          searchPlaceholder: 'Search gateways…',
          kindCloud: 'Hermes Cloud',
          kindLocal: 'Local',
          thisDeviceSuffix: '(this device)',
          kindRemote: 'Remote gateway',
          kindSsh: 'SSH',
          title: 'Registered gateways'
        }
      }
    }
  })
}))

const connectionStore = await import('@/store/connections')
const $activeConnectionId = connectionStore.$activeConnectionId as ReturnType<typeof atom<null | string>>
const $connectionsRegistry = connectionStore.$connectionsRegistry
const $pendingConnectionId = connectionStore.$pendingConnectionId
const selectConnection = vi.mocked(connectionStore.selectConnection)
const onConnect = vi.fn()

const connection = (id: string, label: string, kind: 'local' | 'remote' = 'remote') => ({
  id,
  kind,
  label,
  tokenPreview: null,
  tokenSet: false
})

const registry = (connections: DesktopConnectionsRegistry['connections']): DesktopConnectionsRegistry => ({
  connections,
  primary: connections[0]?.id ?? 'local',
  secureTokenStorage: true,
  version: 2
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $connectionsRegistry.set(null)
  $activeConnectionId.set('local')
  $pendingConnectionId.set(null)
  $machine.set(null)
  $findInPage.set({ active: false, query: '', matchOrdinal: 0, matchCount: 0, focusRequest: 0 })
})

describe('ConnectionSwitcher', () => {
  it('adds no source chrome for a local-only setup', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local')]))
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(screen.queryByRole('group', { name: 'Registered gateways' })).toBeNull()
  })

  it.each([false, true])('switches gateways and opens management (compact=%s)', async compact => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        { ...connection('homelab', 'Homelab'), url: 'https://lab.example.com' },
        connection('work-vps', 'Work VPS')
      ])
    )
    render(<ConnectionSwitcher compact={compact} onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })

    expect(trigger.textContent).toContain('This device')
    expect(trigger.hasAttribute('title')).toBe(false)
    expect(trigger.closest('[data-slot="connection-switcher"]')?.classList.contains('overflow-hidden')).toBe(false)

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect(trigger.getAttribute('data-state')).toBe('open')
    expect(trigger.className).toContain('data-[state=open]:bg-(--ui-control-active-background)')
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Homelab' }))
    expect(selectConnection).toHaveBeenCalledWith('homelab')

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(screen.getByRole('menuitem', { name: 'Manage gateways…' }))
    expect(onConnect).toHaveBeenCalledTimes(1)
    expect(selectConnection).toHaveBeenCalledTimes(1)

    act(() => $activeConnectionId.set('homelab'))
    const remoteTrigger = screen.getByRole('button', { name: 'Registered gateways: Homelab' })
    fireEvent.pointerMove(within(remoteTrigger).getByText('Homelab'), { pointerType: 'mouse' })
    await waitFor(() => expect(screen.getByRole('tooltip').textContent).toBe('Homelab\nhttps://lab.example.com'))
  })

  it.each([
    { label: 'This device', hostname: 'h-mini2', name: 'h-mini2', suffix: '(this device)' },
    { label: 'Studio', hostname: 'h-mini2', name: 'Studio', suffix: '(this device)' },
    { label: 'This device', hostname: '', name: 'This device', suffix: '' }
  ])('names the local device in every surface: $name', ({ label, hostname, name, suffix }) => {
    $connectionsRegistry.set(registry([connection('local', label, 'local'), connection('remote', 'Work')]))
    const { rerender } = render(<ConnectionSwitcher onConnect={onConnect} />)
    act(() =>
      $machine.set({
        hostname,
        ageDays: null,
        arch: 'arm64',
        locale: 'en',
        model: '',
        nvidia: false,
        platform: 'darwin',
        release: '',
        username: ''
      })
    )

    for (const compact of [false, true]) {
      rerender(
        <SidebarProvider>
          <ConnectionSwitcher compact={compact} onConnect={onConnect} />
        </SidebarProvider>
      )
      const fullName = [name, suffix].filter(Boolean).join(' ')
      const trigger = screen.getByRole('button', { name: `Registered gateways: ${fullName}` })
      expect(trigger.querySelector('[data-connection-name]')?.textContent).toBe(name)
      expect(trigger.querySelector('[data-connection-suffix]')?.textContent ?? '').toBe(suffix)
      fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
      const local = screen.getByRole('menuitemradio', { name: fullName })
      expect(local.querySelector('[data-connection-name]')?.textContent).toBe(name)
      expect(local.querySelector('[data-connection-suffix]')?.textContent ?? '').toBe(suffix)
      expect(screen.getByRole('menuitemradio', { name: 'Work' }).querySelector('[data-connection-suffix]')).toBeNull()
      fireEvent.keyDown(globalThis.document, { key: 'Escape' })
    }

    expect($connectionsRegistry.get()?.connections[0].label).toBe(label)
  })

  it.each(['local', 'remote'] as const)(
    'reserves suffix and trailing controls while a long %s name truncates',
    kind => {
      const name = 'Studio workstation with an extraordinarily long device name 1234567890'
      const entry = connection(kind, name, kind)
      $connectionsRegistry.set(registry([entry, connection('other', 'Other')]))
      $activeConnectionId.set(kind)
      const { rerender } = render(<ConnectionSwitcher onConnect={onConnect} />)

      for (const compact of [false, true]) {
        rerender(
          <SidebarProvider>
            <ConnectionSwitcher compact={compact} onConnect={onConnect} />
          </SidebarProvider>
        )

        const trigger = screen.getByRole('button', {
          name: `Registered gateways: ${name}${kind === 'local' ? ' (this device)' : ''}`
        })

        fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })

        const menuRow = screen.getByRole('menuitemradio', {
          name: `${name}${kind === 'local' ? ' (this device)' : ''}`
        })

        for (const row of [trigger, menuRow]) {
          const nameElement = row.querySelector('[data-connection-name]')!
          expect(nameElement.textContent).toBe(name)
          expect(nameElement.classList.contains('min-w-0')).toBe(true)
          expect(nameElement.classList.contains('truncate')).toBe(true)
          const suffix = row.querySelector('[data-connection-suffix]')

          if (kind === 'local') {
            expect(suffix?.textContent).toBe('(this device)')
            expect(suffix?.classList.contains('shrink-0')).toBe(true)
            expect(nameElement.contains(suffix)).toBe(false)
          } else {
            expect(suffix).toBeNull()
          }
        }

        expect(trigger.querySelector('.codicon-chevron-down')?.classList.contains('shrink-0')).toBe(true)
        fireEvent.keyDown(globalThis.document, { key: 'Escape' })
      }
    }
  )

  it('keeps source controls stable while a remote is opening', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    $pendingConnectionId.set('homelab')
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(screen.getByRole('group', { name: 'Registered gateways' }).getAttribute('aria-busy')).toBe('true')
  })

  it('keeps small gateway lists simple and naturally sorted', () => {
    $connectionsRegistry.set(
      registry([
        connection('zulu', 'Zulu'),
        connection('local', 'This device', 'local'),
        connection('studio-10', 'Studio 10'),
        connection('alpha', 'alpha'),
        connection('studio-2', 'Studio 2')
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })

    expect(screen.queryByPlaceholderText('Search gateways…')).toBeNull()
    expect(screen.getAllByRole('menuitemradio').map(item => item.textContent)).toEqual([
      'This device',
      'alpha',
      'Studio 2',
      'Studio 10',
      'Zulu'
    ])
  })

  it('adds search at eight gateways and filters stable results without moving the connect action', async () => {
    $connectionsRegistry.set(
      registry([
        connection('zulu', 'Zulu'),
        connection('local', 'This device', 'local'),
        connection('studio-10', 'Studio 10'),
        connection('alpha', 'Alpha'),
        connection('studio-2', 'Studio 2'),
        connection('work', 'Work VPS'),
        connection('homelab', 'Homelab'),
        connection('cloud', 'Cloud lab')
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })
    fireEvent.pointerDown(trigger, {
      button: 0,
      pointerType: 'mouse'
    })

    const search = screen.getByPlaceholderText('Search gateways…')
    expect(screen.getByRole('menuitem', { name: 'Manage gateways…' })).toBeTruthy()
    expect(screen.getAllByRole('menuitemradio').map(item => item.textContent)).toEqual([
      'This device',
      'Alpha',
      'Cloud lab',
      'Homelab',
      'Studio 2',
      'Studio 10',
      'Work VPS',
      'Zulu'
    ])

    fireEvent.change(search, { target: { value: 'studio 10' } })
    expect(screen.getAllByRole('menuitemradio').map(item => item.textContent)).toEqual(['Studio 10'])

    const result = screen.getByRole('menuitemradio', { name: 'Studio 10' })
    fireEvent.keyDown(search, { key: 'ArrowDown' })
    expect(globalThis.document.activeElement).toBe(result)

    $findInPage.set({ active: true, query: '', matchOrdinal: 0, matchCount: 0, focusRequest: 0 })
    result.focus()
    fireEvent.keyDown(result, { key: 'f', metaKey: true })
    expect(globalThis.document.activeElement).toBe(search)
    expect($findInPage.get().active).toBe(false)

    fireEvent.keyDown(search, { key: 'Escape' })
    expect(screen.queryByPlaceholderText('Search gateways…')).toBeNull()
    await waitFor(() => expect(globalThis.document.activeElement).toBe(trigger))

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect((screen.getByPlaceholderText('Search gateways…') as HTMLInputElement).value).toBe('')

    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Studio 10' }))
    expect(selectConnection).toHaveBeenCalledWith('studio-10')

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect((screen.getByPlaceholderText('Search gateways…') as HTMLInputElement).value).toBe('')
  })

  it('explains an empty large-list search', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        ...Array.from({ length: 7 }, (_, index) => connection(`remote-${index}`, `Remote ${index}`))
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: This device' }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.change(screen.getByPlaceholderText('Search gateways…'), { target: { value: 'missing' } })

    expect(screen.getByText('No gateways match your search.')).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: 'Manage gateways…' })).toBeTruthy()
  })

  // The window lifecycle owns IPC; this consumer paints its published cache.
  it('repaints the menu after the registry changes without reload', () => {
    const before = registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')])

    const after = registry([
      connection('local', 'This device', 'local'),
      connection('homelab', 'Homelab'),
      connection('w2-probe', 'W2Probe')
    ])

    $connectionsRegistry.set(before)
    render(<ConnectionSwitcher onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect(screen.queryByRole('menuitemradio', { name: 'W2Probe' })).toBeNull()
    fireEvent.keyDown(globalThis.document, { key: 'Escape' })

    act(() => $connectionsRegistry.set(after))

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect(screen.getByRole('menuitemradio', { name: 'W2Probe' })).toBeTruthy()
  })
})
