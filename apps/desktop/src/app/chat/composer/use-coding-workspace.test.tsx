import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $codingWorkspaceDrafts, resetCodingWorkspaceDraft } from '@/store/coding-workspaces'
import { $connection } from '@/store/session'

import { useCodingWorkspace } from './use-coding-workspace'

const mocks = vi.hoisted(() => ({ read: vi.fn(), register: vi.fn(), inspect: vi.fn(), set: vi.fn(), pick: vi.fn() }))
vi.mock('@/hermes', () => ({ getHermesConfigRecord: (...args: unknown[]) => mocks.read(...args), profileScopeKey: (s: unknown) => JSON.stringify(s) }))
vi.mock('@/store/profile', async () => {
 const { atom } = await import('nanostores')

 return { $newChatProfile: atom('coder'), $newChatRoute: atom(null), $newChatConnectionId: atom('local'), $activeGatewayProfile: atom('coder'),
 resolveNewChatBackendOwner: () => ({ connectionId: 'local', profile: 'coder' }) }
})
vi.mock('@/store/session', async () => ({ $connection: (await import('nanostores')).atom({ connectionId: 'local' }) }))
vi.mock('@/store/coding-workspaces', async importOriginal => ({
 ...(await importOriginal<Record<string, unknown>>()),
 setCodingWorkspaceIntent: (...a: unknown[]) => mocks.set(...a), inspectCodingWorkspace: (...a: unknown[]) => mocks.inspect(...a),
 registerCodingWorkspaceFolder: (...a: unknown[]) => mocks.register(...a)
}))
vi.mock('@/lib/desktop-fs', () => ({ selectDesktopPaths: (...a: unknown[]) => mocks.pick(...a) }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/i18n', async () => { const { en } = await import('@/i18n/en');

 return { useI18n: () => ({ t: en }) } })

function wrapper({ children }: { children: ReactNode }) {
 const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

 return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

afterEach(() => { cleanup(); $codingWorkspaceDrafts.set({}); vi.clearAllMocks() })

describe('workspace composer opt-in', () => {
 it.each(['current', 'worktree', undefined])('honors owner-scoped default checkout %s for Use as project', async mode => {
  mocks.read.mockResolvedValue({ desktop: { coding: { show_controls: true, default_checkout: mode } } })
  mocks.register.mockResolvedValue({ id: 'project' })
  mocks.inspect.mockResolvedValue({ path: '/reference-folder', repoRoot: '/reference-folder', worktrees: [] })
  const { result } = renderHook(() => useCodingWorkspace(null, true), { wrapper })
  await waitFor(() => expect(result.current.visible).toBe(true))
  await act(async () => result.current.selectFolder('/reference-folder'))
  expect(mocks.read).toHaveBeenCalledWith({ connectionId: 'local', profile: 'coder' })
  expect(mocks.set).toHaveBeenCalledWith(result.current.owner, { projectId: 'project', path: '/reference-folder', mode: mode ?? 'worktree' })
  expect(mocks.pick).not.toHaveBeenCalled()
 })
 it('drops one-chat opt-in when New chat resets the same owner without remounting', async () => {
  mocks.read.mockResolvedValue({})
  const { result } = renderHook(() => useCodingWorkspace(null, true), { wrapper })
  await waitFor(() => expect(mocks.read).toHaveBeenCalled())
  act(() => result.current.enable())
  expect(result.current.visible).toBe(true)
  act(() => resetCodingWorkspaceDraft(result.current.owner!))
  expect(result.current.visible).toBe(false)
  expect(result.current.draft?.intent).toBeNull()
  expect(mocks.inspect).not.toHaveBeenCalled()
  expect(mocks.register).not.toHaveBeenCalled()
 })
 it('does not browse an ambient filesystem belonging to another owner', async () => {
  mocks.read.mockResolvedValue({})
  const { result } = renderHook(() => useCodingWorkspace('draft:one', true), { wrapper })
  act(() => $connection.set({ ...$connection.get()!, connectionId: 'another-connection' }))
  await act(async () => result.current.selectFolder())
  expect(mocks.pick).not.toHaveBeenCalled()
  expect(mocks.register).not.toHaveBeenCalled()
  act(() => $connection.set({ ...$connection.get()!, connectionId: 'local' }))
 })

 it('normalizes an empty new-chat scope to the same __new__ key used by first Send', async () => {
  mocks.read.mockResolvedValue({ desktop: { coding: { show_controls: true } } })
  const { result } = renderHook(() => useCodingWorkspace(null, true), { wrapper })
  await waitFor(() => expect(result.current.visible).toBe(true))
  expect(result.current.owner).toEqual({ connectionId: 'local', profile: 'coder', draftKey: '__new__' })
 })

 it('keeps ordinary drafts untouched until one-chat opt-in, which does not inspect or register', async () => {
  mocks.read.mockResolvedValue({})
  const { result } = renderHook(() => useCodingWorkspace('draft:one', true), { wrapper })
  await waitFor(() => expect(mocks.read).toHaveBeenCalled())
  expect(result.current.visible).toBe(false)
  act(() => result.current.enable())
  expect(result.current.visible).toBe(true)
  expect(mocks.inspect).not.toHaveBeenCalled()
  expect(mocks.register).not.toHaveBeenCalled()
  expect(mocks.set).not.toHaveBeenCalled()
 })
 it('ignores a folder picker from the previous New chat on the same owner', async () => {
  mocks.read.mockResolvedValue({})
  let finish!: (value: string[]) => void
  mocks.pick.mockImplementation(() => new Promise<string[]>(resolve => { finish = resolve }))
  const { result } = renderHook(() => useCodingWorkspace(null, true), { wrapper })
  act(() => result.current.enable())
  let picked!: Promise<void>
  act(() => { picked = result.current.selectFolder() })
  act(() => resetCodingWorkspaceDraft(result.current.owner!))
  await act(async () => { finish(['/repo']); await picked })
  expect(mocks.register).not.toHaveBeenCalled()
  expect(result.current.visible).toBe(false)
 })
 it('does not promote a reference when New chat resets an initially absent draft during registration', async () => {
  mocks.read.mockResolvedValue({})
  let finish!: (value: { id: string }) => void
  mocks.register.mockImplementation(() => new Promise<{ id: string }>(resolve => { finish = resolve }))
  const { result } = renderHook(() => useCodingWorkspace(null, true), { wrapper })
  expect(result.current.draft).toBeUndefined()
  let registering!: Promise<void>
  act(() => { registering = result.current.selectFolder('/reference-folder') })
  expect(mocks.register).toHaveBeenCalled()
  act(() => resetCodingWorkspaceDraft(result.current.owner!))
  await act(async () => { finish({ id: 'registered-project' }); await registering })
  expect(mocks.set).not.toHaveBeenCalled()
  expect(mocks.inspect).not.toHaveBeenCalled()
  expect(result.current.draft?.intent).toBeNull()
  expect(result.current.visible).toBe(false)
 })
 it('preserves attachment references and ignores folder picker completion after a draft swap', async () => {
  mocks.read.mockResolvedValue({ desktop: { coding: { show_controls: true } } })
  let finish!: (value: string[]) => void
  mocks.pick.mockImplementation(() => new Promise<string[]>(resolve => { finish = resolve }))
  const { result, rerender } = renderHook(({ key }) => useCodingWorkspace(key, true), { wrapper, initialProps: { key: 'draft:one' } })
  let picked!: Promise<void>
  act(() => { picked = result.current.selectFolder() })
  rerender({ key: 'draft:two' })
  await act(async () => { finish(['/repo']); await picked })
  expect(mocks.register).not.toHaveBeenCalled()
  expect(mocks.set).not.toHaveBeenCalled()
 })
})
