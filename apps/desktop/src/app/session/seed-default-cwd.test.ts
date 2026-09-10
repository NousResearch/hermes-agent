import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId, $connection, $currentCwd, $newChatWorkspaceTarget, setCurrentCwdTransient } from '@/store/session'

import { draftNeedsReseed, seedDefaultCwd } from './seed-default-cwd'

const { desktopDefaultCwd } = vi.hoisted(() => ({
  desktopDefaultCwd: vi.fn<() => Promise<{ branch: string; cwd: string } | null>>()
}))

vi.mock('@/lib/desktop-fs', () => ({ desktopDefaultCwd }))

const HOST = 'http://gganbu:9119'

function remote(profile: string) {
  return { baseUrl: HOST, mode: 'remote', profile } as never
}

function rememberedKey(profile: string) {
  return `hermes.desktop.workspace-cwd.remote.${encodeURIComponent(HOST)}.${encodeURIComponent(profile)}`
}

beforeEach(() => {
  localStorage.clear()
  $connection.set(null)
  $activeSessionId.set(null)
  $newChatWorkspaceTarget.set(undefined)
  setCurrentCwdTransient('')
  desktopDefaultCwd.mockReset()
  desktopDefaultCwd.mockResolvedValue(null)
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    sanitizeWorkspaceCwd: vi.fn(async (cwd: string) => ({ cwd, sanitized: false })),
    settings: { getDefaultProjectDir: vi.fn(async () => ({ dir: '' })) }
  }
})

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  $connection.set(null)
  setCurrentCwdTransient('')
})

describe('draftNeedsReseed', () => {
  it('is true only for a detached, un-targeted draft on a remote connection', () => {
    expect(draftNeedsReseed(remote('default'))).toBe(true)
  })

  it('leaves local drafts alone — a bare local chat is detached by design', () => {
    expect(draftNeedsReseed({ mode: 'local' } as never)).toBe(false)
    expect(draftNeedsReseed(null)).toBe(false)
  })

  it('never touches an open session, an attached draft, or an explicitly detached one', () => {
    $activeSessionId.set('20260827_050914_0009ee')
    expect(draftNeedsReseed(remote('default'))).toBe(false)
    $activeSessionId.set(null)

    setCurrentCwdTransient('/home/user/repo')
    expect(draftNeedsReseed(remote('default'))).toBe(false)
    setCurrentCwdTransient('')

    // workspaceTarget === null is the "start detached" request (startFreshSessionDraft).
    $newChatWorkspaceTarget.set(null)
    expect(draftNeedsReseed(remote('default'))).toBe(false)
  })
})

describe('seedDefaultCwd', () => {
  it('seeds the remembered workspace of the PUBLISHED profile after a switch (#49293)', async () => {
    // The fresh draft was created while $connection still named the outgoing
    // profile, which never had a remembered workspace → it resolved to ''.
    localStorage.setItem(rememberedKey('default'), '/home/user/repo')
    $connection.set(remote('researcher'))
    setCurrentCwdTransient('')

    // The swap resolves and the descriptor for the target profile lands.
    $connection.set(remote('default'))
    expect(draftNeedsReseed($connection.get())).toBe(true)

    await seedDefaultCwd()

    expect($currentCwd.get()).toBe('/home/user/repo')
  })

  it('falls back to the backend default when the profile has no remembered workspace', async () => {
    desktopDefaultCwd.mockResolvedValue({ branch: 'main', cwd: '/home/user/default-project' })
    $connection.set(remote('researcher'))

    await seedDefaultCwd()

    expect($currentCwd.get()).toBe('/home/user/default-project')
  })

  it('does nothing once the caller has been superseded', async () => {
    localStorage.setItem(rememberedKey('default'), '/home/user/repo')
    desktopDefaultCwd.mockResolvedValue({ branch: 'main', cwd: '/home/user/default-project' })
    $connection.set(remote('default'))

    await seedDefaultCwd(() => false)

    expect($currentCwd.get()).toBe('')
  })
})
