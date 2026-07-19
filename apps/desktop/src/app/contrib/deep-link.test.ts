import { beforeEach, describe, expect, it, vi } from 'vitest'

import { handleDesktopDeepLink } from './deep-link'

vi.mock('@/i18n', () => ({
  translateNow: (key: string, ...args: unknown[]) => `${key}:${args.join('|')}`
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/profile', () => ({
  newSessionInProfile: vi.fn(),
  normalizeProfileKey: (name: string | null | undefined) => name?.trim() || 'default',
  refreshProfiles: vi.fn(),
  switchProfile: vi.fn()
}))

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn(),
  requestComposerSubmit: vi.fn()
}))

const notifications = await import('@/store/notifications')
const notify = vi.mocked(notifications.notify)
const notifyError = vi.mocked(notifications.notifyError)
const profileStore = await import('@/store/profile')
const newSessionInProfile = vi.mocked(profileStore.newSessionInProfile)
const refreshProfiles = vi.mocked(profileStore.refreshProfiles)
const switchProfile = vi.mocked(profileStore.switchProfile)
const composer = await import('@/app/chat/composer/focus')
const requestComposerFocus = vi.mocked(composer.requestComposerFocus)
const requestComposerInsert = vi.mocked(composer.requestComposerInsert)
const requestComposerSubmit = vi.mocked(composer.requestComposerSubmit)

const profile = (name: string, isDefault = false) => ({
  has_env: true,
  is_default: isDefault,
  model: null,
  name,
  path: `/profiles/${name}`,
  provider: null,
  skill_count: 0
})

beforeEach(() => {
  vi.clearAllMocks()
})

describe('handleDesktopDeepLink', () => {
  it('keeps blueprint links reviewable in the composer without submitting', async () => {
    await handleDesktopDeepLink({
      kind: 'blueprint',
      name: 'morning-brief',
      params: { city: 'New York', time: '08:00' }
    })

    expect(requestComposerInsert).toHaveBeenCalledWith('/blueprint morning-brief city="New York" time=08:00', {
      mode: 'block',
      target: 'main'
    })
    expect(requestComposerFocus).toHaveBeenCalledWith('main')
    expect(refreshProfiles).not.toHaveBeenCalled()
    expect(newSessionInProfile).not.toHaveBeenCalled()
  })

  it('opens a blank chat through the live per-session profile route', async () => {
    refreshProfiles.mockResolvedValue([profile('default', true), profile('research')])

    await handleDesktopDeepLink({ kind: 'profile', name: 'research', params: { new: '1' } })

    expect(refreshProfiles).toHaveBeenCalledTimes(1)
    expect(newSessionInProfile).toHaveBeenCalledWith('research')
    expect(switchProfile).not.toHaveBeenCalled()
    expect(requestComposerFocus).toHaveBeenCalledWith('main')
    expect(requestComposerInsert).not.toHaveBeenCalled()
    expect(requestComposerSubmit).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
  })

  it('shows a durable error and leaves the current chat untouched when the profile is missing', async () => {
    refreshProfiles.mockResolvedValue([profile('default', true), profile('work')])

    await handleDesktopDeepLink({ kind: 'profile', name: 'missing', params: { new: '1' } })

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'deep-link-profile-missing:missing',
        kind: 'error',
        title: 'desktop.unknownProfile:',
        message: 'desktop.noProfileNamed:missing|default, work'
      })
    )
    expect(newSessionInProfile).not.toHaveBeenCalled()
    expect(requestComposerFocus).not.toHaveBeenCalled()
  })

  it('fails closed when installed profiles cannot be validated', async () => {
    const error = new Error('profiles unavailable')
    refreshProfiles.mockRejectedValue(error)

    await handleDesktopDeepLink({ kind: 'profile', name: 'research', params: { new: '1' } })

    expect(notifyError).toHaveBeenCalledWith(error, 'desktop.setProfileFailed:')
    expect(newSessionInProfile).not.toHaveBeenCalled()
    expect(requestComposerFocus).not.toHaveBeenCalled()
  })

  it('ignores profile payloads that do not request a new chat', async () => {
    await handleDesktopDeepLink({ kind: 'profile', name: 'research', params: { prompt: 'hello' } })

    expect(refreshProfiles).not.toHaveBeenCalled()
    expect(newSessionInProfile).not.toHaveBeenCalled()
    expect(requestComposerInsert).not.toHaveBeenCalled()
  })
})
