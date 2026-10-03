import { afterEach, beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  vi.resetModules()
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
  document.body.replaceChildren()
})

afterEach(async () => {
  const { clearNotifications } = await import('./notifications')
  clearNotifications()
  document.body.replaceChildren()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

it.each(['session', 'group'] as const)('reports unresolved %s ownership without detaching the tab', async kind => {
  await import('./session')
  await import('./session-states')
  const preview = await import('./preview')
  const { $notifications, clearNotifications } = await import('./notifications')
  const { TRANSLATIONS, setRuntimeI18nLocale } = await import('@/i18n')

  const { capturePreviewAnnotateDestination, readPreviewAnnotateDestination } =
    await import('@/lib/preview-annotate/handoff')

  const surface = document.createElement('div')

  if (kind === 'session') {
    surface.dataset.composerTarget = 'main'
    surface.dataset.composerSurfaceId = 'unknown-session-surface'
    surface.dataset.browserSessionId = 'unknown-session'
  } else {
    surface.dataset.previewAnnotateDestination = 'group'
    surface.dataset.previewAnnotateGroup = 'Legacy room'
    surface.dataset.previewAnnotateComposerKey = 'name:Legacy room'
    surface.dataset.previewAnnotateConnectionId = 'room-host'
    surface.dataset.previewAnnotateProfile = 'default'
  }

  document.body.append(surface)
  expect(capturePreviewAnnotateDestination()).not.toBeNull()
  expect(capturePreviewAnnotateDestination()?.conversation).toBeUndefined()
  const openBrowserWindow = vi.fn().mockResolvedValue({ ok: true })
  Object.assign(window, { hermesDesktop: { browserWorkspace: {}, openBrowserWindow } })
  preview.newBrowserTab()
  const tabs = preview.$previewTabs.get()
  const id = tabs.at(-1)!.id

  for (const locale of ['en', 'de'] as const) {
    setRuntimeI18nLocale(locale)
    clearNotifications()
    preview.popOutBrowserTab(id)
    await Promise.resolve()
    expect(openBrowserWindow).not.toHaveBeenCalled()
    expect(preview.$poppedBrowserTabIds.get().has(id)).toBe(false)
    expect(preview.$previewTabs.get()).toBe(tabs)
    expect(readPreviewAnnotateDestination(id)).toBeNull()
    expect($notifications.get()).toHaveLength(1)
    expect($notifications.get()[0]).toMatchObject({
      kind: 'error',
      title: TRANSLATIONS[locale].preview.popOutFailed,
      message: TRANSLATIONS[locale].preview.popOutOwnerUnavailable
    })
  }
})
