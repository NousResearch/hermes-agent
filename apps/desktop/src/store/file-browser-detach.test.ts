import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * When the workspace detaches (cwd cleared), the files rail must close so a
 * stuck/persisted $fileBrowserOpen cannot re-expand the moment session.info
 * adopts a process cwd. Mirrors the controller wiring in
 * apps/desktop/src/app/contrib/controller.tsx.
 */
describe('file browser closes on workspace detach', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('closes a persisted-open files rail when cwd becomes empty', async () => {
    const layout = await import('./layout')
    const session = await import('./session')

    layout.setFileBrowserOpen(true)
    session.setCurrentCwd('/some/project')
    expect(layout.$fileBrowserOpen.get()).toBe(true)

    const syncFileBrowserClosedWhenDetached = (hasWorkspace: boolean) => {
      if (!hasWorkspace) {
        layout.setFileBrowserOpen(false)
      }
    }

    session.setCurrentCwd('')
    syncFileBrowserClosedWhenDetached(Boolean(session.$currentCwd.get().trim()))

    expect(layout.$fileBrowserOpen.get()).toBe(false)
  })

  it('leaves the files rail alone when cwd stays set', async () => {
    const layout = await import('./layout')
    const session = await import('./session')

    layout.setFileBrowserOpen(true)
    session.setCurrentCwd('/some/project')

    const syncFileBrowserClosedWhenDetached = (hasWorkspace: boolean) => {
      if (!hasWorkspace) {
        layout.setFileBrowserOpen(false)
      }
    }

    session.setCurrentCwd('/other/project')
    syncFileBrowserClosedWhenDetached(Boolean(session.$currentCwd.get().trim()))

    expect(layout.$fileBrowserOpen.get()).toBe(true)
  })
})
