import { describe, expect, it } from 'vitest'

import {
  browserBack,
  browserForward,
  browserNavigate,
  browserSessionRoot,
  browserUp,
  getBrowserWorkspace,
  resetBrowserWorkspace,
  syncBrowserWorkspace
} from './browser-workspace'

describe('independent browser workspace', () => {
  it('seeds from the session once without following later run cwd updates', () => {
    resetBrowserWorkspace()
    syncBrowserWorkspace('/session/a', 'local:default')
    browserNavigate('/chosen')
    syncBrowserWorkspace('/agent/new-cwd', 'local:default')

    expect(getBrowserWorkspace()).toMatchObject({ location: '/chosen', sessionRoot: '/agent/new-cwd' })
  })

  it('resets safely when the connection identity changes', () => {
    resetBrowserWorkspace()
    syncBrowserWorkspace('/local', 'local:default')
    browserNavigate('/picked')
    syncBrowserWorkspace('/remote', 'remote:prod:https://example.test')

    expect(getBrowserWorkspace()).toMatchObject({
      back: [],
      connectionKey: 'remote:prod:https://example.test',
      forward: [],
      location: '/remote'
    })
  })

  it('supports bounded back/forward/up/session-root history without file contents', () => {
    resetBrowserWorkspace()
    syncBrowserWorkspace('/repo', 'local:default')
    browserNavigate('/repo/src')
    browserNavigate('/repo/src/app')
    browserBack()
    expect(getBrowserWorkspace().location).toBe('/repo/src')
    browserForward()
    expect(getBrowserWorkspace().location).toBe('/repo/src/app')
    browserUp()
    expect(getBrowserWorkspace().location).toBe('/repo/src')
    browserSessionRoot()
    expect(getBrowserWorkspace().location).toBe('/repo')
    expect(JSON.stringify(getBrowserWorkspace())).not.toContain('content')
  })
})
