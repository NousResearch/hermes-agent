import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { OVERLAY_VIEWS, SETTINGS_ROUTE, appViewForPath } from './routes'

/**
 * Regression for #43825: overlay navigation must not unmount ChatView.
 * Mounting chat on the overlay Route still remounts when switching / → /settings;
 * chat is rendered outside <Routes> for non-full-page views instead.
 */
describe('overlay routes preserve chat mount', () => {
  it('settings is an overlay view, not chat', () => {
    expect(OVERLAY_VIEWS.has('settings')).toBe(true)
    expect(appViewForPath(SETTINGS_ROUTE)).toBe('settings')
  })

  it('desktop controller keeps chat outside Routes for overlay navigation', () => {
    const controllerPath = join(dirname(fileURLToPath(import.meta.url)), 'desktop-controller.tsx')
    const source = readFileSync(controllerPath, 'utf8')

    expect(source).toContain('fullPageMainView')
    expect(source).toContain('{!fullPageMainView ? chatView : null}')
    expect(source).toContain('<Route element={null} index />')
    expect(source).toContain('<Route element={null} path=":sessionId" />')
    expect(source).not.toMatch(/<Route element=\{chatView\}/)

    for (const route of ['settings', 'command-center', 'agents', 'cron', 'profiles']) {
      expect(source).toContain(`<Route element={null} path="${route}" />`)
    }

    for (const route of ['skills', 'messaging', 'artifacts']) {
      expect(source).not.toContain(`<Route element={null} path="${route}" />`)
    }
  })
})