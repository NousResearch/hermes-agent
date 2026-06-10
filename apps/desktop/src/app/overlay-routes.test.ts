import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { OVERLAY_VIEWS, SETTINGS_ROUTE, appViewForPath } from './routes'

/**
 * Regression for #43825: overlay routes must keep ChatView mounted so the
 * assistant-ui composer draft survives opening Settings (and other overlays).
 */
describe('overlay routes preserve chat mount', () => {
  it('settings is an overlay view, not chat', () => {
    expect(OVERLAY_VIEWS.has('settings')).toBe(true)
    expect(appViewForPath(SETTINGS_ROUTE)).toBe('settings')
  })

  it('desktop controller mounts chatView on overlay paths', () => {
    const controllerPath = join(dirname(fileURLToPath(import.meta.url)), 'desktop-controller.tsx')
    const source = readFileSync(controllerPath, 'utf8')

    for (const route of ['settings', 'command-center', 'agents', 'cron', 'profiles']) {
      expect(source).toContain(`<Route element={chatView} path="${route}" />`)
      expect(source).not.toContain(`<Route element={null} path="${route}" />`)
    }
  })
})