import { readFileSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { appViewForPath, CONTEXT_ROUTE, isOverlayView, OVERLAY_VIEWS } from './routes'

describe('context inspector route', () => {
  it('maps /context to the context overlay view', () => {
    expect(appViewForPath(CONTEXT_ROUTE)).toBe('context')
  })

  it('registers context as an overlay view', () => {
    expect(OVERLAY_VIEWS.has('context')).toBe(true)
    expect(isOverlayView('context')).toBe(true)
  })

  it('declares a <Route path="context"> in DesktopController so navigating to /context does not hit the wildcard redirect', () => {
    // Guards the exact blocker found in review: the context overlay was
    // registered in route metadata (OVERLAY_VIEWS) but had no matching
    // <Route> in the controller, so navigating to /context fell through to
    // `path="*"` and redirected to `/`, silently closing the overlay.
    // Isolated component tests don't mount the real router, so this
    // source-level assertion is the backstop.
    const controllerSrc = readFileSync(
      path.resolve(process.cwd(), 'src/app/desktop-controller.tsx'),
      'utf8'
    )

    const contextPath = CONTEXT_ROUTE.replace(/^\//, '')
    expect(controllerSrc).toMatch(new RegExp(`path=["']${contextPath}["']`))
    // And it must sit before the catch-all wildcard route.
    const contextIdx = controllerSrc.indexOf(`path="${contextPath}"`)
    const wildcardIdx = controllerSrc.indexOf('path="*"')
    expect(contextIdx).toBeGreaterThan(-1)
    expect(wildcardIdx).toBeGreaterThan(-1)
    expect(contextIdx).toBeLessThan(wildcardIdx)
  })
})
