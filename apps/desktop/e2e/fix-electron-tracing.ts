/**
 * Monkey-patch: playwright's test runner never calls tracing.start() on
 * Electron's internal BrowserContext because:
 *  1. Playwright._allContexts() only returns [chromium, firefox, webkit]
 *     contexts — Electron's context is excluded.
 *  2. ArtifactsRecorder.didCreateBrowserContext runs in willStartTest, before
 *     beforeAll launches the electron app.
 *  3. The runAfterCreateBrowserContext hook doesn't exist on the Electron
 *     class (only on BrowserType).
 *
 * As a result, trace screenshots (screencast) and DOM snapshots are never
 * captured for electron tests.
 *
 * This patch:
 *  1. Patches _allContexts() to include electron contexts, so the test
 *     runner's didFinishTest() cleanup calls _stopTracing() → stopChunk()
 *     on the electron context (saving the trace chunk + merging it into
 *     the final trace.zip).
 *  2. Manually calls tracing.start() + startChunk() after launch.
 *  3. Wraps tracing.start to become startChunk after the first call,
 *     so the test runner's willStartTest doesn't throw "already started".
 *
 * Imported from playwright.config.ts so it runs before any test.
 *
 * Pinned dependency: this file reaches into Playwright internals (_playwright,
 * _allContexts, _context) that have no public contract. @playwright/test is
 * pinned exact (=1.58.2 in package.json) so a bump can't silently break the
 * monkeypatch. When bumping, re-verify these private symbols still exist on
 * the Electron / PlaywrightInternal classes and that tracing still merges.
 */

import { _electron as electron, type BrowserContext } from '@playwright/test'
import * as crypto from 'node:crypto'

const electronContexts = new Set<BrowserContext>()
const originalLaunch = electron.launch.bind(electron)

electron.launch = async (options: any) => {
  const app = await originalLaunch(options)
  const ctx = (app as any)._context as BrowserContext
  electronContexts.add(ctx)
  ctx.once('close', () => electronContexts.delete(ctx))

  // Patch _allContexts so the test runner sees the electron context
  // (didFinishTest cleanup → _stopTracing → stopChunk → merge into trace.zip).
  const pw = (electron as any)._playwright as any
  if (pw && !pw.__electronTracingPatched) {
    pw.__electronTracingPatched = true
    const original = pw._allContexts.bind(pw)
    pw._allContexts = () => [...original(), ...electronContexts]
  }

  // Start tracing — mirrors ArtifactsRecorder.didCreateBrowserContext.
  const traceName = crypto.randomUUID()
  await ctx.tracing.start({
    screenshots: true,
    snapshots: true,
    sources: true,
  }).catch(() => {})
  await ctx.tracing.startChunk({ title: 'electron', name: traceName }).catch(() => {})

  // Wrap tracing.start to redirect to startChunk after the first call.
  // The test runner's willStartTest calls tracing.start() on all contexts
  // in _allContexts(). Since we already started, redirect to startChunk
  // to avoid "Tracing has been already started" errors.
  const tracing = ctx.tracing as any
  tracing.start = async (opts: any) => {
    return tracing.startChunk(opts)
  }

  return app
}
