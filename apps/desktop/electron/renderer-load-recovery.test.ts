import { describe, expect, it } from 'vitest'

import { decideFailedLoadRecovery, handleFailedWindowLoad, loadFailureErrorCode } from './renderer-load-recovery'

/**
 * A window whose INITIAL renderer load fails used to sit blank forever with the
 * only explanation in logs/desktop.log (main.ts loadWindowUrl caught the
 * rejection and logged it). The primary window has the renderer-lifecycle
 * reload policy; every other window type — secondary session windows, browser
 * popouts, instance windows — had nothing, so their failure surfaced as an
 * empty window with no way forward.
 *
 * Contract under test:
 *   - a plain failed load ends in the VISIBLE recovery page, with the failed
 *     URL as the Reload target;
 *   - ERR_ABORTED (-3) is a superseded navigation, never a failure to surface;
 *   - a destroyed window is never loaded into;
 *   - a window whose lifecycle policy owns recovery (the primary window) is
 *     left alone — surfacing a page there would race its bounded reload and
 *     could cover a UI that healed;
 *   - the caller is never left holding a rejection.
 */

interface RecordedPage {
  errorCode?: number | string | undefined
  errorDescription?: string
  url?: string
  reloadUrl?: string
}

function harness(options: { pageRejects?: boolean; destroyed?: boolean } = {}) {
  const logs: string[] = []
  const pages: RecordedPage[] = []

  const win = {
    loadURL: () => Promise.resolve(),
    isDestroyed: () => Boolean(options.destroyed)
  }

  const deps = {
    log: (message: string) => void logs.push(message),
    showErrorPage: (_win: unknown, details: RecordedPage) => {
      pages.push(details)

      return options.pageRejects ? Promise.reject(new Error('the recovery page itself failed')) : Promise.resolve()
    }
  }

  return { logs, pages, win, deps }
}

describe('decideFailedLoadRecovery', () => {
  it('surfaces a failed load in a window that has no recovery owner', () => {
    const decision = decideFailedLoadRecovery({
      label: 'Session window',
      url: 'file:///app/dist/index.html',
      errorCode: -2
    })

    expect(decision).toEqual({ showErrorPage: true, reason: 'no-recovery-owner' })
  })

  it('never surfaces ERR_ABORTED: a superseded navigation is not a failure', () => {
    for (const errorCode of [-3, '-3']) {
      expect(decideFailedLoadRecovery({ label: 'Renderer', url: 'file:///app/dist/index.html', errorCode })).toEqual({
        showErrorPage: false,
        reason: 'superseded-navigation'
      })
    }
  })

  it('never loads a page into a destroyed window', () => {
    expect(
      decideFailedLoadRecovery({
        label: 'Browser window',
        url: 'file:///app/dist/index.html',
        errorCode: -2,
        isDestroyed: true
      })
    ).toEqual({ showErrorPage: false, reason: 'window-destroyed' })
  })

  it('leaves a window whose lifecycle policy owns recovery alone', () => {
    expect(
      decideFailedLoadRecovery({
        label: 'Renderer',
        url: 'file:///app/dist/index.html',
        errorCode: -2,
        recoveryOwnedByLifecycle: true
      })
    ).toEqual({ showErrorPage: false, reason: 'lifecycle-owns-recovery' })
  })

  it('prefers teardown and supersession over the missing-owner case', () => {
    expect(
      decideFailedLoadRecovery({
        label: 'Renderer',
        url: 'file:///app/dist/index.html',
        errorCode: -3,
        isDestroyed: true,
        recoveryOwnedByLifecycle: true
      })
    ).toEqual({ showErrorPage: false, reason: 'window-destroyed' })
  })
})

describe('handleFailedWindowLoad', () => {
  it('loads the visible recovery page with the failed URL as the Reload target', async () => {
    const { logs, pages, win, deps } = harness()

    const decision = await handleFailedWindowLoad(
      win,
      { label: 'Session window', url: 'file:///app/dist/index.html?session=s1', errorCode: -2, errorDescription: 'ERR_FAILED' },
      deps
    )

    expect(decision).toEqual({ showErrorPage: true, reason: 'no-recovery-owner' })
    expect(pages).toHaveLength(1)
    expect(pages[0]).toMatchObject({
      errorCode: -2,
      url: 'file:///app/dist/index.html?session=s1',
      reloadUrl: 'file:///app/dist/index.html?session=s1'
    })
    expect(logs.some(line => line.includes('Session window') && line.includes('recovery page'))).toBe(true)
  })

  it('does nothing for a window the lifecycle already recovers', async () => {
    const { logs, pages, win, deps } = harness()

    const decision = await handleFailedWindowLoad(
      win,
      { label: 'Renderer', url: 'file:///app/dist/index.html', errorCode: -2, recoveryOwnedByLifecycle: true },
      deps
    )

    expect(decision.showErrorPage).toBe(false)
    expect(pages).toEqual([])
    expect(logs.some(line => line.includes('recovery page'))).toBe(false)
  })

  it('does nothing for a superseded navigation or a destroyed window', async () => {
    const aborted = harness()
    expect(
      (await handleFailedWindowLoad(aborted.win, { label: 'Renderer', url: 'u', errorCode: -3 }, aborted.deps)).showErrorPage
    ).toBe(false)
    expect(aborted.pages).toEqual([])

    const dead = harness({ destroyed: true })
    expect(
      (await handleFailedWindowLoad(dead.win, { label: 'Renderer', url: 'u', errorCode: -2 }, dead.deps)).showErrorPage
    ).toBe(false)
    expect(dead.pages).toEqual([])
  })

  it('never rejects the caller when the recovery page itself fails to load', async () => {
    const { win, deps } = harness({ pageRejects: true })

    await expect(
      handleFailedWindowLoad(win, { label: 'Instance window', url: 'u', errorCode: -2 }, deps)
    ).resolves.toEqual({ showErrorPage: true, reason: 'no-recovery-owner' })
  })
})

/**
 * Electron's typings declare loadURL's rejection as a bare Promise<void>: the
 * error shape is undocumented, and the only sure evidence is what the app has
 * actually logged —
 *   Renderer failed to load: Error: ERR_ABORTED (-3) loading 'file:///…'
 *   Renderer failed to load: Error: ERR_FAILED (-2) loading 'file:///…'
 * (logs/desktop.log). Reading the code matters because an ABORTED navigation
 * must never be reported to the user as a load failure.
 */
describe('loadFailureErrorCode', () => {
  it('names the code in the message Electron actually rejects with', () => {
    for (const [message, expected] of [
      ["ERR_ABORTED (-3) loading 'file:///app/dist/index.html'", 'ERR_ABORTED'],
      ["ERR_FAILED (-2) loading 'file:///app/dist/index.html'", 'ERR_FAILED'],
      ["Error: ERR_ABORTED (-3) loading 'file:///app/dist/index.html'", 'ERR_ABORTED']
    ] as const) {
      expect(loadFailureErrorCode(new Error(message))).toBe(expected)
    }
  })

  it('prefers a structured code/errno when the rejection carries one', () => {
    expect(loadFailureErrorCode(Object.assign(new Error('boom'), { code: 'ERR_FILE_NOT_FOUND' }))).toBe(
      'ERR_FILE_NOT_FOUND'
    )
    expect(loadFailureErrorCode({ errno: -6 })).toBe(-6)
  })

  it('returns nothing it cannot name', () => {
    expect(loadFailureErrorCode(new Error('boom'))).toBeUndefined()
    expect(loadFailureErrorCode('a thrown string')).toBeUndefined()
    expect(loadFailureErrorCode(undefined)).toBeUndefined()
  })

  it('keeps an aborted navigation away from the error page end to end', () => {
    const error = new Error("ERR_ABORTED (-3) loading 'file:///app/dist/index.html'")

    expect(
      decideFailedLoadRecovery({
        label: 'Session window',
        url: 'file:///app/dist/index.html',
        errorCode: loadFailureErrorCode(error)
      })
    ).toEqual({ showErrorPage: false, reason: 'superseded-navigation' })
  })
})
