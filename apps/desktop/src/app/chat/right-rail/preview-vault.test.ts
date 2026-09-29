import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'

import {
  invalidatePreviewScriptRunner,
  markPreviewDocumentReady,
  registerPreviewScriptRunner
} from './preview-script-runner'
import {
  closePreviewVaultBinding,
  evaluatePreviewVaultBinding,
  openPreviewVaultBinding,
  type PreviewVaultScope,
  resetPreviewVaultBindingsForTests
} from './preview-vault'

const scope: PreviewVaultScope = { connectionId: 'connection-a', profile: 'default', sessionId: 'session-a' }

function target(url: string): PreviewTarget {
  return { kind: 'url', label: 'Browser', source: url, url }
}

function fileTarget(path: string): PreviewTarget {
  return { kind: 'file', label: path, path, previewKind: 'text', source: path, url: `file://${path}` }
}

describe('preview vault target binding', () => {
  let cleanups: Array<() => void> = []

  beforeEach(() => {
    cleanups.forEach(cleanup => cleanup())
    cleanups = []
    resetPreviewVaultBindingsForTests()
    closeRightRail()
  })

  it('pins the selected preview runner and evaluates only through that runner', async () => {
    openPreview(target('https://one.example'))
    const first = $rightRailActiveTabId.get()!
    const run = vi.fn(async () => 'completion')
    cleanups.push(registerPreviewScriptRunner(first, run))
    const binding = openPreviewVaultBinding(scope)

    expect(binding).toMatchObject({ success: true })

    if (!binding.success) {
      return
    }

    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        target: binding.target,
        expression: 'return 42',
        isSessionActive: () => true
      })
    ).resolves.toEqual({ success: true, result: 'completion' })
    expect(run).toHaveBeenCalledWith('return 42')
  })

  it('refuses evaluation after the selected preview changes', async () => {
    openPreview(fileTarget('/one.html'))
    const first = $rightRailActiveTabId.get()!
    const run = vi.fn(async () => 'completion')
    cleanups.push(registerPreviewScriptRunner(first, run))
    const binding = openPreviewVaultBinding(scope)
    openPreview(fileTarget('/two.html'))

    if (!binding.success) {
      throw new Error('expected a live preview binding')
    }

    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        target: binding.target,
        expression: 'return 42',
        isSessionActive: () => true
      })
    ).resolves.toMatchObject({ success: false })
    expect(run).not.toHaveBeenCalled()
  })

  it.each(['tab removal', 'runner replacement'])('invalidates a pinned page after %s', async change => {
    openPreview(fileTarget('/pinned.html'))
    const tabId = $rightRailActiveTabId.get()!
    const original = vi.fn(async () => 'original')
    cleanups.push(registerPreviewScriptRunner(tabId, original))
    const binding = openPreviewVaultBinding(scope)

    if (!binding.success) {
      throw new Error('expected a live preview binding')
    }

    if (change === 'tab removal') {
      closeRightRail()
    } else {
      cleanups.push(
        registerPreviewScriptRunner(
          tabId,
          vi.fn(async () => 'replacement')
        )
      )
    }

    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        target: binding.target,
        expression: 'return 42',
        isSessionActive: () => true
      })
    ).resolves.toMatchObject({ success: false })
    expect(original).not.toHaveBeenCalled()
  })

  it('invalidates same-tab bindings on document navigation and permits a fresh binding', async () => {
    openPreview(target('https://one.example'))
    const tabId = $rightRailActiveTabId.get()!
    const run = vi.fn(async () => 'completion')
    cleanups.push(registerPreviewScriptRunner(tabId, run))
    const previous = openPreviewVaultBinding(scope)

    if (!previous.success) {
      throw new Error('expected a live preview binding')
    }

    invalidatePreviewScriptRunner(tabId)
    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        target: previous.target,
        expression: 'return 42',
        isSessionActive: () => true
      })
    ).resolves.toMatchObject({ success: false })
    expect(run).not.toHaveBeenCalled()

    expect(openPreviewVaultBinding(scope)).toEqual({ error: 'No live preview page is available.', success: false })

    markPreviewDocumentReady(tabId)

    const current = openPreviewVaultBinding(scope)

    if (!current.success) {
      throw new Error('expected a fresh binding after navigation')
    }

    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        target: current.target,
        expression: 'return 42',
        isSessionActive: () => true
      })
    ).resolves.toEqual({ result: 'completion', success: true })
    expect(run).toHaveBeenCalledOnce()
  })

  it('declines unknown or differently scoped bindings without evaluating', async () => {
    openPreview(target('https://one.example'))
    const run = vi.fn(async () => 'completion')
    cleanups.push(registerPreviewScriptRunner($rightRailActiveTabId.get()!, run))
    const binding = openPreviewVaultBinding(scope)

    if (!binding.success) {
      throw new Error('expected a live preview binding')
    }

    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        connectionId: 'connection-b',
        target: binding.target,
        expression: 'return 42',
        isSessionActive: () => true
      })
    ).resolves.toEqual({ decline: true })
    await expect(
      evaluatePreviewVaultBinding({ ...scope, target: 'unknown', expression: 'return 42', isSessionActive: () => true })
    ).resolves.toEqual({ decline: true })
    expect(run).not.toHaveBeenCalled()
  })

  it('checks session activity both before and after executing the page script', async () => {
    openPreview(target('https://one.example'))
    const tabId = $rightRailActiveTabId.get()!
    let active = true

    const run = vi.fn(async () => {
      active = false

      return 'completion'
    })

    cleanups.push(registerPreviewScriptRunner(tabId, run))
    const binding = openPreviewVaultBinding(scope)

    if (!binding.success) {
      throw new Error('expected a live preview binding')
    }

    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        target: binding.target,
        expression: 'return 42',
        isSessionActive: () => active
      })
    ).resolves.toMatchObject({ success: false })
    expect(run).toHaveBeenCalledOnce()
  })

  it('does not evaluate a request already cancelled after the bridge loaded', async () => {
    openPreview(target('https://one.example'))
    const runner = vi.fn(async () => 'completion')
    cleanups.push(registerPreviewScriptRunner($rightRailActiveTabId.get()!, runner))
    const binding = openPreviewVaultBinding(scope)

    if (!binding.success) {
      throw new Error('expected a live preview binding')
    }

    const controller = new AbortController()
    controller.abort('interrupted')

    await expect(
      evaluatePreviewVaultBinding({
        ...scope,
        target: binding.target,
        expression: 'return 42',
        isSessionActive: () => true,
        signal: controller.signal
      })
    ).resolves.toMatchObject({ success: false })
    expect(runner).not.toHaveBeenCalled()
  })

  it('releases a binding on close and refuses to open when the selected runner is absent', () => {
    openPreview(target('https://one.example'))
    expect(openPreviewVaultBinding(scope)).toMatchObject({ success: false })
    cleanups.push(
      registerPreviewScriptRunner(
        $rightRailActiveTabId.get()!,
        vi.fn(async () => null)
      )
    )
    const binding = openPreviewVaultBinding(scope)

    if (!binding.success) {
      throw new Error('expected a live preview binding')
    }

    expect(closePreviewVaultBinding({ ...scope, target: binding.target })).toEqual({ success: true })
    expect(closePreviewVaultBinding({ ...scope, target: binding.target })).toEqual({ decline: true })
  })
})
