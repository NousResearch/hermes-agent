import { EventEmitter } from 'node:events'

import type { App, Session, WebPreferences } from 'electron'
import { expect, it, vi } from 'vitest'

import * as policy from './plugin-viewer-policy'

function harness() {
  const app = new EventEmitter()

  const guestSession = {
    setPermissionRequestHandler: vi.fn(),
    setPermissionCheckHandler: vi.fn()
  }

  const fromPartition = vi.fn(() => guestSession as unknown as Session)
  expect(policy.installViewerGuestPolicy).toBeTypeOf('function')
  policy.installViewerGuestPolicy(app as App, { fromPartition })
  const contents = new EventEmitter()
  app.emit('web-contents-created', {}, contents)

  return {
    guestSession,
    fromPartition,
    attach(partition: string, preferences: WebPreferences = {}) {
      const event = { preventDefault: vi.fn() }
      contents.emit('will-attach-webview', event, preferences, { partition, src: 'https://example.org/viewer' })

      return { event, preferences }
    }
  }
}

it('enforces unprivileged guest preferences and rejects preload or partition escalation at attachment', () => {
  const { attach, fromPartition } = harness()
  const partition = 'hermes-viewer-8e195494-8de3-4e84-8241-92d0301ef34c'

  for (const selected of [partition, 'persist:hermes-preview', 'persist:hermes-embed']) {
    const { event, preferences } = attach(selected, {
      nodeIntegration: true,
      nodeIntegrationInWorker: true,
      nodeIntegrationInSubFrames: true,
      contextIsolation: false,
      sandbox: false,
      webviewTag: true,
      webSecurity: false,
      allowRunningInsecureContent: true
    })

    expect(event.preventDefault).not.toHaveBeenCalled()
    expect(preferences).toMatchObject({
      partition: selected,
      nodeIntegration: false,
      nodeIntegrationInWorker: false,
      nodeIntegrationInSubFrames: false,
      contextIsolation: true,
      sandbox: true,
      webviewTag: false,
      webSecurity: true,
      allowRunningInsecureContent: false
    })
  }

  fromPartition.mockClear()

  for (const selected of [
    '',
    'persist:hermes-viewer-test',
    'hermes-viewer-',
    'hermes-viewer-invalid',
    'persist:hermes-remote-oauth',
    'hermes-plugin-viewer:1:demo:watch',
    'arbitrary'
  ]) {
    expect(attach(selected).event.preventDefault).toHaveBeenCalledOnce()
  }

  for (const selected of [partition, 'persist:hermes-preview']) {
    for (const preferences of [
      { preload: '/opt/hermes/preload.cjs' },
      { preloadURL: 'file:///opt/hermes/preload.cjs' },
      { partition: 'persist:hermes-remote-oauth' },
      { session: {} as Session }
    ]) {
      expect(attach(selected, preferences as WebPreferences).event.preventDefault).toHaveBeenCalledOnce()
    }
  }

  expect(fromPartition).not.toHaveBeenCalled()
})

it('denies all isolated guest permission paths synchronously before attachment without changing Browser sessions', () => {
  const { attach, fromPartition, guestSession } = harness()
  const partition = 'hermes-viewer-8e195494-8de3-4e84-8241-92d0301ef34c'
  const { event } = attach(partition)
  expect(event.preventDefault).not.toHaveBeenCalled()
  expect(fromPartition).toHaveBeenCalledWith(partition)
  const check = guestSession.setPermissionCheckHandler.mock.calls[0][0]
  const request = guestSession.setPermissionRequestHandler.mock.calls[0][0]

  for (const permission of ['media', 'geolocation', 'notifications', 'clipboard-read', 'unknown-future-permission']) {
    const callback = vi.fn()
    expect(check(null, permission, 'https://example.org', {})).toBe(false)
    request(null, permission, callback, {})
    expect(callback).toHaveBeenCalledExactlyOnceWith(false)
  }

  fromPartition.mockClear()
  guestSession.setPermissionCheckHandler.mockClear()
  guestSession.setPermissionRequestHandler.mockClear()

  for (const partition of ['persist:hermes-preview', 'persist:hermes-embed']) {
    expect(attach(partition).event.preventDefault).not.toHaveBeenCalled()
  }

  expect(fromPartition).not.toHaveBeenCalled()
  expect(guestSession.setPermissionCheckHandler).not.toHaveBeenCalled()
  expect(guestSession.setPermissionRequestHandler).not.toHaveBeenCalled()
})
