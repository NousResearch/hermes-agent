/**
 * Tests for electron/mcp-oauth-callback-ipc.ts — the client-side one-shot
 * loopback listener MCP OAuth uses against remote backends. Uses a REAL
 * ephemeral http listener (it binds 127.0.0.1:0, no fixed ports) with the
 * electron ipcMain mocked, and drives synthetic browser hits with fetch.
 *
 * Run with: vitest run --project electron mcp-oauth-callback-ipc
 */

import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

const handlers = new Map<string, (...args: unknown[]) => unknown>()

vi.mock('electron', () => ({
  ipcMain: {
    handle: (channel: string, fn: (...args: unknown[]) => unknown) => {
      handlers.set(channel, fn)
    }
  }
}))

const { registerMcpOauthCallbackIpc } = await import('./mcp-oauth-callback-ipc')

registerMcpOauthCallbackIpc()

const invoke = (channel: string, ...args: unknown[]) => {
  const fn = handlers.get(channel)

  assert.ok(fn, `handler registered for ${channel}`)

  return fn!({}, ...args)
}

test('listen binds a loopback listener and wait resolves with the redirect params', async () => {
  const { id, redirectUri } = (await invoke('hermes:mcp-oauth:listen')) as { id: string; redirectUri: string }

  assert.match(redirectUri, /^http:\/\/127\.0\.0\.1:\d+\/callback$/)

  const waitPromise = invoke('hermes:mcp-oauth:wait', id, 5000) as Promise<{
    code: null | string
    error: null | string
    state: null | string
  }>

  const res = await fetch(`${redirectUri}?code=abc123&state=st-1`)

  assert.equal(res.status, 200)
  assert.match(await res.text(), /return to Hermes/)

  const result = await waitPromise

  assert.equal(result.code, 'abc123')
  assert.equal(result.state, 'st-1')
  assert.equal(result.error, null)

  // Listener is one-shot: the port must be closed after the callback.
  await assert.rejects(fetch(`${redirectUri}?code=again&state=st-1`))
})

test('non-callback noise (favicon) does not settle the listener', async () => {
  const { id, redirectUri } = (await invoke('hermes:mcp-oauth:listen')) as { id: string; redirectUri: string }
  const origin = redirectUri.replace(/\/callback$/, '')

  const res = await fetch(`${origin}/favicon.ico`)

  assert.equal(res.status, 200)

  const waitPromise = invoke('hermes:mcp-oauth:wait', id, 5000) as Promise<{ code: null | string }>

  await fetch(`${redirectUri}?code=late-code&state=s`)

  const result = await waitPromise

  assert.equal(result.code, 'late-code')
})

test('provider error param is forwarded', async () => {
  const { id, redirectUri } = (await invoke('hermes:mcp-oauth:listen')) as { id: string; redirectUri: string }

  const waitPromise = invoke('hermes:mcp-oauth:wait', id, 5000) as Promise<{ code: null | string; error: null | string }>

  await fetch(`${redirectUri}?error=access_denied&state=s`)

  const result = await waitPromise

  assert.equal(result.code, null)
  assert.equal(result.error, 'access_denied')
})

test('cancel tears the listener down and wait reports listener not found afterwards', async () => {
  const { id, redirectUri } = (await invoke('hermes:mcp-oauth:listen')) as { id: string; redirectUri: string }

  assert.equal(await invoke('hermes:mcp-oauth:cancel', id), true)

  await assert.rejects(fetch(`${redirectUri}?code=x&state=s`))

  const result = (await invoke('hermes:mcp-oauth:wait', id, 100)) as { error: null | string }

  assert.equal(result.error, 'listener not found')
})

test('wait times out when no callback arrives', async () => {
  const { id } = (await invoke('hermes:mcp-oauth:listen')) as { id: string }

  const result = (await invoke('hermes:mcp-oauth:wait', id, 1000)) as { code: null | string; error: null | string }

  assert.equal(result.code, null)
  assert.match(String(result.error), /timeout/)
})
