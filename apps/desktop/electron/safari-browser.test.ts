import assert from 'node:assert/strict'
import { test } from 'node:test'
import http from 'node:http'
import { SafariBrowser } from './safari-browser'
import { actionConsentScope, consentCovers } from './action-consent'
import { validateBrowserPrepare } from './browser-policy'

test('Safari actions require foreground consent and unsupported profiles fail closed', () => {
  assert.equal(actionConsentScope('browser_prepare', { browser: 'safari' }), 'foreground')
  assert.equal(consentCovers(new Set(['background']), 'browser_click', { target_id: 'saf-fixture' }), false)
  assert.throws(() => validateBrowserPrepare({ browser: 'safari', allow_launch: true, profile: { mode: 'athena_profile' } }), /Safari requires/)
  if (process.platform === 'darwin') validateBrowserPrepare({ browser: 'safari', allow_launch: true, profile: { mode: 'isolated_new' } })
})

test('Safari WebDriver wire contract: exact target, snapshots, blocked file controls and revoke', async () => {
  const requests: { method: string; route: string; body: any }[] = []
  let url = 'https://example.com/'
  const server = http.createServer(async (req, res) => {
    let text = ''
    for await (const chunk of req) text += chunk
    const route = req.url!.replace('/session/fixture-session', '')
    const body = text ? JSON.parse(text) : undefined
    requests.push({ method: req.method!, route, body })
    let value: any = null
    if (route === '/url' && req.method === 'GET') value = url
    if (route === '/url' && req.method === 'POST') url = body.url
    if (route === '/title') value = 'Example Domain'
    if (route === '/execute/sync') value = { title: 'Example Domain', text: 'Fixture', controls: [
      { element: { 'element-6066-11e4-a52e-4f735466cecf': 'link-one' }, tag: 'a', type: '', href: 'https://www.iana.org/help/example-domains', download: false, text: 'Learn more' },
      { element: { 'element-6066-11e4-a52e-4f735466cecf': 'upload-one' }, tag: 'input', type: 'file', href: '', download: false, text: 'Upload' },
    ] }
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ value }))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const browser = new SafariBrowser()
  // Seed an already-created fixture session; never launch a real GUI in unit tests.
  Object.assign(browser, { endpoint: `http://127.0.0.1:${(server.address() as any).port}`, session: 'fixture-session', target: 'saf-fixture', tab: 'tab-fixture', child: { exitCode: null, kill() {} } })
  const ids = { target_id: 'saf-fixture', tab_id: 'tab-fixture' }
  try {
    await assert.rejects(browser.call('get_browser_state', { ...ids, target_id: 'other-chat' }), /another conversation/)
    assert.equal(requests.length, 0)
    const first: any = (await browser.call('get_browser_state', ids)).structuredContent
    const second: any = (await browser.call('get_browser_state', ids)).structuredContent
    await assert.rejects(browser.call('browser_click', { ...ids, ref: first.controls[0].ref }), /Stale/)
    await assert.rejects(browser.call('browser_click', { ...ids, ref: second.controls[1].ref }), /File upload/)
    await browser.call('browser_click', { ...ids, ref: second.controls[0].ref })
    assert.equal(requests.filter(r => r.route === '/element/link-one/click').length, 1)
    await assert.rejects(browser.call('browser_click', { ...ids, ref: second.controls[0].ref }), /Stale/)
    await assert.rejects(browser.call('browser_navigate', { ...ids, url: 'file:///private.txt' }), /local|privileged/i)
    assert.equal(requests.filter(r => r.route === '/url' && r.method === 'POST').length, 0)
    await browser.close()
    assert.equal(requests.filter(r => r.method === 'DELETE').length, 1)
    await assert.rejects(browser.call('get_browser_state', ids), /ended session/)
  } finally {
    await browser.close()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
