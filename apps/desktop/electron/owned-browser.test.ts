import assert from 'node:assert/strict'
import test from 'node:test'
import http from 'node:http'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { OwnedBrowsers, installedBrowser, webUrl } from './owned-browser'

test('browser URL boundary rejects local and privileged targets', () => {
  for (const url of ['file:///C:/Users/Kayla/secret', 'about:config', 'javascript:alert(1)', 'edge://settings', 'https://user:secret@example.com']) assert.throws(() => webUrl(url))
  assert.equal(webUrl('https://example.com'), 'https://example.com/')
})

for (const product of ['edge', 'firefox'] as const) {
  test(`${product} real protocol: navigation, refs, Unicode typing, isolation and revoke`, { timeout: 120000 }, async () => {
    const server = http.createServer((req, res) => {
      res.setHeader('Content-Type', 'text/html; charset=utf-8')
      res.end(req.url === '/done' ? '<title>Verified destination</title><p>Arrived</p>' : '<title>Bridge fixture</title><label>Text<input aria-label="Text"></label><a href="/done">Learn more</a><a href="file:///C:/secret">Local file</a><input type="file" aria-label="Upload">')
    })
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    const address = server.address() as { port: number }
    const a = new OwnedBrowsers(), b = new OwnedBrowsers()
    try {
      const executable = await installedBrowser(product)
      const prepared = await a.prepare(product, executable, true)
      const { target_id, tab_id, pid } = prepared.structuredContent as any
      assert.equal((prepared.structuredContent as any).private_storage_context, true)
      assert.equal((prepared.structuredContent as any).guest_mode, product === 'edge')
      const ids = { target_id, tab_id }
      assert.equal(typeof pid, 'number')
      await assert.rejects(b.call('get_browser_state', ids), /another conversation/)
      await a.call('browser_navigate', { ...ids, url: `http://127.0.0.1:${address.port}/` })
      const first = (await a.call('get_browser_state', ids)).structuredContent as any
      assert.equal(first.title, 'Bridge fixture')
      const input = first.controls.find((el: any) => el.text === 'Text')
      const stale = first.controls.find((el: any) => el.text === 'Learn more').ref
      await a.call('browser_type', { ...ids, ref: input.ref, text: 'Athena α — quotes \' and $dollar', replace: true })
      await assert.rejects(a.call('browser_click', { ...ids, ref: stale }), /Stale/)
      const captured = await a.call('get_browser_state', { ...ids, include_screenshot: true })
      assert.equal(captured.content.some(block => block.type === 'image'), true)
      let fresh = captured.structuredContent as any
      assert.equal(fresh.controls.find((el: any) => el.text === 'Text').value, 'Athena α — quotes \' and $dollar')
      await assert.rejects(a.call('browser_click', { ...ids, ref: fresh.controls.find((el: any) => el.text === 'Local file').ref }), /local files/)
      await assert.rejects(a.call('browser_click', { ...ids, ref: fresh.controls.find((el: any) => el.text === 'Upload').ref }), /Uploads/)
      await a.call('browser_pointer', { ...ids, action: 'hover', ref: fresh.controls.find((el: any) => el.text === 'Text').ref })
      fresh = (await a.call('get_browser_state', ids)).structuredContent as any
      if (product === 'edge') {
        // Simulate the rendering callback never being delivered in a hidden
        // window. Protocol input must not wait on IntersectionObserver.
        const run = (a as any).runs.get(target_id)
        const page = run.tabs.get(tab_id).page
        await page.evaluate(() => {
          window.IntersectionObserver = class {
            observe() {} unobserve() {} disconnect() {} takeRecords() { return [] }
          } as any
        })
      }
      await a.call('browser_click', { ...ids, ref: fresh.controls.find((el: any) => el.text === 'Learn more').ref })
      // Wait for the observed destination without replaying input.
      let observed: any
      for (let attempt = 0; attempt < 20; attempt++) {
        observed = (await a.call('get_browser_state', ids)).structuredContent
        if (observed.title === 'Verified destination') break
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      assert.equal(observed.title, 'Verified destination')
      assert.equal(observed.url, `http://127.0.0.1:${address.port}/done`)
      await a.close()
      await assert.rejects(a.call('get_browser_state', ids), /unavailable/)
      await assert.rejects(a.prepare(product, executable, true), /revoked/)
    } finally { await a.close(); await b.close(); await new Promise<void>(resolve => server.close(() => resolve())) }
  })
}

for (const product of ['edge', 'firefox'] as const) {
test(`${product} persistent profile uses its launch tab, clicks and retains fixture cookie across restarts`, { timeout: 120000 }, async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-profile-fixture-'))
  const server = http.createServer((req, res) => {
    if (req.url === '/set') res.setHeader('Set-Cookie', 'hermes_fixture=roundtrip; Max-Age=3600; Path=/')
    res.end('<title>Persistence fixture</title><p>' + (req.headers.cookie?.includes('hermes_fixture=roundtrip') ? 'COOKIE_PRESENT' : 'COOKIE_ABSENT') + '</p><a href="/check">Check cookie</a>')
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const url = `http://127.0.0.1:${(server.address() as { port: number }).port}`
  const executable = await installedBrowser(product)
  const first = new OwnedBrowsers(), second = new OwnedBrowsers()
  try {
    const p = (await first.prepare(product, executable, true, directory)).structuredContent as any
    assert.equal(p.persistent_profile, true)
    assert.equal(p.private_storage_context, false)
    assert.equal(first.accountScope(p.target_id), `athena_profile:${product}`)
    if (product === 'edge') {
      const run = (first as any).runs.get(p.target_id)
      const launchPages = (await run.browser.pages()).filter((page: any) => page.url() === 'about:blank')
      assert.equal(launchPages.length, 1)
      assert.equal(run.tabs.get(p.tab_id).page, launchPages[0])
    }
    await first.call('browser_navigate', { target_id: p.target_id, tab_id: p.tab_id, url: url + '/set' })
    const before = (await first.call('get_browser_state', { target_id: p.target_id, tab_id: p.tab_id })).structuredContent as any
    await first.call('browser_click', { target_id: p.target_id, tab_id: p.tab_id, ref: before.controls.find((item: any) => item.text === 'Check cookie').ref })
    const after = (await first.call('get_browser_state', { target_id: p.target_id, tab_id: p.tab_id })).structuredContent as any
    assert.equal(after.url, url + '/check')
    await first.close()
    const q = (await second.prepare(product, executable, true, directory)).structuredContent as any
    await second.call('browser_navigate', { target_id: q.target_id, tab_id: q.tab_id, url: url + '/check' })
    const state = (await second.call('get_browser_state', { target_id: q.target_id, tab_id: q.tab_id })).structuredContent as any
    assert.match(state.visible_text, /COOKIE_PRESENT/)
  } finally {
    await first.close(); await second.close()
    await new Promise<void>(resolve => server.close(() => resolve()))
    const resolved = await fs.realpath(directory)
    if (path.dirname(resolved) !== await fs.realpath(os.tmpdir()) || !path.basename(resolved).startsWith('hermes-profile-fixture-')) throw new Error('Fixture cleanup path escaped its temporary root.')
    await fs.rm(resolved, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 })
  }
})
}
