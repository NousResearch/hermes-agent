import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  proxyFromConfig,
  proxyFromResolveProxyResult,
  proxyRulesFor,
  readConfiguredUpdateProxy,
  resolveProxyFromSources,
  resolveUpdateProxy,
  validProxyUrl
} from './update-transport-proxy'

const keys = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'] as const
const saved = Object.fromEntries(keys.map(key => [key, process.env[key]]))

function clearProxyEnv() {
  for (const key of keys) {
    delete process.env[key]
  }
}

 
;

(globalThis as any).__restoreProxyEnv = () => {
  for (const key of keys) {
    const value = saved[key]

    if (value === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = value
    }
  }
}

test('a GUI launch with no config and no env resolves no proxy', () => {
  clearProxyEnv()
  const { proxy, env } = resolveUpdateProxy(process.env, null)

  assert.equal(proxy, null)
  assert.equal(env, process.env)
})

test('the system proxy is the third rung: config wins over it, env wins over it', () => {
  clearProxyEnv()
  const system = 'http://127.0.0.1:7890'

  // Nothing configured, nothing exported: the system answer is used.
  assert.equal(resolveProxyFromSources({}, null, system), system)

  // Ambient env outranks the system proxy.
  process.env.HTTPS_PROXY = 'http://127.0.0.1:8080'
  assert.equal(resolveProxyFromSources(process.env, null, system), 'http://127.0.0.1:8080')

  // Config outranks both.
  assert.equal(
    resolveProxyFromSources(process.env, { updates: { proxy: 'http://127.0.0.1:7897' } }, system),
    'http://127.0.0.1:7897'
  )

   
  ;(globalThis as any).__restoreProxyEnv()
})

test('the system rung reaches the hand-off child env overlay', () => {
  clearProxyEnv()
  const { proxy, env } = resolveUpdateProxy(process.env, null, process.env, 'http://127.0.0.1:7890')

  assert.equal(proxy, 'http://127.0.0.1:7890')
  assert.equal(env.HTTPS_PROXY, 'http://127.0.0.1:7890')
  assert.equal(env.https_proxy, 'http://127.0.0.1:7890')
})

test('PAC results follow Chromium order: DIRECT first prefers direct, else first usable PROXY/HTTPS entry', () => {
  // Chromium returns the ordered fallback list; the first entry decides.
  assert.equal(proxyFromResolveProxyResult('PROXY 127.0.0.1:7890; DIRECT'), 'http://127.0.0.1:7890')
  // A DIRECT-first list prefers direct: the proxy is only the fallback.
  assert.equal(proxyFromResolveProxyResult('DIRECT; PROXY 127.0.0.1:7890'), null)
  assert.equal(proxyFromResolveProxyResult('DIRECT'), null)
  // SOCKS entries cannot be consumed by any update transport; skip them.
  assert.equal(proxyFromResolveProxyResult('SOCKS5 127.0.0.1:1080; PROXY 127.0.0.1:7890'), 'http://127.0.0.1:7890')
  assert.equal(proxyFromResolveProxyResult('SOCKS5 127.0.0.1:1080'), null)
  // HTTPS scheme is preserved; unparseable entries are skipped in order.
  assert.equal(proxyFromResolveProxyResult('HTTPS proxy.corp:8443'), 'https://proxy.corp:8443')
  assert.equal(proxyFromResolveProxyResult('PROXY ; PROXY 127.0.0.1:7890'), 'http://127.0.0.1:7890')
  // Empty or garbage is no proxy, never a throw.
  assert.equal(proxyFromResolveProxyResult(''), null)
  assert.equal(proxyFromResolveProxyResult('   '), null)
  assert.equal(proxyFromResolveProxyResult('garbage'), null)
  assert.equal(proxyFromResolveProxyResult('PROXY '), null)
})

test('updates.proxy from config is used even with no env (the GUI-launched case, #60049)', () => {
  clearProxyEnv()
  const config = { updates: { proxy: 'http://127.0.0.1:7897' } }
  const { proxy, env } = resolveUpdateProxy(process.env, config)

  assert.equal(proxy, 'http://127.0.0.1:7897')
  assert.equal(env.https_proxy, 'http://127.0.0.1:7897')
  assert.equal(env.HTTP_PROXY, 'http://127.0.0.1:7897')
  // The ambient env is not mutated; the overlay is a copy for children.
  assert.equal(process.env.https_proxy, undefined)
})

test('an exported proxy env still wins over config for the same key', () => {
  clearProxyEnv()
  process.env.HTTPS_PROXY = 'http://127.0.0.1:9999'
  const config = { updates: { proxy: 'http://127.0.0.1:7897' } }
  const { proxy, env } = resolveUpdateProxy(process.env, config)

  // Config is the explicit opt-in and resolves first, but the overlay never
  // clobbers an exported value: the proxied shell's choice survives.
  assert.equal(proxy, 'http://127.0.0.1:7897')
  assert.equal(env.HTTPS_PROXY, 'http://127.0.0.1:9999')
  assert.equal(env.https_proxy, 'http://127.0.0.1:7897')
   
  ;(globalThis as any).__restoreProxyEnv()
})

test('ambient proxy env alone is honored (shell-launched app)', () => {
  clearProxyEnv()
  process.env.HTTPS_PROXY = 'http://127.0.0.1:8080'
  assert.equal(resolveProxyFromSources(process.env, null), 'http://127.0.0.1:8080')
  process.env.https_proxy = 'http://127.0.0.1:8081'
  assert.equal(resolveProxyFromSources(process.env, null), 'http://127.0.0.1:8081')
   
  ;(globalThis as any).__restoreProxyEnv()
})

test('malformed proxy values are rejected, never thrown', () => {
  clearProxyEnv()
  assert.equal(validProxyUrl('not a url'), null)
  assert.equal(validProxyUrl('ftp://127.0.0.1:7897'), null)
  assert.equal(validProxyUrl('http://'), null)
  assert.equal(validProxyUrl(''), null)
  assert.equal(validProxyUrl(null), null)
  assert.equal(validProxyUrl('http://127.0.0.1:7897'), 'http://127.0.0.1:7897')
  assert.equal(validProxyUrl('socks5://127.0.0.1:7897'), 'socks5://127.0.0.1:7897')

  // A malformed config value means "no configured proxy", not a crash.
  assert.equal(proxyFromConfig({ updates: { proxy: 'garbage' } }), null)
  assert.equal(proxyFromConfig({}), null)
  assert.equal(proxyFromConfig('updates: yes'), null)
  assert.equal(resolveProxyFromSources({}, { updates: { proxy: 'garbage' } }), null)
})

test('session proxy rules map the proxy scheme for electron-updater', () => {
  assert.equal(proxyRulesFor('http://127.0.0.1:7897'), 'http=http://127.0.0.1:7897;https=http://127.0.0.1:7897')
  assert.equal(proxyRulesFor('https://proxy.corp:8443'), 'http=https://proxy.corp:8443;https=https://proxy.corp:8443')
  assert.equal(proxyRulesFor('socks5://127.0.0.1:1080'), 'http=socks5://127.0.0.1:1080;https=socks5://127.0.0.1:1080')
  assert.equal(proxyRulesFor('garbage'), null)
})

test('readConfiguredUpdateProxy finds updates.proxy inside the updates block only', async () => {
  const read = async (text: string) => text

  assert.equal(
    await readConfiguredUpdateProxy('x', async () => 'updates:\n  proxy: "http://127.0.0.1:7897"\n'),
    'http://127.0.0.1:7897'
  )
  assert.equal(
    await readConfiguredUpdateProxy('x', async () => 'updates:\n  proxy: http://127.0.0.1:7897 # clash\n'),
    'http://127.0.0.1:7897'
  )
  // A proxy key in another section is never picked up.
  assert.equal(
    await readConfiguredUpdateProxy('x', async () => 'gateway:\n  proxy: "http://elsewhere:1"\nupdates:\n  check: true\n'),
    null
  )
  // No updates block at all.
  assert.equal(await readConfiguredUpdateProxy('x', async () => 'gateway:\n  run: true\n'), null)
  // Unreadable config means no configured proxy, never a throw.
  assert.equal(
    await readConfiguredUpdateProxy('x', async () => {
      throw new Error('ENOENT')
    }),
    null
  )
})
