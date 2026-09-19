import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildRendererLoadErrorPage,
  describeRendererCrashCause,
  loadRendererLoadErrorPage,
  parseRecoveryAction
} from './renderer-load-error-page'

test('error page names the failure and carries a Reload button', () => {
  const html = buildRendererLoadErrorPage({
    errorCode: -6,
    errorDescription: 'The desktop renderer bundle is incomplete after the last update (2 missing file(s)).',
    missingAssets: ['assets/app-C0ffee.js', 'assets/shiki-block-DeadBeef.js'],
    repairHint: 'hermes desktop --force-build'
  })

  assert.match(html, /Hermes couldn.t start the desktop UI/)
  assert.match(html, /incomplete after the last update \(2 missing file\(s\)\)/)
  assert.match(html, /-6/)
  assert.match(html, /assets\/app-C0ffee\.js/)
  assert.match(html, /assets\/shiki-block-DeadBeef\.js/)
  assert.match(html, /hermes desktop --force-build/)
  assert.match(html, /Reload/)
  assert.match(html, /location\.reload\(\)/)
})

test('error page reload button targets the real renderer URL when provided', () => {
  const html = buildRendererLoadErrorPage({
    errorDescription: 'load failed',
    reloadUrl: 'file:///C:/Hermes%20Agent/dist/index.html'
  })

  // A data: page cannot recover with location.reload() (it would re-render
  // the error page) — the button must navigate back to the app URL.
  assert.match(html, /location\.replace\("file:\/\/\/C:\/Hermes%20Agent\/dist\/index\.html"\)/)
  assert.doesNotMatch(html, /location\.reload\(\)/)
})

test('error page escapes HTML in failure details', () => {
  const html = buildRendererLoadErrorPage({
    errorDescription: '<script>alert("pwned")</script>',
    url: 'file:///C:/x/index.html?<b>',
    missingAssets: ['assets/<img src=x onerror=alert(1)>.js']
  })

  assert.doesNotMatch(html, /<script>alert\("pwned"\)<\/script>/)
  assert.match(html, /&lt;script&gt;alert/)
})

test('reloadUrl cannot break out of the inline script block', () => {
  // JSON.stringify alone does not escape '<' — a reloadUrl containing a
  // script-tag sequence would otherwise terminate the inline <script>
  // element and inject markup/script into the error page.
  const html = buildRendererLoadErrorPage({
    errorDescription: 'load failed',
    reloadUrl: 'file:///C:/x/index.html</script><script>alert("pwned")</script>'
  })

  assert.doesNotMatch(html, /<\/script><script>/)
  assert.doesNotMatch(html, /alert\("pwned"\)/)
  // The payload is preserved as inert \u003c escapes inside the JS string.
  assert.match(html, /\\u003c\/script/)
  assert.match(html, /\\u003cscript\\u003ealert/)
})

test('error page renders without any details', () => {
  const html = buildRendererLoadErrorPage()

  assert.match(html, /The desktop renderer failed to load\./)
  assert.match(html, /Reload/)
})

test('loadRendererLoadErrorPage loads a data: URL and swallows loadURL rejections', async () => {
  const loads: string[] = []

  const win = {
    loadURL: async (url: string) => {
      loads.push(url)
    }
  }

  await loadRendererLoadErrorPage(win, { errorCode: -6, errorDescription: 'torn bundle' })

  assert.equal(loads.length, 1)
  assert.ok(loads[0].startsWith('data:text/html;charset=utf-8,'))
  assert.ok(decodeURIComponent(loads[0]).includes('torn bundle'))

  // A rejected loadURL must not become an unhandled rejection — the white
  // screen is strictly better than a crashed recovery path.
  const rejectingWin = {
    loadURL: async () => {
      throw new Error('boom')
    }
  }

  await assert.doesNotReject(() => loadRendererLoadErrorPage(rejectingWin))
})

test('crash page names the cause and offers restart buttons only for handlers that exist', () => {
  const html = buildRendererLoadErrorPage({
    title: 'Hermes\u2019 desktop UI keeps crashing',
    errorDescription: describeRendererCrashCause({ reason: 'oom', exitCode: -9 }),
    recovery: { restart: () => undefined }
  })

  assert.match(html, /keeps crashing/)
  assert.match(html, /ran out of memory \(exit code -9\)/)
  assert.match(html, /href="hermes-recovery:restart">Restart Hermes</)
  assert.doesNotMatch(html, /software rendering/)

  const both = buildRendererLoadErrorPage({
    recovery: { restart: () => undefined, restartSoftwareRendering: () => undefined }
  })

  assert.match(both, /hermes-recovery:restart-software-rendering">Restart with software rendering</)
  assert.match(describeRendererCrashCause({ reason: 'crashed', exitCode: 133 }), /crashed \(exit code 133\).*GPU driver/)
  assert.equal(parseRecoveryAction('hermes-recovery:restart'), 'restart')
  assert.equal(parseRecoveryAction('hermes-recovery://restart-software-rendering'), 'restartSoftwareRendering')
  assert.equal(parseRecoveryAction('file:///dist/index.html'), null)
})

test('recovery buttons are intercepted on will-navigate; other navigations pass through', async () => {
  const listeners: Array<(event: { preventDefault: () => void }, url: string) => void> = []
  const fired: string[] = []
  let prevented = 0

  const win = {
    loadURL: async () => undefined,
    webContents: {
      on: (_event: 'will-navigate', listener: (event: { preventDefault: () => void }, url: string) => void) => {
        listeners.push(listener)
      },
      removeListener: (_event: 'will-navigate', listener: (...args: any[]) => void) => {
        listeners.splice(listeners.indexOf(listener), 1)
      }
    }
  }

  await loadRendererLoadErrorPage(win, {
    recovery: {
      restart: () => {
        fired.push('restart')
      }
    }
  })

  assert.equal(listeners.length, 1)

  const preventDefault = () => {
    prevented += 1
  }

  // The Reload button's real-renderer navigation is not ours to block.
  listeners[0]({ preventDefault }, 'file:///dist/index.html')
  assert.equal(prevented, 0)
  assert.deepEqual(fired, [])

  // A recovery URL for a handler that is NOT offered is swallowed, not executed.
  listeners[0]({ preventDefault }, 'hermes-recovery:restart-software-rendering')
  assert.equal(prevented, 1)
  assert.deepEqual(fired, [])
  assert.equal(listeners.length, 1)

  listeners[0]({ preventDefault }, 'hermes-recovery:restart')
  assert.equal(prevented, 2)
  assert.deepEqual(fired, ['restart'])
  assert.equal(listeners.length, 0, 'intercept removes itself once it fired')
})
