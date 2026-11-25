import assert from 'node:assert/strict'

import { test } from 'vitest'

import { handleLaunchDeepLink, registerDeepLinkProtocol } from './deep-link-protocol'

function run(env: NodeJS.ProcessEnv, defaultApp = true) {
  const calls: unknown[][] = []
  const log: string[] = []
  registerDeepLinkProtocol({
    app: {
      setAsDefaultProtocolClient: (...args: unknown[]) => {
        calls.push(args)

        return true
      }
    },
    protocol: 'hermes',
    env,
    defaultApp,
    argv: ['/electron', '.'],
    execPath: '/electron',
    resolve: entry => `/checkout/${entry}`,
    log: line => log.push(line)
  })

  return { calls, log }
}

test('the E2E/dev guard leaves the OS hermes:// handler untouched; normal launches still register', () => {
  const skipped = run({ HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION: '1' })
  assert.deepEqual(skipped.calls, [])
  assert.match(skipped.log.join('\n'), /registration skipped/)

  assert.deepEqual(run({}).calls, [['hermes', '/electron', ['/checkout/.']]])
  assert.deepEqual(run({}, false).calls, [['hermes']])
})

function launch(url: string, ready = true) {
  let activated = 0
  const handled = handleLaunchDeepLink(url, ['hermes-dev', 'hermes'], {
    ready,
    activate: () => {
      activated += 1
    }
  })

  return { handled, activated }
}

test('hermes://launch only focuses the app, whatever it carries, malformed tails included', () => {
  for (const url of [
    'hermes://launch',
    'hermes://launch/',
    'hermes://launch/chat/new?prompt=rm%20-rf',
    'hermes://launch/%',
    'hermes-dev://launch?x=1'
  ]) {
    assert.deepEqual(launch(url), { handled: true, activated: 1 }, url)
  }

  // A window still booting reveals itself; the link is consumed without an early show.
  assert.deepEqual(launch('hermes://launch', false), { handled: true, activated: 0 })

  for (const url of [
    'hermes://',
    'hermes://open/launch',
    'hermes://plugin/install?repo=launch',
    'hermes://launcher',
    'hermes://close-preview',
    'other://launch',
    'not a url'
  ]) {
    assert.deepEqual(launch(url), { handled: false, activated: 0 }, url)
  }
})
