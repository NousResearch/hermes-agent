/**
 * Invariants for what is eager vs lazy in the root ``package.json``.
 *
 * The root ``package.json`` is installed by ``hermes update`` on every user,
 * including users who never opted into a given browser backend. Anything
 * listed in ``dependencies`` therefore runs its npm postinstall script for
 * everyone — including binary-fetching backends, on every update.
 *
 * The contract:
 *
 * - ``agent-browser`` IS eager. It is the default Chromium-driving backend
 *   used whenever the agent makes a browser call without a cloud provider
 *   configured, so it must already be installed before any session starts.
 *   Its postinstall is also small.
 *
 * - ``@askjo/camofox-browser`` is NOT eager. It is an explicit opt-in
 *   alternative browser backend, selected by the user via
 *   ``hermes tools`` → Browser Automation → Camofox, and only used at
 *   runtime when ``CAMOFOX_URL`` is set. Its postinstall fetches a ~300MB
 *   Firefox-fork binary, which silently blocked ``hermes update`` for
 *   multi-minute stretches on slow / network-restricted connections
 *   (notably users in China running through a VPN). The package is
 *   installed on demand by ``tools_config.py`` ``post_setup_key ==
 *   "camofox"`` when the user actually selects Camofox.
 *
 * If a future PR re-adds Camofox (or any other binary-postinstall package)
 * to root ``dependencies``, this test fails — read the lazy-install
 * guidance in the ``hermes-agent-dev`` skill before changing the
 * expectations.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const ROOT_PKG = path.join(REPO_ROOT, 'package.json')
const ROOT_LOCK = path.join(REPO_ROOT, 'package-lock.json')

function rootPackageJson(): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(ROOT_PKG, 'utf-8'))
}

test('camofox is not in root dependencies (must stay opt-in)', () => {
  const deps = (rootPackageJson().dependencies ?? {}) as Record<string, string>
  assert.ok(
    !('@askjo/camofox-browser' in deps),
    'Camofox is a ~300MB binary-postinstall backend that must stay ' +
      'out of root package.json dependencies. It belongs in the ' +
      'Camofox post_setup handler in hermes_cli/tools_config.py so it ' +
      'only installs when the user explicitly selects Camofox via ' +
      '`hermes tools` → Browser Automation → Camofox.'
  )
})

test('agent-browser stays eager (default backend)', () => {
  const deps = (rootPackageJson().dependencies ?? {}) as Record<string, string>
  assert.ok(
    'agent-browser' in deps,
    'agent-browser is the default browser-tool backend used by every ' +
      'session that doesn\'t have a cloud browser provider configured. ' +
      'It must stay in root package.json dependencies so it is present ' +
      'after `hermes setup` / `hermes update` without an explicit ' +
      'post_setup step.'
  )
})

test('root lockfile has no camofox entries', () => {
  if (!fs.existsSync(ROOT_LOCK)) {
    // Some CI matrix shards skip lockfile materialization.
    return
  }

  const text = fs.readFileSync(ROOT_LOCK, 'utf-8')
  assert.ok(
    !text.includes('@askjo/camofox-browser'),
    'package-lock.json still references @askjo/camofox-browser. ' +
      'Regenerate the lockfile after removing the dep: ' +
      '`rm package-lock.json && npm install --package-lock-only ' +
      '--ignore-scripts --no-fund --no-audit`.'
  )
  assert.ok(
    !text.includes('camoufox-js'),
    'package-lock.json still references camoufox-js (transitive of ' +
      '@askjo/camofox-browser). Regenerate the lockfile.'
  )
})
