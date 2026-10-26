/**
 * Invariants for what is eager vs lazy in the root ``package.json``.
 *
 * The root ``package.json`` is installed by ``hermes update`` on every user,
 * including users who never opted into a given browser backend. Anything
 * listed in ``dependencies`` therefore runs its npm postinstall script for
 * everyone, and — per #43564 — is also part of the npm workspace install
 * graph, where a workspace-scoped ``npm ci`` (``--workspace ui-tui
 * --workspace web``) can silently prune it right back out on the next
 * ``hermes update``.
 *
 * The contract:
 *
 * - ``@streamdown/math`` is NOT a root dependency. It's imported only
 *   by desktop's own TS code (``apps/desktop/src/...``), so it belongs in
 *   ``apps/desktop/package.json`` (alongside its sibling ``@streamdown/code``)
 *   — not root, where it was subject to the exact same pruning risk.
 *
 * - ``@askjo/camofox-browser`` is NOT eager. Camofox is an opt-in browser
 *   backend that ships as the ``camofox`` catalog plugin (Hermes core carries
 *   no Camofox code). Its postinstall fetches a ~300MB Firefox-fork binary,
 *   which silently blocked ``hermes update`` for multi-minute stretches on
 *   slow / network-restricted connections (notably users in China running
 *   through a VPN), so it installs only when the user sets the plugin up.
 *
 * (agent-browser, once listed here, is no longer used by Hermes in any form:
 * browser_exec launches the pm-pinned Chromium itself.)
 *
 * If a future PR re-adds any of these to root ``dependencies``, this test
 * fails — read the lazy-install guidance in the ``hermes-agent-dev`` skill
 * before changing the expectations.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const ROOT_PKG = path.join(REPO_ROOT, 'package.json')

function rootPackageJson(): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(ROOT_PKG, 'utf-8'))
}

test('camofox is not in root dependencies (must stay opt-in)', () => {
  const deps = (rootPackageJson().dependencies ?? {}) as Record<string, string>
  assert.ok(
    !('@askjo/camofox-browser' in deps),
    'Camofox is a ~300MB binary-postinstall backend that must stay ' +
      'out of root package.json dependencies. It ships as the camofox ' +
      'catalog plugin, which installs it only when the user sets that ' +
      'plugin up.'
  )
})
