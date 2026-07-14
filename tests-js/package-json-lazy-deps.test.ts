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
 * - ``agent-browser`` is NOT a root dependency. It used to be eager (see
 *   #27055, which reasoned its postinstall was small enough to keep eager
 *   unlike Camofox's) but #43564 found that keeping ANY dependency in root
 *   ``package.json`` — however small its postinstall — entangles it with
 *   the ui-tui/web workspace install and risks it being pruned. It now
 *   resolves at runtime via ``npx agent-browser`` (see
 *   ``tools/browser_tool.py::_find_agent_browser``), which sidesteps the
 *   workspace graph entirely. ``hermes update`` and ``hermes doctor --fix``
 *   both fire-and-forget ``warm_agent_browser_npx_cache()`` to keep npx's
 *   own cache warm, preserving the "available before any session starts"
 *   property #27055 cared about without re-entangling the dependency.
 *
 * - ``@streamdown/math`` is NOT a root dependency either. It's imported only
 *   by desktop's own TS code (``apps/desktop/src/...``), so it belongs in
 *   ``apps/desktop/package.json`` (alongside its sibling ``@streamdown/code``)
 *   — not root, where it was subject to the exact same pruning risk.
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
const ROOT_LOCK = path.join(REPO_ROOT, 'package-lock.json')
const DESKTOP_PKG = path.join(REPO_ROOT, 'apps', 'desktop', 'package.json')

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

test('agent-browser is not in root dependencies (resolves via npx, #43564)', () => {
  const deps = (rootPackageJson().dependencies ?? {}) as Record<string, string>
  assert.ok(
    !('agent-browser' in deps),
    'agent-browser must not be a root package.json dependency — it ' +
      'resolves lazily via `npx agent-browser` instead (see ' +
      'tools/browser_tool.py::_find_agent_browser and ' +
      'warm_agent_browser_npx_cache). Putting it back in root ' +
      'dependencies re-entangles it with the ui-tui/web workspace ' +
      'install graph and reintroduces #43564.'
  )
})

test('@streamdown/math is not in root dependencies (desktop-only import)', () => {
  const deps = (rootPackageJson().dependencies ?? {}) as Record<string, string>
  assert.ok(
    !('@streamdown/math' in deps),
    '@streamdown/math is only imported by apps/desktop\'s own TS code ' +
      '(markdown-text.tsx, katex-memo.ts) — it belongs in ' +
      'apps/desktop/package.json alongside its sibling @streamdown/code, ' +
      'not root, where it\'s subject to the same workspace-pruning risk ' +
      'agent-browser had (#43564).'
  )
})

test('@streamdown/math is in desktop dependencies', () => {
  const deps = (JSON.parse(fs.readFileSync(DESKTOP_PKG, 'utf-8')).dependencies ??
    {}) as Record<string, string>

  assert.ok(
    '@streamdown/math' in deps,
    '@streamdown/math is imported by apps/desktop\'s own TS code ' +
      '(markdown-text.tsx, katex-memo.ts) and must be declared in ' +
      'apps/desktop/package.json now that it is no longer a root ' +
      'dependency.'
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
