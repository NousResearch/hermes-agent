import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { describe, test } from 'vitest'
import { resolveConfig } from 'vite'

const REPO_ROOT = path.resolve(__dirname, '..')

/** Every Vite/Vitest config the app loads, and the directory its `@` alias must point at.
 *
 * Vite's `configLoader: 'bundle'` default bundles the config with Rolldown and injects
 * `__dirname`/`__filename` shims. `'native'` is slated to become the default, and there the
 * config is imported as a plain ES module, so a CJS `__dirname` reference throws before the
 * build starts — the configs must resolve their own directory with `import.meta.dirname`.
 */
const CONFIG_ALIASES: Record<string, string> = {
  'web/vite.config.ts': 'web/src',
  'web/vitest.config.ts': 'web/src',
  'apps/desktop/vite.config.ts': 'apps/desktop/src',
  'apps/bootstrap-installer/vite.config.ts': 'apps/bootstrap-installer/src',
}

/** Configs whose imports come from their own workspace, plus a package only that workspace
 *  installs. A checkout without it (a dashboard-only install skips `apps/desktop`) cannot load
 *  the config at all, so the case skips instead of failing on an unrelated missing package;
 *  the JS tests job installs the full workspace and runs it. */
const WORKSPACE_DEPENDENCIES: Record<string, string[]> = {
  'apps/desktop/vite.config.ts': ['apps/desktop/node_modules/driver.js', 'node_modules/driver.js'],
}

function resolveAlias(config: Awaited<ReturnType<typeof resolveConfig>>, find: string): string {
  const entry = (config.resolve.alias ?? []).find((alias) => alias.find === find)
  assert.ok(entry, `alias ${find} is missing`)
  return entry.replacement
}

describe('Vite configs load under the native config loader', () => {
  for (const [configPath, aliasTarget] of Object.entries(CONFIG_ALIASES)) {
    const dependencies = WORKSPACE_DEPENDENCIES[configPath]
    const skip =
      dependencies !== undefined &&
      !dependencies.some((relativePath) => fs.existsSync(path.join(REPO_ROOT, relativePath)))

    test.skipIf(skip)(`${configPath} resolves @ from its own directory`, async () => {
      const config = await resolveConfig(
        { configFile: path.join(REPO_ROOT, configPath), configLoader: 'native' },
        'build',
        'production'
      )

      assert.equal(resolveAlias(config, '@'), path.join(REPO_ROOT, aliasTarget))
    }, 120_000)
  }
})
