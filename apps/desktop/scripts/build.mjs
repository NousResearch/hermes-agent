// npm's source-development composition; the compiler consumes prepared inputs.
import { execFileSync } from 'node:child_process'
import { cpSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { isMain, repoRoot } from '../../../scripts/build/frontend-common.mjs'

// The vite/rolldown transform of the desktop tree outgrows the default V8 heap
// (~4 GiB) on 8-16 GB machines and dies with "Zone Allocation failed" (#125502).
// Same shape as run-electron-builder.mjs's builderNodeOptions: set here on the
// step() children, not via a cross-env prefix on the `build` script (cross-env
// strips `'` from forwarded arguments (#103010) and needs its bin installed
// (#110121)) — main deliberately removed that prefix (d968ee260bf).
const HEAP_FLAG = '--max-old-space-size=16384'

/**
 * Inherited NODE_OPTIONS stay byte-identical (quoted preload paths survive); the
 * heap flag goes last so it wins. @param {string} [inherited] @returns {string}
 */
export function buildNodeOptions(inherited = process.env.NODE_OPTIONS ?? '') {
  return `${inherited} ${HEAP_FLAG}`.trim()
}

export function buildSourceDesktop({ source = repoRoot, icons, run = execFileSync } = {}) {
  source = resolve(source)
  const app = join(source, 'apps/desktop')
  const env = { ...process.env, NODE_OPTIONS: buildNodeOptions() }
  const step = (script, args = []) =>
    run(process.execPath, [join(source, script), ...args], { cwd: app, stdio: 'inherit', env })
  step('apps/desktop/scripts/assert-root-install.mjs')
  // Stamp before the flavored icons land: they overwrite committed files, and
  // the stamp's git check would then call the checkout dirty.
  step('apps/desktop/scripts/write-build-stamp.mjs')
  // Default-brand icons are committed; only flavored release builds pass --icons.
  icons = resolve(icons ?? source)
  if (icons !== source) {
    // electron-builder consumes packaging artwork in the workspace. Copy the
    // prepared pixels; do not create another Python environment to redraw them.
    cpSync(join(icons, 'apps/desktop/assets'), join(app, 'assets'), { recursive: true })
  }
  // locales/_keys.desktop.json is a committed artifact (i18n-keys.test.mjs pins it to en.ts);
  // the build must not write into the checkout — a dirty tree breaks `hermes update`.
  step('apps/desktop/scripts/stage-native-deps.mjs')
  step('scripts/build/desktop.mjs', ['--source', source, '--icons', icons,
    '--stamp', join(app, 'build/install-stamp.json'), '--native-deps', join(app, 'build/native-deps'), '--out', join(app, 'dist')])
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: { icons: { type: 'string' } } })
  buildSourceDesktop(values)
}
