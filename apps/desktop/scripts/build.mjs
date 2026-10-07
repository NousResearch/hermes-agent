// npm's source-development composition; the compiler consumes prepared inputs.
import { execFileSync } from 'node:child_process'
import { cpSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { isMain, repoRoot } from '../../../scripts/build/frontend-common.mjs'
import { readNativeInputs } from './prepared-native-deps.mjs'

export function buildSourceDesktop({ source = repoRoot, icons, run = execFileSync } = {}) {
  source = resolve(source)
  const app = join(source, 'apps/desktop')
  const step = (script, args = []) => run(process.execPath, [join(source, script), ...args], { cwd: app, stdio: 'inherit' })
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
  // The native inputs carry their own receipt (inputs identity + tree digest): restaging the
  // same sources recompiles two universal helpers for identical bytes on every update.
  const nativeDeps = join(app, 'build/native-deps')
  if (!nativeCurrent({ source, nativeDeps })) step('apps/desktop/scripts/stage-native-deps.mjs')
  step('scripts/build/desktop.mjs', ['--source', source, '--icons', icons,
    '--stamp', join(app, 'build/install-stamp.json'), '--native-deps', nativeDeps, '--out', join(app, 'dist')])
}

function nativeCurrent(inputs) {
  try {
    readNativeInputs(inputs)
    return true
  } catch {
    return false
  }
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: { icons: { type: 'string' } } })
  buildSourceDesktop(values)
}
