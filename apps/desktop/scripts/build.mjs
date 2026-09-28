// npm's source-development composition; the compiler consumes prepared inputs.
import { execFileSync } from 'node:child_process'
import { cpSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { isMain, repoRoot } from '../../../scripts/build/frontend-common.mjs'

export function buildSourceDesktop({ source = repoRoot, icons, run = execFileSync } = {}) {
  source = resolve(source)
  const app = join(source, 'apps/desktop')
  const step = (script, args = []) => run(process.execPath, [join(source, script), ...args], { cwd: app, stdio: 'inherit' })
  step('apps/desktop/scripts/assert-root-install.mjs')
  // Default-brand icons are committed; only flavored release builds pass --icons.
  icons = resolve(icons ?? source)
  if (icons !== source) {
    // electron-builder consumes packaging artwork in the workspace. Copy the
    // prepared pixels; do not create another Python environment to redraw them.
    cpSync(join(icons, 'apps/desktop/assets'), join(app, 'assets'), { recursive: true })
  }
  step('apps/desktop/scripts/write-build-stamp.mjs')
  step('apps/desktop/scripts/stage-native-deps.mjs')
  step('scripts/build/desktop.mjs', ['--source', source, '--icons', icons,
    '--stamp', join(app, 'build/install-stamp.json'), '--native-deps', join(app, 'build/native-deps'), '--out', join(app, 'dist')])
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: { icons: { type: 'string' } } })
  buildSourceDesktop(values)
}
