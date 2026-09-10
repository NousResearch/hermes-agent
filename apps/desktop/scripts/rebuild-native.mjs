// rebuild-native.mjs
import { rebuild } from '@electron/rebuild'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { isMain } from './utils.mjs'
import packageJson from '../package.json' with { type: 'json' }
const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')

export async function rebuildNodePty({ arch = process.arch } = {}) {
  await rebuild({
    buildPath: projectRoot, // where node_modules lives
    electronVersion: packageJson.devDependencies.electron.replace('^', ''),
    arch,
    onlyModules: ['node-pty'],
    force: true
  })
}

if (isMain(import.meta.url)) {
  const [arch] = process.argv.slice(2)
  await rebuildNodePty({ arch })
}
