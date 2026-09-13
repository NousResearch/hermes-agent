import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
const directory = path.dirname(fileURLToPath(import.meta.url))
const require = createRequire(path.resolve(directory, '../../apps/desktop/package.json'))
const { build } = require('esbuild')
await build({
  entryPoints: [path.join(directory, 'electron-main.ts')], outfile: process.argv[2],
  bundle: true, platform: 'node', format: 'cjs', target: 'node22', external: ['electron'],
  nodePaths: [path.resolve(directory, '../../node_modules'), path.resolve(directory, '../../apps/desktop/node_modules')]
})
