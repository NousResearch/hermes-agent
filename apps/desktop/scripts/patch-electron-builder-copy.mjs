// builder-util's walk() drops .gitkeep/.DS_Store and copyDir() never makes empty dirs; the payload needs every entry.
import path from 'node:path'
import { createRequire, registerHooks } from 'node:module'
import { pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const builderUtil = createRequire(require.resolve('app-builder-lib')).resolve('builder-util')
const fsUrl = pathToFileURL(path.join(path.dirname(builderUtil), 'fs.js')).href

const patches = [
  [
    `            if (name === ".DS_Store" || name === ".gitkeep") {
                return null;
            }
`,
    ''
  ],
  [
    `        consume: async (file, stat, parent) => {
            if (!stat.isFile() && !stat.isSymbolicLink()) {`,
    `        consume: async (file, stat, parent) => {
            if (stat.isDirectory()) {
                await mkdir(file.replace(src, destination), { recursive: true });
                return;
            }
            if (!stat.isFile() && !stat.isSymbolicLink()) {`
  ]
]

// Transform builder-util in memory only; an upstream shape change must fail the build, not drop files again.
registerHooks({
  load(url, context, nextLoad) {
    const loaded = nextLoad(url, context)
    if (url !== fsUrl) return loaded
    let source = loaded.source.toString()
    for (const [needle, replacement] of patches) {
      if (!source.includes(needle)) {
        throw new Error('builder-util copy shape changed; revalidate patch-electron-builder-copy.mjs')
      }
      source = source.replace(needle, replacement)
    }
    return { ...loaded, source }
  }
})
