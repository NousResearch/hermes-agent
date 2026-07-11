// rebuild-native.mjs — ensure host-native modules exist before packaging.
import { rebuild } from '@electron/rebuild'
import { rmSync } from 'node:fs'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { isMain } from './utils.mjs'
import {
  assertNodePtyNativePayload,
  findNodePtyNativePayload,
  isHostTarget,
  resolveNodePtyRoot
} from './stage-native-deps.mjs'
import packageJson from '../package.json' with { type: 'json' }
const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const workspaceRoot = resolve(projectRoot, '../..')

export async function rebuildNodePty({ platform = process.platform, arch = process.arch } = {}) {
  if (!isHostTarget(platform, arch)) {
    if (findNodePtyNativePayload({ platform, arch })) return false
    throw new Error(
      `cannot cross-compile node-pty for ${platform}-${arch} from ` +
        `${process.platform}-${process.arch}; install a target prebuild instead`
    )
  }

  // A non-empty local binding may still target Node's ABI, the wrong CPU, or
  // be corrupt. Remove local candidates and always rebuild for this Electron
  // version so they cannot shadow a usable prebuild at runtime.
  const nodePtyRoot = resolveNodePtyRoot()
  rmSync(join(nodePtyRoot, 'build', 'Release'), { recursive: true, force: true })
  rmSync(join(nodePtyRoot, 'build', 'Debug'), { recursive: true, force: true })

  console.log(`[rebuild-native] compiling node-pty for Electron ${packageJson.devDependencies.electron}`)
  await rebuild({
    // Read dependencies from the desktop workspace, but keep walking through
    // the npm workspace root where node-pty is actually hoisted. Without
    // projectRootPath, @electron/rebuild stops at apps/desktop/package.json and
    // exits successfully without finding or rebuilding anything.
    buildPath: projectRoot,
    projectRootPath: workspaceRoot,
    electronVersion: packageJson.devDependencies.electron.replace('^', ''),
    platform,
    arch,
    onlyModules: ['node-pty'],
    force: true,
    buildFromSource: true
  })

  assertNodePtyNativePayload(nodePtyRoot, { platform, arch })
  console.log(`[rebuild-native] node-pty ready for ${platform}-${arch}`)
  return true
}

if (isMain(import.meta.url)) {
  const [arch] = process.argv.slice(2)
  await rebuildNodePty({ arch })
}
