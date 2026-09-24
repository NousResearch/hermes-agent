// `hermes:sdk-versions` (#120879): the Settings -> Providers "SDK and Runtime
// Versions" panel's read door. Each agent SDK's version is resolved at
// runtime from its installed package.json - never hardcoded - so the panel
// always reflects what is actually on the machine. A missing or unreadable
// install reports null; the panel renders that as "Not installed".
import fs from 'node:fs'
import path from 'node:path'

import { ipcMain } from 'electron'

export interface SdkVersionEntry {
  name: string
  /** Installed version from the package's package.json; null when absent. */
  version: null | string
}

export interface SdkVersionsInfo {
  electron: string
  node: string
  sdks: SdkVersionEntry[]
}

export interface SdkVersionsDeps {
  /** The app's own root; its node_modules covers dev checkouts/bundled deps. */
  appPath: string
  hermesHome: string
  env: NodeJS.ProcessEnv
  platform: NodeJS.Platform
}

// The agent SDKs the panel promises to report, in display order.
export const SDK_PACKAGES = [
  '@anthropic-ai/claude-code',
  '@anthropic-ai/claude-agent-sdk',
  '@google/genai',
  '@openai/codex',
  '@modelcontextprotocol/sdk'
] as const

/**
 * node_modules roots to search, most-authoritative first. Hermes's own global
 * installs go to the managed node tree (`npm i -g --prefix $HERMES_HOME/node`,
 * see scripts/install.sh): `<prefix>/lib/node_modules` on posix,
 * `<prefix>/node_modules` on Windows. After that come the app's own
 * node_modules, any NODE_PATH entries, and npm's platform-global default.
 */
export function sdkModuleRoots({ appPath, env, hermesHome, platform }: SdkVersionsDeps): string[] {
  const roots = [
    path.join(hermesHome, 'node', 'node_modules'),
    path.join(hermesHome, 'node', 'lib', 'node_modules'),
    path.join(appPath, 'node_modules')
  ]

  for (const entry of (env.NODE_PATH ?? '').split(path.delimiter)) {
    if (entry) {
      roots.push(entry)
    }
  }

  if (platform === 'win32') {
    if (env.APPDATA) {
      roots.push(path.join(env.APPDATA, 'npm', 'node_modules'))
    }
  } else {
    roots.push('/usr/local/lib/node_modules', '/usr/lib/node_modules')
  }

  return roots
}

/** First readable `<root>/<pkg>/package.json` version wins; null when none. */
function readInstalledVersion(roots: readonly string[], pkg: string): null | string {
  for (const root of roots) {
    try {
      const raw = fs.readFileSync(path.join(root, pkg, 'package.json'), 'utf8')
      const parsed: unknown = JSON.parse(raw)

      if (parsed && typeof parsed === 'object') {
        const version = (parsed as { version?: unknown }).version

        if (typeof version === 'string' && version) {
          return version
        }
      }
    } catch {
      // Not here (or unreadable/malformed): keep looking in the next root.
    }
  }

  return null
}

export function collectSdkVersions(deps: SdkVersionsDeps): SdkVersionsInfo {
  const roots = sdkModuleRoots(deps)

  return {
    electron: process.versions.electron ?? '',
    node: process.versions.node ?? '',
    sdks: SDK_PACKAGES.map(name => ({ name, version: readInstalledVersion(roots, name) }))
  }
}

export function registerSdkVersionsIpc(deps: SdkVersionsDeps) {
  ipcMain.handle('hermes:sdk-versions', async () => collectSdkVersions(deps))
}
