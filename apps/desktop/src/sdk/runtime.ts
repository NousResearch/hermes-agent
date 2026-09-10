/**
 * Runtime SDK injection — the other half of the vscode-module model. Bundled
 * plugins resolve `@hermes/plugin-sdk` through the vite alias; RUNTIME-loaded
 * plugins (disk / fetched) import the same specifier and get the same object:
 * the loader rewrites bare specifiers to shim modules that re-export the
 * live namespaces installed here. React ships as the app's singletons —
 * a second React instance would break hooks.
 */

import * as React from 'react'
import * as jsxDevRuntime from 'react/jsx-dev-runtime'
import * as jsxRuntime from 'react/jsx-runtime'

import * as sdk from './index'

/**
 * Resolve live namespaces at call time — never snapshot them in a
 * module-scope object. This file sits in
 * `sdk/index → contrib/* → runtime-loader → sdk/runtime → sdk/index`.
 * Rolldown emits the SDK namespace as a hoisted `var`; a module-scope
 * `{ __HERMES_PLUGIN_SDK__: sdk }` therefore captures `undefined` and
 * every disk plugin that imports `{ host, cn }` fails to link. Callers
 * (`installPluginSdk`, `shimUrl`) only run after the app is up.
 */
export function pluginNamespaces() {
  return {
    __HERMES_PLUGIN_SDK__: sdk,
    __HERMES_REACT__: React,
    __HERMES_REACT_JSX__: jsxRuntime,
    __HERMES_REACT_JSX_DEV__: jsxDevRuntime
  }
}

type PluginGlobalKey = keyof ReturnType<typeof pluginNamespaces>

export function installPluginSdk(): void {
  Object.assign(globalThis, pluginNamespaces())
}

/**
 * Named exports a live `import * as` namespace can re-export on a shim.
 * A cycle-time snapshot or a missing production binding (notably
 * `react/jsx-dev-runtime`) can still be null — `Object.keys` then throws
 * `TypeError: Cannot convert undefined or null to object` and kills
 * every disk plugin because `sdkImportMap()` walks all four slots first.
 */
export function namespaceExportNames(ns: unknown): string[] {
  if (ns == null || (typeof ns !== 'object' && typeof ns !== 'function')) {
    return []
  }

  return Object.keys(ns).filter(name => name !== 'default' && /^[A-Za-z_$][\w$]*$/.test(name))
}

/** Build a shim ESM blob that re-exports a global namespace's live members.
 *  Export names come from the namespace itself, so the list can't drift. */
function shimUrl(globalKey: PluginGlobalKey): string {
  const names = namespaceExportNames(pluginNamespaces()[globalKey])

  // Read the live global at evaluation time. If installPluginSdk() did not
  // stick (or a slot is null), coerce to {} so `m.default` / destructure
  // cannot throw TypeError: Cannot convert undefined or null to object.
  const source =
    `const m = globalThis.${globalKey};\n` +
    `const ns = m != null && typeof m === 'object' ? m : {};\n` +
    `export default ns.default ?? ns;\n` +
    // Guard the destructuring: `export const {  } = m` is a syntax error, so
    // only emit it when the namespace actually has named exports.
    (names.length ? `export const { ${names.join(', ')} } = ns;\n` : '')

  return URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))
}

let cached: Record<string, string> | null = null

/** Drop the specifier→shim cache so a later `sdkImportMap()` rebuilds blobs. */
export function resetSdkImportMapCache(): void {
  cached = null
}

/** Specifier -> shim URL map for the runtime loader (longest keys first). */
export function sdkImportMap(): Record<string, string> {
  cached ??= {
    '@hermes/plugin-sdk': shimUrl('__HERMES_PLUGIN_SDK__'),
    'react/jsx-dev-runtime': shimUrl('__HERMES_REACT_JSX_DEV__'),
    'react/jsx-runtime': shimUrl('__HERMES_REACT_JSX__'),
    react: shimUrl('__HERMES_REACT__')
  }

  return cached
}
