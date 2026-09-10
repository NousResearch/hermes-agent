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

/** Specifier -> injected global, longest keys first (matches rewrite order). */
const SPECIFIER_GLOBALS = {
  '@hermes/plugin-sdk': '__HERMES_PLUGIN_SDK__',
  'react/jsx-dev-runtime': '__HERMES_REACT_JSX_DEV__',
  'react/jsx-runtime': '__HERMES_REACT_JSX__',
  react: '__HERMES_REACT__'
} as const

type GlobalKey = (typeof SPECIFIER_GLOBALS)[keyof typeof SPECIFIER_GLOBALS]

/** Read the injected namespaces NOW, not at module evaluation. This module
 *  sits in an import cycle (sdk/index -> contrib/* -> contrib/runtime-loader
 *  -> sdk/runtime -> sdk/index), and production bundlers may evaluate a
 *  module-scope capture before the bindings it references are assigned —
 *  which is exactly how every disk plugin ended up failing to load. Both
 *  entry points below run long after module evaluation, so reading here is
 *  always safe. */
export function pluginNamespaces(): Record<GlobalKey, Record<string, unknown> | undefined> {
  return {
    __HERMES_PLUGIN_SDK__: sdk as unknown as Record<string, unknown>,
    __HERMES_REACT__: React as unknown as Record<string, unknown>,
    __HERMES_REACT_JSX__: jsxRuntime as unknown as Record<string, unknown>,
    __HERMES_REACT_JSX_DEV__: jsxDevRuntime as unknown as Record<string, unknown>
  }
}

export function installPluginSdk(): void {
  Object.assign(globalThis, pluginNamespaces())
}

/** Shim body for one injected namespace. The namespace rides as a parameter
 *  (not read off the module table) so the missing-namespace contract is
 *  unit-testable without a renderer. */
export function shimSource(
  globalKey: string,
  namespace: Record<string, unknown> | null | undefined,
  specifier: string
): string {
  if (!namespace) {
    // A namespace can go missing when the bundler drops or renames a module
    // the map assumes (seen in the wild with react/jsx-dev-runtime): fail
    // LOUDLY at import time, naming the missing piece, instead of throwing
    // "Cannot convert undefined or null to object" while building the map —
    // which blocks EVERY disk plugin, including ones that never import the
    // missing specifier.
    return `throw new Error(${JSON.stringify(
      `Cannot load '${specifier}': the ${globalKey} namespace is missing from this app build. Update Hermes Desktop, then reload plugins.`
    )});\n`
  }

  const names = Object.keys(namespace).filter(name => name !== 'default' && /^[A-Za-z_$][\w$]*$/.test(name))

  return (
    `const m = globalThis.${globalKey};\n` +
    `export default m.default ?? m;\n` +
    // Guard the destructuring: `export const {  } = m` is a syntax error, so
    // only emit it when the namespace actually has named exports.
    (names.length ? `export const { ${names.join(', ')} } = m;\n` : '')
  )
}

/** Build a shim ESM blob that re-exports a global namespace's live members.
 *  Export names come from the namespace itself, so the list can't drift. */
function shimUrl(globalKey: GlobalKey, specifier: string): string {
  const source = shimSource(globalKey, pluginNamespaces()[globalKey], specifier)

  return URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))
}

let cached: Record<string, string> | null = null

/** Specifier -> shim URL map for the runtime loader (longest keys first). */
export function sdkImportMap(): Record<string, string> {
  cached ??= Object.fromEntries(
    Object.entries(SPECIFIER_GLOBALS).map(([specifier, globalKey]) => [specifier, shimUrl(globalKey, specifier)])
  )

  return cached
}
