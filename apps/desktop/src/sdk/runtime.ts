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

/** The namespaces exposed to runtime plugins.
 *
 *  Built ON CALL, never hoisted into a module-level object literal. The
 *  bundler is free to merge this module with `./index` into one chunk and
 *  emit our statements BEFORE the SDK namespace object is assigned; a
 *  module-level `{ __HERMES_PLUGIN_SDK__: sdk }` would then capture
 *  `undefined` forever and every plugin would fall back to plain text.
 *  Reading `sdk` inside a function defers it past chunk initialisation. */
const globals = () =>
  ({
    __HERMES_PLUGIN_SDK__: sdk,
    __HERMES_REACT__: React,
    __HERMES_REACT_JSX__: jsxRuntime,
    __HERMES_REACT_JSX_DEV__: jsxDevRuntime
  }) as const

type GlobalKey = keyof ReturnType<typeof globals>

export function installPluginSdk(): void {
  const ns = globals()

  // Fail loudly instead of shipping a namespace that silently resolves to
  // `undefined` — a plugin importing it would render nothing but its text.
  for (const [key, value] of Object.entries(ns)) {
    if (!value || typeof value !== 'object') {
      throw new Error(`[plugin-sdk] ${key} is not initialised — SDK namespace evaluated too early`)
    }
  }

  Object.assign(globalThis, ns)
}

/** Build a shim ESM blob that re-exports a global namespace's live members.
 *  Export names come from the namespace itself, so the list can't drift. */
function shimUrl(globalKey: GlobalKey): string {
  const names = Object.keys(globals()[globalKey]).filter(name => name !== 'default' && /^[A-Za-z_$][\w$]*$/.test(name))

  const source =
    `const m = globalThis.${globalKey};\n` +
    `export default m.default ?? m;\n` +
    // Guard the destructuring: `export const {  } = m` is a syntax error, so
    // only emit it when the namespace actually has named exports.
    (names.length ? `export const { ${names.join(', ')} } = m;\n` : '')

  return URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))
}

let cached: Record<string, string> | null = null

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
