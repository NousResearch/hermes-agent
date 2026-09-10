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

type SdkGlobals = {
  __HERMES_PLUGIN_SDK__: typeof sdk
  __HERMES_REACT__: typeof React
  __HERMES_REACT_JSX__: typeof jsxRuntime
  __HERMES_REACT_JSX_DEV__: typeof jsxDevRuntime
}

/** The live namespaces — resolved on EVERY call, never captured at module init.
 *
 *  LOCAL PATCH (2026-09-10): a module-level map can be built BEFORE `./index`
 *  assigns its namespace binding (bundler ordering is not a contract), and
 *  `Object.keys(undefined)` in shimUrl then fails EVERY runtime plugin with
 *  "Cannot convert undefined or null to object". Reading the bindings here is
 *  order-independent. */
function sdkGlobals(): SdkGlobals {
  return {
    __HERMES_PLUGIN_SDK__: sdk,
    __HERMES_REACT__: React,
    __HERMES_REACT_JSX__: jsxRuntime,
    __HERMES_REACT_JSX_DEV__: jsxDevRuntime
  }
}

export function installPluginSdk(): void {
  Object.assign(globalThis, sdkGlobals())
}

/** Build a shim ESM blob that re-exports a global namespace's live members.
 *  Export names come from the namespace itself, so the list can't drift. */
function shimUrl(globalKey: keyof SdkGlobals): string {
  const names = Object.keys(sdkGlobals()[globalKey]).filter(name => name !== 'default' && /^[A-Za-z_$][\w$]*$/.test(name))

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
