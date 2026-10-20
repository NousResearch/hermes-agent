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

// Resolved LAZILY, never as a module-scope literal. This module sits in an
// import cycle — `sdk/index` → `contrib/*` → `contrib/runtime-loader` →
// `sdk/runtime` — so a module-scope `{ __HERMES_PLUGIN_SDK__: sdk, … }` is
// evaluated BEFORE `sdk/index`'s own body runs. In the bundled app that read
// yields `undefined` (the bundler emits the namespace as a hoisted `var`), so
// `Object.keys(GLOBALS.__HERMES_PLUGIN_SDK__)` threw
// "Cannot convert undefined or null to object" and EVERY runtime (disk)
// plugin failed to load. Reading them at call time — installPluginSdk() and
// the shim builder only ever run once the app is up — gets the live
// namespaces.
function pluginNamespaces() {
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

/** Reserved words pass the identifier shape test below but are a SyntaxError as a destructuring
 *  binding. A minified SDK chunk can leak one as an export alias (proven: `in`), which produced
 *  `export const { in } = m` and surfaced in Chromium as a bare "Unexpected token ')'" pointing
 *  at the generated blob, not at the plugin that caused it. */
const RESERVED_WORDS = new Set([
  'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete',
  'do', 'else', 'enum', 'export', 'extends', 'false', 'finally', 'for', 'function', 'if',
  'import', 'in', 'instanceof', 'new', 'null', 'return', 'super', 'switch', 'this', 'throw',
  'true', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'await',
  'implements', 'package', 'protected', 'interface', 'private', 'public'
])

/** The shim's own local binding must not collide with an export name: a minified chunk can export
 *  ANY valid identifier, including the name the shim uses for the namespace itself — the
 *  session-list-density chunk exports `m`, which made `export const { m } = m` an
 *  "Identifier 'm' has already been declared". An unlikely name, also excluded from the
 *  destructure list, keeps a future collision impossible. */
const SHIM_LOCAL = '__hermes_shim_ns__'

/** The shim's ESM source for one global namespace.
 *
 *  Pure and exported so the emitted source can be asserted in a unit test: the real shim lives in
 *  a Blob URL and only ever executes inside Chromium, where the failure appears as a SyntaxError
 *  from generated code. */
export function shimSource(globalKey: string, namespace: object): string {
  const names = Object.keys(namespace).filter(
    name =>
      name !== 'default' &&
      name !== SHIM_LOCAL &&
      /^[A-Za-z_$][\w$]*$/.test(name) &&
      !RESERVED_WORDS.has(name)
  )

  return (
    `const ${SHIM_LOCAL} = globalThis.${globalKey};\n` +
    `export default ${SHIM_LOCAL}.default ?? ${SHIM_LOCAL};\n` +
    // Guard the destructuring: `export const {  } = m` is a syntax error, so
    // only emit it when the namespace actually has named exports.
    (names.length ? `export const { ${names.join(', ')} } = ${SHIM_LOCAL};\n` : '')
  )
}

/** Build a shim ESM blob that re-exports a global namespace's live members.
 *  Export names come from the namespace itself, so the list can't drift. */
function shimUrl(globalKey: PluginGlobalKey): string {
  const source = shimSource(globalKey, pluginNamespaces()[globalKey])

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
