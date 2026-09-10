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

const GLOBALS = {
  __HERMES_PLUGIN_SDK__: sdk,
  __HERMES_REACT__: React,
  __HERMES_REACT_JSX__: jsxRuntime,
  __HERMES_REACT_JSX_DEV__: jsxDevRuntime
} as const

export function installPluginSdk(): void {
  Object.assign(globalThis, GLOBALS)
}

/** Build a shim ESM blob that re-exports a global namespace's live members.
 *  Export names come from the namespace itself, so the list can't drift.
 *
 *  Names must be valid binding identifiers — the /^[A-Za-z_$][\w$]*$/ shape
 *  alone is NOT enough: a minified SDK chunk can leak a reserved word as an
 *  export alias (proven: `in`), and `export const { in } = m` is a SyntaxError
 *  that Chromium surfaces as a confusing "Unexpected token ')'" at the chunk
 *  line. Filter reserved words too. */
const RESERVED_WORDS = new Set([
  'break','case','catch','class','const','continue','debugger','default','delete',
  'do','else','enum','export','extends','false','finally','for','function','if',
  'import','in','instanceof','new','null','return','super','switch','this','throw',
  'true','try','typeof','var','void','while','with','yield','let','static','await',
  'implements','package','protected','interface','private','public'
])

function shimUrl(globalKey: keyof typeof GLOBALS): string {
  // The shim's own local binding must not collide with an export name: a
  // minified SDK chunk can export ANY valid identifier, including the local
  // name the shim uses for the namespace (`const m = ...; export const { m }
  // = m;` is `Identifier 'm' has already been declared` — proven: the
  // session-list-density chunk exports `m`). Use an unlikely internal name
  // AND exclude it from the destructure list so a future collision stays
  // impossible.
  const LOCAL = '__hermes_shim_ns__'
  const names = Object.keys(GLOBALS[globalKey]).filter(
    name =>
      name !== 'default' &&
      name !== LOCAL &&
      /^[A-Za-z_$][\w$]*$/.test(name) &&
      !RESERVED_WORDS.has(name)
  )

  const source =
    `const ${LOCAL} = globalThis.${globalKey};\n` +
    `export default ${LOCAL}.default ?? ${LOCAL};\n` +
    // Guard the destructuring: `export const {  } = m` is a syntax error, so
    // only emit it when the namespace actually has named exports.
    (names.length ? `export const { ${names.join(', ')} } = ${LOCAL};\n` : '')

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
