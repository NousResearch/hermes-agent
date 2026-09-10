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

type GlobalKey =
  | '__HERMES_PLUGIN_SDK__'
  | '__HERMES_REACT__'
  | '__HERMES_REACT_JSX__'
  | '__HERMES_REACT_JSX_DEV__'

// NOTE: resolve namespaces at call time, never at module scope. The bundler
// may evaluate this module before the sdk barrel's namespace binding is
// initialized (a hoisted `var` still reads as undefined), which silently
// empties ({...undefined} === {}) or crashes (Object.keys(undefined)) every
// runtime plugin load. Both consumers run long after boot, so laziness is
// free and always safe.
function resolveNamespace(key: GlobalKey): Record<string, unknown> | null | undefined {
  switch (key) {
    case '__HERMES_PLUGIN_SDK__':
      // Spread into a plain object: the loader re-exports these members
      // dynamically (Object.keys), invisible to tree-shaking, so a static
      // use of every member keeps plugin-facing exports in the bundle.
      return { ...sdk }
    case '__HERMES_REACT__':
      return React as unknown as Record<string, unknown>
    case '__HERMES_REACT_JSX__':
      return jsxRuntime as unknown as Record<string, unknown>
    case '__HERMES_REACT_JSX_DEV__':
      return jsxDevRuntime as unknown as Record<string, unknown> | null | undefined
  }
}

export function installPluginSdk(): void {
  Object.assign(globalThis, {
    __HERMES_PLUGIN_SDK__: resolveNamespace('__HERMES_PLUGIN_SDK__'),
    __HERMES_REACT__: resolveNamespace('__HERMES_REACT__'),
    __HERMES_REACT_JSX__: resolveNamespace('__HERMES_REACT_JSX__'),
    __HERMES_REACT_JSX_DEV__: resolveNamespace('__HERMES_REACT_JSX_DEV__')
  })
}

/** Build a shim ESM blob that re-exports a global namespace's live members.
 *  Export names come from the namespace itself, so the list can't drift. */
function shimUrl(globalKey: GlobalKey): string {
  const ns = resolveNamespace(globalKey)

  // A nullish namespace (e.g. a bundler-mangled react/jsx-dev-runtime) must
  // not break every plugin load: emit a module that throws only if imported.
  if (ns == null) {
    return URL.createObjectURL(
      new Blob([`throw new Error('unavailable runtime namespace: ${globalKey}')`], {
        type: 'text/javascript'
      })
    )
  }

  const names = Object.keys(ns).filter(name => name !== 'default' && /^[A-Za-z_$][\w$]*$/.test(name))

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
