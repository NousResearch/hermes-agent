/**
 * Host-SDK bridge.
 *
 * Mission Control runs in two hosts, and both must supply React:
 *
 *   1. The Hermes dashboard (`web/src/plugins/registry.ts`) exposes React on
 *      `window.__HERMES_PLUGIN_SDK__` and renders the component we register.
 *   2. The standalone server (`samagent/ui_server.py`) loads a real React host
 *      bundle that exposes the same globals and mounts the component itself.
 *
 * So React is ALWAYS an external — never bundled. Bundling our own copy would
 * give us a second React instance, and hooks called inside a component created
 * by React A but rendered by React B throw on the first render.
 *
 * esbuild maps this module's imports onto these globals (see
 * `scripts/build-mission-control.mjs`), which is what keeps the dashboard path
 * sharing the dashboard's single React.
 */
const sdk = (typeof window !== 'undefined' && window.__HERMES_PLUGIN_SDK__) || {};

function required(name) {
  const fn = sdk.hooks ? sdk.hooks[name] : undefined;
  if (!fn) {
    throw new Error(
      `SamAgent Mission Control: host did not provide React hook "${name}". ` +
        'The standalone host bundle (dist/host.js) must load before dist/index.js.'
    );
  }
  return fn;
}

export const React = sdk.React;
export const createElement = sdk.React.createElement;

export const useState = required('useState');
export const useEffect = required('useEffect');
export const useCallback = required('useCallback');
export const useMemo = sdk.hooks && sdk.hooks.useMemo ? sdk.hooks.useMemo : required('useState');
export const useRef = sdk.hooks && sdk.hooks.useRef ? sdk.hooks.useRef : required('useCallback');

/** Register the root component with whichever host is running. */
export function registerRoot(Component) {
  const registry = (typeof window !== 'undefined' && window.__HERMES_PLUGINS__) || null;
  if (!registry || typeof registry.register !== 'function') {
    throw new Error('SamAgent Mission Control: no plugin registry on the host.');
  }
  registry.register('samagent', Component);
}
