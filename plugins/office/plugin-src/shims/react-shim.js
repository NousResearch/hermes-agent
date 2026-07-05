/**
 * React shim — pulls React from the dashboard's plugin SDK global
 * instead of bundling a separate copy. This prevents "Invalid hook call"
 * caused by duplicate React instances.
 *
 * react-dom is NOT shimmed — it's bundled inline by esbuild because R3F's
 * Canvas needs react-dom/client's createRoot to manage its own React root
 * inside the WebGL canvas, and the dashboard SDK doesn't expose ReactDOM.
 */
const SDK = window.__HERMES_PLUGIN_SDK__;
const React = SDK.React;

// Export everything React exports (for R3F/drei which import from 'react')
export default React;
export const {
  createElement,
  Fragment,
  Component,
  PureComponent,
  memo,
  forwardRef,
  useRef,
  useState,
  useEffect,
  useLayoutEffect,
  useCallback,
  useMemo,
  useReducer,
  useContext,
  useImperativeHandle,
  useDebugValue,
  useId,
  useDeferredValue,
  useTransition,
  useSyncExternalStore,
  useInsertionEffect,
  // React 19 API — its-fine's useContextMap calls React.use() to consume contexts
  use,
  unstable_use,
  cloneElement,
  isValidElement,
  createContext,
  createFactory,
  createRef,
  lazy,
  Suspense,
  StrictMode,
  startTransition,
  Children,
  version,
} = React;

// jsx-runtime exports — used by the automatic JSX runtime
export const jsx = React.createElement;
export const jsxs = React.createElement;
