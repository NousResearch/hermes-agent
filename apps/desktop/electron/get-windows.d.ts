// Type declarations for the get-windows optionalDependency.
//
// get-windows ships no bundled types and is an optionalDependency: on Linux
// `npm ci` skips it when its node-pre-gyp install script fails (no Linux
// prebuilt; the node-gyp fallback needs `gyp` in the active Python), so the
// package is legitimately absent from node_modules on Linux builds. Declaring
// the module here keeps the typecheck independent of whether the package
// installed — the runtime import in window-below.ts degrades to null when it
// is absent.

declare module 'get-windows' {
  export interface GetWindowsWindow {
    bounds?: { height?: number; width?: number; x?: number; y?: number }
    id?: number
    owner?: { name?: string; processId?: number }
    title?: string
  }

  export function openWindows(options?: {
    accessibilityPermission?: boolean
    screenRecordingPermission?: boolean
  }): Promise<GetWindowsWindow[]>
}
