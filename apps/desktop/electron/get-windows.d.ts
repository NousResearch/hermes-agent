// Type declarations for the get-windows optionalDependency.
//
// get-windows ships no bundled types and is an optionalDependency. `npm ci`
// can skip it when its native install fails, including Linux and Windows ARM64
// where 9.3.0 has no prebuilt, so it can legitimately be absent from
// node_modules. Declaring the module here keeps typecheck independent of
// whether the package installed. The runtime import in window-below.ts
// degrades to null when it is absent.

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
