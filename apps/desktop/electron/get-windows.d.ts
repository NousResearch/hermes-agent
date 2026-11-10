// Type declarations for get-windows.
//
// get-windows ships no bundled types. An interrupted install can still leave it
// absent or half-extracted, so declaring the module here keeps typecheck
// independent of whether the package installed. The runtime import in
// window-below.ts degrades to a failure note when it is absent.

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
