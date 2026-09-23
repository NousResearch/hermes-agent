/// <reference types="vite/client" />

// @novnc/novnc ships no typings (its export is core/rfb.js); the surface the Bot Screen pane uses.
// Declared here because `apps/desktop/src/**/*.d.ts` is gitignored except for the allowlisted files,
// and an ambient `declare module` only works in a script-scoped (import-free) declaration file.
declare module '@novnc/novnc' {
  export default class RFB {
    constructor(target: HTMLElement, urlOrChannel: string | WebSocket | RTCDataChannel, options?: Record<string, unknown>)
    viewOnly: boolean
    scaleViewport: boolean
    resizeSession: boolean
    focusOnClick: boolean
    background: string
    qualityLevel: number
    compressionLevel: number
    addEventListener(type: string, listener: (event: CustomEvent) => void): void
    removeEventListener(type: string, listener: (event: CustomEvent) => void): void
    disconnect(): void
    focus(): void
    blur(): void
    clipboardPasteFrom(text: string): void
  }
}
