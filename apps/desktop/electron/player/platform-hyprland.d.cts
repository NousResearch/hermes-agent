declare const api: {
  PLAYER_TITLE: string
  applyHyprlandPlayerPolicy(options: { enabled: boolean; pid?: number; title?: string }): Promise<Record<string, any>>
  observeHyprlandPlayer(options?: { pid?: number; title?: string }): Promise<Record<string, any>>
  setHyprlandPlayerFullscreen(options: {
    enabled: boolean
    pid?: number
    title?: string
    desiredPinned?: boolean | (() => boolean)
    timeoutMs?: number
  }): Promise<Record<string, any>>
  setHyprlandPlayerPin(options: { enabled: boolean; pid?: number; title?: string }): Promise<Record<string, any>>
}

export = api
