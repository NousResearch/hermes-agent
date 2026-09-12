import { type BrowserWindow, ipcMain, screen } from 'electron'

import { centeredBounds, type GrowRequest, growWindowBounds } from './window-growth'

interface ChatOnboardingWindowOptions {
  enabled: boolean
  mainWindow: () => BrowserWindow | null
}

export function registerChatOnboardingWindow({ enabled, mainWindow }: ChatOnboardingWindowOptions): void {
  ipcMain.on('hermes:chat-onboarding:grow', (event, request: GrowRequest) => {
    const win = mainWindow()

    if (!enabled || !win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    // The request arrives in renderer CSS pixels; growWindowBounds converts it to DIP with the zoom factor below.
    const bounds = win.getBounds()

    win.setBounds(
      growWindowBounds(request, {
        bounds,
        frameWidth: bounds.width - win.getContentBounds().width,
        workArea: screen.getDisplayMatching(bounds).workArea,
        zoom: event.sender.getZoomFactor() || 1
      }),
      true
    )
  })

  ipcMain.on('hermes:chat-onboarding:solo-boot', event => {
    const win = mainWindow()

    if (!enabled || !win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    const area = screen.getDisplayMatching(win.getBounds()).workArea
    const width = Math.min(600, area.width)
    const height = Math.min(640, area.height)

    win.setBounds(centeredBounds(area, width, height), true)
  })
}
