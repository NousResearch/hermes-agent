import { Menu, nativeImage, Tray } from 'electron'

import { type BackgroundWorkItem, readBackgroundWork } from './background-work-status'

function trayIcon(busy: boolean) {
  const fill = busy ? '#ff9f0a' : '#7d7d7d'
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18"><circle cx="9" cy="9" r="6" fill="${fill}"/><circle cx="9" cy="9" r="3" fill="none" stroke="white" stroke-width="1.5"/></svg>`
  const image = nativeImage.createFromDataURL(`data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`)

  image.setTemplateImage(!busy)

  return image
}

function itemLabel(item: BackgroundWorkItem): string {
  const identity = [item.profile, item.model, item.worker].filter(Boolean).join(' · ')

  return `${item.title} — ${item.elapsed}${identity ? `\n${identity}` : ''}`
}

export interface BackgroundWorkTrayController {
  start(): void
  stop(): void
}

export function createBackgroundWorkTray(
  hermesHome: string,
  options: { platform?: NodeJS.Platform; showMainWindow?: () => void; pollMilliseconds?: number } = {}
): BackgroundWorkTrayController {
  const platform = options.platform ?? process.platform
  let tray: Tray | null = null
  let timer: NodeJS.Timeout | null = null
  let lastSignature = ''

  const render = () => {
    if (!tray) {
      return
    }

    const items = readBackgroundWork(hermesHome)
    const signature = JSON.stringify(items.map(item => [item.key, item.elapsed]))

    if (signature === lastSignature) {
      return
    }

    lastSignature = signature
    const busy = items.length > 0

    tray.setImage(trayIcon(busy))
    tray.setTitle(busy ? `${items.length}` : '')
    tray.setToolTip(busy ? `Hermes · ${items.length} background worker${items.length === 1 ? '' : 's'}` : 'Hermes · Idle')
    tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: busy ? `${items.length} active worker${items.length === 1 ? '' : 's'}` : 'Idle', enabled: false },
        { type: 'separator' },
        ...(busy
          ? items.slice(0, 10).map(item => ({ label: itemLabel(item), enabled: false }))
          : [{ label: 'No background AI workers', enabled: false }]),
        { type: 'separator' },
        { label: 'Open Hermes', click: () => options.showMainWindow?.() }
      ])
    )
  }

  return {
    start() {
      if (platform !== 'darwin' || tray) {
        return
      }

      tray = new Tray(trayIcon(false))
      render()
      timer = setInterval(render, options.pollMilliseconds ?? 500)
      timer.unref()
    },
    stop() {
      if (timer) {
        clearInterval(timer)
      }

      timer = null
      tray?.destroy()
      tray = null
      lastSignature = ''
    }
  }
}
