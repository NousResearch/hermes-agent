import { currentPluginSession } from '@/sdk/preview'
import { startViewerKeepAlive } from '@/sdk/viewer-keep-alive'

import { safeViewerUrl } from '../../electron/plugin-viewer-policy'

import type { PluginSessionContext } from './session'

export interface PluginViewerInput {
  id: string
  url: string
  title: string
  session: PluginSessionContext
  /** Renew a scoped lease while this original native viewer/document remains open. Host renderer only. */
  onKeepAlive?: () => Promise<void>
}

export function createPluginViewerActions(pluginId: string, track: (dispose: () => void) => void) {
  let disposed = false
  let opened = false
  const viewers = new Map<string, { stop?: () => void }>()
  const retire = (id: string) => {
    viewers.get(id)?.stop?.()
    viewers.delete(id)
  }
  track(() => {
    disposed = true

    for (const id of viewers.keys()) {
      retire(id)
    }

    if (opened) {
      void window.hermesDesktop?.closePluginViewer?.(pluginId).catch(() => false)
    }
  })

  return {
    async openViewer(input: PluginViewerInput): Promise<boolean> {
      if (disposed || !safeViewerUrl(input?.url) || !currentPluginSession(input.session)) {
        return false
      }

      const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop

      if (!bridge?.openPluginViewer) {
        return false
      }

      opened = true
      const { id, url, title, onKeepAlive } = input
      retire(id)
      const entry: { stop?: () => void } = {}
      viewers.set(id, entry)

      try {
        const accepted = await bridge.openPluginViewer(pluginId, { id, url, title })

        if (disposed || viewers.get(id) !== entry) {
          return false
        }

        if (!accepted) {
          retire(id)

          return false
        }

        if (onKeepAlive && bridge.isPluginViewerOpen) {
          entry.stop = startViewerKeepAlive({
            isOpen: () => bridge.isPluginViewerOpen!(pluginId, id, url),
            onKeepAlive,
            onStop: () => {
              if (viewers.get(id) === entry) {
                viewers.delete(id)
              }
            }
          })
        }

        return true
      } catch {
        if (viewers.get(id) === entry) {
          retire(id)
        }

        return false
      }
    },
    async closeViewer(id: string): Promise<boolean> {
      if (disposed) {
        return false
      }

      retire(id)

      try {
        return (await window.hermesDesktop?.closePluginViewer?.(pluginId, id)) ?? false
      } catch {
        return false
      }
    }
  }
}
