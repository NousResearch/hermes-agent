import { currentPluginSession } from '@/sdk/preview'

import { safeViewerUrl } from '../../electron/plugin-viewer-policy'

import type { PluginSessionContext } from './session'

export interface PluginViewerInput {
  id: string
  url: string
  title: string
  session: PluginSessionContext
}

export function createPluginViewerActions(pluginId: string, track: (dispose: () => void) => void) {
  let disposed = false
  let opened = false
  track(() => {
    disposed = true

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

      try {
        return await bridge.openPluginViewer(pluginId, { id: input.id, url: input.url, title: input.title })
      } catch {
        return false
      }
    },
    async closeViewer(id: string): Promise<boolean> {
      if (disposed) {
        return false
      }

      try {
        return (await window.hermesDesktop?.closePluginViewer?.(pluginId, id)) ?? false
      } catch {
        return false
      }
    }
  }
}
