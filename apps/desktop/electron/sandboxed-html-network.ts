import type { WebFrameMain } from 'electron'

export const SANDBOXED_HTML_FRAME_PREFIX = 'hermes-sandboxed-html:'

interface FrameRegistration {
  frameTreeNodeId: number
  ownerWebContentsId: number
}

export interface SandboxedHtmlNetworkGuard {
  register: (ownerWebContentsId: number, senderFrame: WebFrameMain, frameName: string) => boolean
  shouldBlock: (frame?: null | WebFrameMain, url?: string) => boolean
  unregister: (ownerWebContentsId: number, frameName: string) => boolean
  unregisterOwner: (ownerWebContentsId: number) => void
}

function validFrameName(frameName: string): boolean {
  return (
    frameName.startsWith(SANDBOXED_HTML_FRAME_PREFIX) &&
    frameName.length > SANDBOXED_HTML_FRAME_PREFIX.length &&
    frameName.length <= 160
  )
}

function isInlineUrl(url: string): boolean {
  return url.startsWith('about:') || url.startsWith('blob:') || url.startsWith('data:')
}

/**
 * Tracks the browser-global frame identity of approved HTML preview iframes.
 * The frame name is used only while the trusted bootstrap document is loaded;
 * network decisions use the immutable FrameTreeNode id after registration.
 */
export function createSandboxedHtmlNetworkGuard(): SandboxedHtmlNetworkGuard {
  const registrations = new Map<string, FrameRegistration>()
  const protectedFrameIds = new Set<number>()

  return {
    register(ownerWebContentsId, senderFrame, frameName) {
      if (!validFrameName(frameName)) {
        return false
      }

      const child = senderFrame.frames.find(frame => !frame.detached && frame.name === frameName)

      if (!child) {
        return false
      }

      const existing = registrations.get(frameName)

      if (existing && existing.ownerWebContentsId !== ownerWebContentsId) {
        return false
      }

      if (existing) {
        protectedFrameIds.delete(existing.frameTreeNodeId)
      }

      registrations.set(frameName, {
        frameTreeNodeId: child.frameTreeNodeId,
        ownerWebContentsId
      })
      protectedFrameIds.add(child.frameTreeNodeId)

      return true
    },

    shouldBlock(frame, url = '') {
      return Boolean(frame && protectedFrameIds.has(frame.frameTreeNodeId) && !isInlineUrl(url))
    },

    unregister(ownerWebContentsId, frameName) {
      const registration = registrations.get(frameName)

      if (!registration || registration.ownerWebContentsId !== ownerWebContentsId) {
        return false
      }

      registrations.delete(frameName)
      protectedFrameIds.delete(registration.frameTreeNodeId)

      return true
    },

    unregisterOwner(ownerWebContentsId) {
      for (const [frameName, registration] of registrations) {
        if (registration.ownerWebContentsId !== ownerWebContentsId) {
          continue
        }

        registrations.delete(frameName)
        protectedFrameIds.delete(registration.frameTreeNodeId)
      }
    }
  }
}
