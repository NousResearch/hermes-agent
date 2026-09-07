export const PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR = 'net::ERR_BLOCKED_BY_CLIENT'

export interface PreviewUblockBlockRequestDetails {
  error?: string
  id?: number
  webContentsId?: number
}

export interface PreviewUblockBlockCounterGuest {
  on(event: 'destroyed' | 'did-navigate', listener: () => void): unknown
  removeListener?(event: 'destroyed' | 'did-navigate', listener: () => void): unknown
  reload?(): unknown
}

export interface PreviewUblockBlockCounterSession {
  webRequest: {
    onCompleted(listener: (details: PreviewUblockBlockRequestDetails) => void): unknown
    onErrorOccurred(listener: (details: PreviewUblockBlockRequestDetails) => void): unknown
    removeCompletedListener?(listener: (details: PreviewUblockBlockRequestDetails) => void): unknown
    removeListener?(listener: (details: PreviewUblockBlockRequestDetails) => void): unknown
  }
}

export interface PreviewUblockBlockedRequestCountUpdate {
  blockedRequestCount: number
  ownerWebContentsId: number
  webContentsId: number
}

export interface PreviewUblockBlockCounter {
  dispose(): void
  getCount(webContentsId: number): number
  recordBlockedRequest(webContentsId: number, requestId?: number): void
  registerGuest(webContentsId: number, ownerWebContentsId: number, guest: PreviewUblockBlockCounterGuest): boolean
  setActive(active: boolean): void
  unregisterGuest(webContentsId: number): void
}

interface RegisteredGuest {
  blockedRequestCount: number
  guest: PreviewUblockBlockCounterGuest
  onDestroyed: () => void
  onNavigate: () => void
  ownerWebContentsId: number
  webContentsId: number
}

export function createPreviewUblockBlockCounter({
  onUpdate,
  session
}: {
  onUpdate: (update: PreviewUblockBlockedRequestCountUpdate) => void
  session: PreviewUblockBlockCounterSession
}): PreviewUblockBlockCounter {
  const guests = new Map<number, RegisteredGuest>()
  const observedRequestIds = new Set<number>()
  let active = false
  let disposed = false

  const publish = (registered: RegisteredGuest): void => {
    onUpdate({
      blockedRequestCount: registered.blockedRequestCount,
      ownerWebContentsId: registered.ownerWebContentsId,
      webContentsId: registered.webContentsId
    })
  }

  const clear = (registered: RegisteredGuest): void => {
    if (registered.blockedRequestCount === 0) {
      publish(registered)
      return
    }

    registered.blockedRequestCount = 0
    publish(registered)
  }

  const forget = (webContentsId: number, publishZero: boolean): void => {
    const registered = guests.get(webContentsId)

    if (!registered) {
      return
    }

    registered.guest.removeListener?.('did-navigate', registered.onNavigate)
    registered.guest.removeListener?.('destroyed', registered.onDestroyed)
    guests.delete(webContentsId)

    if (publishZero) {
      publish({ ...registered, blockedRequestCount: 0 })
    }
  }

  const recordBlockedRequest = (webContentsId: number, requestId?: number): void => {
    if (!active) {
      return
    }

    const registered = guests.get(webContentsId)

    if (!registered) {
      return
    }

    if (typeof requestId === 'number') {
      if (observedRequestIds.has(requestId)) {
        return
      }

      observedRequestIds.add(requestId)
      if (observedRequestIds.size > 10_000) {
        observedRequestIds.clear()
      }
    }

    registered.blockedRequestCount += 1
    publish(registered)
  }

  const onBlockedRequest = (details: PreviewUblockBlockRequestDetails): void => {
    if (details.error !== PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR || typeof details.webContentsId !== 'number') {
      return
    }

    recordBlockedRequest(details.webContentsId, details.id)
  }

  session.webRequest.onErrorOccurred(onBlockedRequest)
  session.webRequest.onCompleted(onBlockedRequest)

  return {
    dispose() {
      if (disposed) {
        return
      }

      disposed = true
      session.webRequest.removeListener?.(onBlockedRequest)
      session.webRequest.removeCompletedListener?.(onBlockedRequest)
      for (const webContentsId of guests.keys()) {
        forget(webContentsId, false)
      }
    },
    getCount(webContentsId) {
      return guests.get(webContentsId)?.blockedRequestCount ?? 0
    },
    recordBlockedRequest,
    registerGuest(webContentsId, ownerWebContentsId, guest) {
      if (disposed || !Number.isInteger(webContentsId) || !Number.isInteger(ownerWebContentsId)) {
        return false
      }

      forget(webContentsId, false)
      const registered: RegisteredGuest = {
        blockedRequestCount: 0,
        guest,
        onDestroyed: () => forget(webContentsId, true),
        onNavigate: () => clear(registered),
        ownerWebContentsId,
        webContentsId
      }
      guests.set(webContentsId, registered)
      guest.on('did-navigate', registered.onNavigate)
      guest.on('destroyed', registered.onDestroyed)
      publish(registered)

      return true
    },
    setActive(nextActive) {
      const becameActive = !active && nextActive
      active = nextActive

      if (becameActive) {
        for (const registered of guests.values()) {
          try {
            registered.guest.reload?.()
          } catch {
            // A guest can disappear between activation and reload.
          }
        }
      }

      if (!active) {
        for (const registered of guests.values()) {
          clear(registered)
        }
      }
    },
    unregisterGuest(webContentsId) {
      forget(webContentsId, true)
    }
  }
}
