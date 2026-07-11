interface RemoteConnectionGate {
  hasWaiter: () => boolean
  resume: () => void
  wait: () => Promise<void>
}

export function assertConnectionModeAllowed(mode: unknown, remoteOnly: boolean): void {
  if (remoteOnly && mode !== 'remote' && mode !== 'cloud') {
    throw new Error('This Hermes Desktop build requires a remote Hermes connection.')
  }
}

export function shouldResumeRemoteConnectionGate(remoteOnly: boolean, profile: null | string, waiting: boolean): boolean {
  return remoteOnly && !profile && waiting
}

export function createRemoteConnectionGate(): RemoteConnectionGate {
  let waiter: { promise: Promise<void>; resolve: () => void } | null = null

  return {
    hasWaiter: () => Boolean(waiter),
    resume: () => {
      const active = waiter

      waiter = null
      active?.resolve()
    },
    wait: () => {
      if (waiter) {
        return waiter.promise
      }

      let resolveWaiter: () => void = () => {}

      const promise = new Promise<void>(resolve => {
        resolveWaiter = resolve
      })

      waiter = { promise, resolve: resolveWaiter }

      return promise
    }
  }
}
