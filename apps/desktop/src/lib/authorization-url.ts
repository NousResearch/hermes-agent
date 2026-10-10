export interface AuthorizationUrlPrompt {
  url: string
}

let current: AuthorizationUrlPrompt | null = null
const listeners = new Set<(value: AuthorizationUrlPrompt | null) => void>()

function publish() {
  for (const listener of listeners) {
    listener(current)
  }
}

/** Hand the user an authorization URL. Does not open a browser. */
export function presentAuthorizationUrl(url: string) {
  current = { url }
  publish()
}

export function dismissAuthorizationUrl() {
  if (!current) {
    return
  }

  current = null
  publish()
}

export function authorizationUrlNow(): string | null {
  return current?.url ?? null
}

export function subscribeAuthorizationUrl(listener: (value: AuthorizationUrlPrompt | null) => void) {
  listeners.add(listener)
  listener(current)

  return () => {
    listeners.delete(listener)
  }
}
