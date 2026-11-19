type ClerkTokenGetter = () => Promise<string | null>

let tokenGetter: ClerkTokenGetter | null = null

export function setClerkTokenGetter(getter: ClerkTokenGetter | null): void {
  tokenGetter = getter
}

export async function getClerkSessionToken(): Promise<string | null> {
  if (!tokenGetter) {
    throw new Error('Clerk session is not ready')
  }

  return tokenGetter()
}
