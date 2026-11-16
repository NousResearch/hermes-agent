export const CLERK_RENDERER_SCHEME = 'unitalk-app'
export const CLERK_DEV_RENDERER_SCHEME = 'unitalk-app-dev'
export const CLERK_RENDERER_HOST = 'renderer'
export const CLERK_RENDERER_ORIGIN = `${CLERK_RENDERER_SCHEME}://${CLERK_RENDERER_HOST}/`
export const CLERK_TOKEN_STORAGE_NAME = 'unitalk-clerk-tokens'

export interface ClerkOAuthRenderer {
  host: string
  scheme: string
}

/**
 * `@clerk/electron` needs a renderer origin to build its OAuth redirect. Packaged
 * builds serve the app from `unitalk-app://renderer/`. Windows can register a
 * custom scheme for an unpackaged app at runtime, so dev gets a separate
 * `unitalk-app-dev://renderer/` (never steal the installed app's handler).
 * macOS and Linux Launch Services only route custom schemes to packaged bundles,
 * so dev keeps the transport disabled there.
 */
export function clerkOAuthRenderer({
  devServer,
  platform,
  registerProtocol
}: {
  devServer: boolean
  platform: NodeJS.Platform
  registerProtocol: boolean
}): ClerkOAuthRenderer | undefined {
  if (!registerProtocol) {
    return undefined
  }

  if (!devServer) {
    return { host: CLERK_RENDERER_HOST, scheme: CLERK_RENDERER_SCHEME }
  }

  return platform === 'win32' ? { host: CLERK_RENDERER_HOST, scheme: CLERK_DEV_RENDERER_SCHEME } : undefined
}

declare const __UNITALK_CLERK_PUBLISHABLE_KEY__: string | undefined

export function clerkPublishableKey(): string {
  return typeof __UNITALK_CLERK_PUBLISHABLE_KEY__ === 'string' ? __UNITALK_CLERK_PUBLISHABLE_KEY__.trim() : ''
}

export function clerkEnabled(): boolean {
  return clerkPublishableKey().length > 0
}
