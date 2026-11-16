const publishableKey = 'VITE_CLERK_PUBLISHABLE_KEY'
const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop?.clerk

export const clerkPublishableKey = publishableKey
export const clerkEnabled = Boolean(publishableKey && bridge?.enabled)

if (bridge?.enabled && !publishableKey) {
  console.warn('[clerk] renderer and Electron bridge Clerk configuration do not match')
}
