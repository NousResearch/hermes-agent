import { isBrowserHostedDesktop } from './platform'

const publishableKey = 'pk_live_Y2xlcmsudW5pdGFsay5haSQ'
const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop?.clerk

export const clerkPublishableKey = publishableKey
export const clerkEnabled = Boolean(publishableKey && bridge?.enabled)
export const clerkBrowserEnabled = Boolean(publishableKey && !bridge && isBrowserHostedDesktop())

if (bridge?.enabled && !publishableKey) {
  console.warn('[clerk] renderer and Electron bridge Clerk configuration do not match')
}
