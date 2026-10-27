import { isBrowserHostedDesktop } from '@/lib/platform'

// A remote connection refuses /browser connect|disconnect: both drive the CDP
// connection inside the gateway process. The Webapp is always remote and has no
// local gateway to switch to, so it names the machine the command acts on instead.
export const remoteBrowserManageRefusal = (action: 'connect' | 'disconnect'): string =>
  isBrowserHostedDesktop()
    ? `/browser ${action} isn't available in the Webapp — it manages a Chromium-family browser on the machine running Hermes. Run it from Hermes Desktop or the CLI on that machine.`
    : '/browser connect manages a Chromium-family browser on the gateway host — only available when connected to a local gateway.'
