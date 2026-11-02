import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into en.ts.
export const enGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Gateway settings unavailable',
  unavailableDesc: 'Connection settings can only be changed from the Hermes Desktop app on the computer running it.',
  webappHostTitle: 'Hermes host',
  webappHostDesc:
    'The Webapp always uses the Hermes host that serves it. To switch gateways, sign in to Hermes Cloud or manage saved connections, use the Hermes Desktop app.',
  webAccess: {
    title: 'Web access',
    intro:
      'Open Hermes on this computer from your phone or another computer, signed in with your Nous account. Whoever signs in can run commands on this computer.',
    failed: "Couldn't change web access",
    running: 'Running',
    stopped: 'Stopped',
    starting: 'Starting',
    stopping: 'Stopping',
    restartNeeded: 'Restart needed',
    unavailable: 'Unavailable',
    notRunningDetail: 'Start it to open Hermes from another device.',
    needsNous: 'Sign in to your Nous account under Providers first.',
    runningLan: 'Open this address on a device on the same network.',
    runningPublic: 'Served through your URL.',
    runningLocal: 'Only this computer can open it. Stop it, then start it again to open it from other devices.',
    pendingOn: 'Saved. The gateway serves the web app once it restarts.',
    pendingOff: 'Turned off, but the gateway keeps serving the web app until it restarts.',
    unavailableDetail: reason => `The gateway couldn't serve the web app: ${reason}`,
    restartBusy: "Saved. A chat is running, so the gateway didn't restart. Choose Restart gateway once it ends.",
    restartFailed: "Saved, but the gateway couldn't restart. Choose Restart gateway to try again.",
    restartFailedBecause: reason => `The gateway couldn't restart: ${reason}`,
    startTimedOut: "The gateway didn't serve the web app within 90 seconds.",
    stopTimedOut: "The gateway didn't stop serving the web app within 90 seconds.",
    start: 'Start',
    stop: 'Stop',
    restartGateway: 'Restart gateway',
    modeLan: 'This network',
    modeLanDetail: url => `Devices on your Wi-Fi open ${url}. Use it only on a network you trust.`,
    lanUnencrypted: 'The connection is not encrypted: others on this network could read what you send and receive.',
    noLan: 'This computer has no network address other devices can reach.',
    modePublic: 'My own URL',
    modePublicDetail: port =>
      `An HTTPS address you already run, such as Tailscale Serve or a tunnel, forwarding to http://127.0.0.1:${port}.`
  }
}
