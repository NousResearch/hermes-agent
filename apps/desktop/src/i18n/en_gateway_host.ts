import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Machines copy where the connection cannot change, spread into en.ts.
export const enGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Machine settings unavailable',
  unavailableDesc: 'Connection settings can only be changed from the Hermes Desktop app on the computer running it.',
  webappHostTitle: 'Hermes host',
  webappHostDesc:
    "The Webapp always runs on the Hermes host that serves it. The saved connections below are this machine's list, shared with the Hermes Desktop app; other machines are reached through this host.",
  webappOauthPaste: {
    title: 'Finish signing in',
    description:
      'If the sign-in tab ended on an address that would not load (this browser is not on the machine running Hermes), paste that address here.',
    placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…',
    finish: 'Finish',
    failed: 'Could not finish signing in'
  },
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
    pendingOn: 'Saved. The web app will be available after Hermes restarts.',
    pendingOff: 'Turned off. The web app stays available until Hermes restarts.',
    unavailableDetail: reason => `Hermes couldn't start the web app: ${reason}`,
    restartBusy: "Saved. A chat is running, so Hermes didn't restart. Choose Restart Hermes once it ends.",
    restartFailed: "Saved, but Hermes couldn't restart. Choose Restart Hermes to try again.",
    restartFailedBecause: reason => `Hermes couldn't restart: ${reason}`,
    startTimedOut: "Hermes didn't start the web app within 90 seconds.",
    stopTimedOut: "Hermes didn't stop the web app within 90 seconds.",
    startStages: {
      preparing: 'Getting the web app ready. Building it can take a few minutes.',
      building: 'Building the web app. This can take a few minutes.',
      registering: 'Setting up Nous sign-in for this address.',
      restarting: 'Restarting Hermes to start the web app.'
    },
    startElapsed: time => `${time} elapsed`,
    start: 'Start',
    stop: 'Stop',
    restartGateway: 'Restart Hermes',
    modeLan: 'This network',
    modeLanDetail: url => `Devices on your Wi-Fi open ${url}. Use it only on a network you trust.`,
    lanUnencrypted: 'The connection is not encrypted: others on this network could read what you send and receive.',
    noLan: 'This computer has no network address other devices can reach.',
    modePublic: 'My own URL',
    modePublicDetail: port =>
      `An HTTPS address you already run, such as Tailscale Serve or a tunnel, forwarding to http://127.0.0.1:${port}.`
  }
}
