import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const enMemoryDiscovery: MemoryDiscoveryTranslations = {
  installed: 'Installed',
  availableToInstall: 'Available to install',
  installationRequired: 'Installation required',
  reviewInstall: 'Review & install',
  exploreAll: 'Explore all…',
  missing: 'Missing',
  installConsent:
    'Installs and enables the plugin with its dependencies. Your active memory provider stays unchanged until you choose Use provider.',
  builtin: 'Built-in',
  configureElsewhere: 'Configure this provider with its CLI setup, or update Hermes for save-only settings.',
  notReady:
    'Finish configuration and install missing dependencies. If just installed, restart the backend, then retry.',
  useFailed: 'Could not use this provider. Retry after checking its configuration.',

  activeProvider: name => `Active: ${name}`,
  useProvider: 'Use provider',
  loadFailed: 'Could not load memory providers',
  ownerChanged: 'Switch back to the connection and profile where you opened this installer, then try again.',
  notDiscovered:
    'The package was installed, but its memory provider is not discovered yet. Return to Memory settings to retry discovery.',
  installedNotice: 'Provider discovered. Configure it in Memory settings, then choose Use provider.',
  backToMemory: 'Back to Memory settings',
  connect: 'Connect',
  reconnect: 'Reconnect',
  connectOAuth: 'Connect via OAuth',
  apiKeySet: 'API key set',
  oauthSet: 'OAuth connected',
  waitingConsent: 'Waiting for browser consent…',
  stopWaiting: 'Stop waiting',
  stoppedWaiting: 'Stopped waiting. Authorization may still be pending.',
  startFailed: 'Could not start the connection.',
  connectionFailed: 'Connection failed.'
}
