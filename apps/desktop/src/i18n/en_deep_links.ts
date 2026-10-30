import type { GatewayGroupOpenCopy, SkillDeepLinkCopy } from './types_deep_links'

export const enSkillDeepLink: SkillDeepLinkCopy = {
  installTitle: (name: string) => `Install “${name}”?`,
  installDescription: 'This skill will be available in new sessions. Only install sources you trust.',
  installTo: 'Install to',
  thisComputer: 'This computer',
  installing: 'Installing…',
  installComplete: (name: string) => `“${name}” installed`,
  destinationChanged: 'The destination changed. Close this dialog and open the install link again.',
  installed: 'Installed',
  source: 'Source'
}

export const enGatewayGroupOpen: GatewayGroupOpenCopy = {
  opening: 'Opening gateway group…',
  lookupFailed: 'Could not look up this group on the gateway. Try again.',
  missing: 'This gateway group no longer exists.'
}
