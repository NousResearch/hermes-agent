import { isValidInstallCommit } from './install-stamp-identity'

export interface AboutPanelVersionInput {
  applicationVersion: string
  installCommit?: string | null
  installDirty?: boolean
  bundleOutOfSync?: boolean
}

export function buildAboutPanelVersionOptions({
  applicationVersion,
  installCommit,
  installDirty = false,
  bundleOutOfSync = false
}: AboutPanelVersionInput) {
  const validCommit = isValidInstallCommit(installCommit)
  const buildVersion = validCommit ? `${installCommit!.slice(0, 7)}${installDirty ? '-dirty' : ''}` : null

  return {
    applicationVersion,
    ...(buildVersion ? { version: buildVersion } : {}),
    ...(bundleOutOfSync ? { credits: 'App build out of date. Update the desktop app.' } : {})
  }
}
