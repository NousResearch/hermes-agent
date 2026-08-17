const FULL_GIT_SHA = /^[0-9a-f]{40}$/i

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
  const validCommit = FULL_GIT_SHA.test(installCommit || '') && !/^0{40}$/.test(installCommit || '')
  const buildVersion = validCommit ? `${installCommit!.slice(0, 7)}${installDirty ? '-dirty' : ''}` : null

  return {
    applicationVersion,
    ...(buildVersion ? { version: buildVersion } : {}),
    ...(bundleOutOfSync ? { credits: 'App build out of date. Update the desktop app.' } : {})
  }
}
