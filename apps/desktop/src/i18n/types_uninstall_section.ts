// Settings > About > Uninstall; `Translations.settings.uninstallSection`.
export interface UninstallSectionTranslations {
  dangerZone: string
  checkingInstalled: string
  uninstallHermes: string
  managedBody: string
  dataKept: (path: string) => string
  openAppsSettings: string
  chooseHowMuch: string
  confirmUninstall: string
  confirmBody: (what: string) => string
  appLabel: string
  couldNotStart: string
  uninstalling: string
  yesUninstall: string
  options: {
    gui: { title: string; description: string; consequence: string }
    lite: { title: string; description: string; consequence: string }
    full: { title: string; description: string; consequence: string }
  }
}
