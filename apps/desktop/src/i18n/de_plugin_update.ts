import type { PluginUpdateTranslations } from './types_plugin_update'

/** Copy for updating an installed plugin (Plugins tab): catalog re-pin consent and the security-scan override. */
export const dePluginUpdate: PluginUpdateTranslations = {
  updateToPin: sha => `Auf ${sha} aktualisieren`,
  updateFailed: name => `${name} konnte nicht aktualisiert werden`,
  updated: name =>
    `${name} wurde auf den aktuellen Katalog-Pin aktualisiert. Starten Sie das Gateway neu, damit es wirksam wird.`,
  updateConsentTitle: (name: string) => `${name} fordert mehr an`,
  updateConsentBody: (name: string, sha: string) =>
    `Der neue Katalog-Pin von ${name} (${sha}) fügt Oberflächen hinzu, die die installierte Version nicht hat. Wenden Sie ihn nur an, wenn Sie ihnen vertrauen:`,
  updateConsentConfirm: 'Update anwenden',
  updateScanTitle: name => `Sicherheitsprüfung hat ${name} markiert`,
  updateScanBody: (name, count) =>
    `Das Update von ${name} erhielt das Urteil „Vorsicht“ (${count} ${count === 1 ? 'Fund' : 'Funde'}). Jedes Update wird geprüft. Wenden Sie es nur an, wenn Sie der Quelle vertrauen:`,
  updateScanMore: count => `+ ${count} weitere`,
  updateScanConfirm: 'Trotzdem aktualisieren',
  updateScanBlocked: name =>
    `Die Sicherheitsprüfung hat das Update von ${name} blockiert (Urteil „gefährlich“). Das lässt sich nicht übergehen.`,
  updatedGit: name => `${name} wurde auf den neuesten Commit aktualisiert.`
}
