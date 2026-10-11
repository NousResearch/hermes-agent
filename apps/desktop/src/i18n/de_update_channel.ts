export const deUpdateChannel = {
  title: 'Update-Kanal',
  description:
    'Stabil installiert nur veröffentlichte Releases. Jeder Commit folgt dem main-Branch, sobald Änderungen eintreffen.',
  stable: 'Stabile Releases',
  main: 'Jeder Commit',
  stableConfirmTitle: 'Zu stabilen Releases wechseln?',
  stableConfirmBody:
    'Hermes wechselt zum neuesten veröffentlichten Release, das älter sein kann als der Code, den du gerade nutzt. Von neuerem Code geschriebene Daten werden eventuell nicht korrekt gelesen – sichere sie vorher.',
  stableConfirm: 'Zu Stabil wechseln',
  failed: 'Update-Kanal konnte nicht geändert werden',
  detailsLabel: 'Kanal',
  branch: (b: string) => `Branch: ${b}`,
  change: 'Ändern',
  latestRelease: (v: string) => `Du nutzt das neueste stabile Release (${v}).`
}
