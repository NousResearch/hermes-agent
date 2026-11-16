import type { UpdateChannelCopy } from './types_update_channel'

export const svUpdateChannel: UpdateChannelCopy = {
  title: 'Uppdateringskanal',
  description: 'Stabil installerar bara publicerade utgåvor. Varje incheckning följer grenen main när ändringar landar.',
  stable: 'Stabila utgåvor',
  main: 'Varje incheckning',
  stableConfirmTitle: 'Byta till stabila utgåvor?',
  stableConfirmBody:
    'Hermes flyttas till den senaste publicerade utgåvan, som kan vara äldre än koden du kör nu. Data som nyare kod har skrivit kanske inte går att läsa korrekt, så säkerhetskopiera först.',
  stableConfirm: 'Byt till stabil',
  failed: 'Kunde inte ändra uppdateringskanalen',
  detailsLabel: 'Kanal',
  branch: branch => `Gren: ${branch}`,
  change: 'Ändra',
  latestRelease: version => `Du använder den senaste stabila utgåvan (${version}).`,
  rollbackTitle: version => `Tillbaka till stabil utgåva ${version}`,
  rollbackBody: version =>
    `Den här versionen är nyare än den stabila utgåvan ${version}. Vid installation går Hermes tillbaka till den utgåvan.`,
  heldBody: version =>
    `Den här versionen är nyare än den stabila utgåvan ${version}. Dina sessioner använder redan en gatewaykörmiljö som ${version} inte kan öppna, så Hermes ligger kvar på den här versionen tills en nyare stabil utgåva släpps.`,
  rollbackAction: version => `Återställ till ${version}`,
  allChanges: 'Visa alla ändringar'
}
