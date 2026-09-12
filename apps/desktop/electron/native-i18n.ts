// Native dialogs use the initiating window's UI language. Before its renderer
// hydrates, use the system locale; never borrow another window's preference.
const en = {
  pluginsInactive: 'Some plugins were not loaded',
  pluginsNeedUpdate: 'Plugins need an update',
  pluginRow: (name: string, count: number, oldPath: string, newPath: string) =>
    `• ${name} — ${count} import${count === 1 ? '' : 's'} (e.g. ${oldPath} → ${newPath})`,
  pluginMessage: (count: number, date: string, inactive: boolean) =>
    inactive
      ? `${count} plugin${count === 1 ? '' : 's'} import${count === 1 ? 's' : ''} module paths that were removed on ${date} and ${count === 1 ? 'was' : 'were'} not loaded.`
      : `${count} plugin${count === 1 ? '' : 's'} import${count === 1 ? 's' : ''} module paths that stop working on ${date}.`,
  pluginDetail: (list: string, date: string, inactive: boolean) =>
    inactive
      ? `${list}\n\nUpdate the plugin(s), or force-load them with plugins.allow_deprecated_imports: true in config.yaml (they will still break once the compatibility layer is removed).\n\nFull list: hermes plugins compat`
      : `${list}\n\nCheck for plugin updates or notify the author before ${date}. After that date these plugins are not loaded.\n\nFull list: hermes plugins compat`,
  saveImage: 'Save Image',
  images: 'Images',
  allFiles: 'All Files',
  saveFile: 'Save File',
  addContext: 'Add context',
  save: 'Save',
  chooseProject: 'Choose default project directory',
  update: 'Hermes update',
  updateNeedsStep: 'The update finished, but needs one more step',
  updateFailed: 'Hermes update did not finish',
  updateDetails: (message: string, path: string) => `${message}\n\nDetails: ${path}`,
  keepRunning: 'Keep Running',
  quitAnyway: 'Quit Anyway',
  quitMore: (count: number) => `• ${count} more`,
  quitWarning: 'Quitting stops the agent mid-turn. Any work it has not finished writing is lost.',
  quitMessage: (count: number) =>
    count === 1 ? 'Hermes is still working on 1 chat.' : `Hermes is still working on ${count} chats.`
}

export type NativeMessages = {
  [Key in keyof typeof en]: (typeof en)[Key] extends (...args: infer Args) => string
    ? (...args: Args) => string
    : string
}

const sv: NativeMessages = {
  pluginsInactive: 'Vissa pluginer lästes inte in',
  pluginsNeedUpdate: 'Pluginer behöver uppdateras',
  pluginRow: (name, count, oldPath, newPath) =>
    `• ${name} — ${count} ${count === 1 ? 'import' : 'importer'} (t.ex. ${oldPath} → ${newPath})`,
  pluginMessage: (count, date, inactive) =>
    inactive
      ? `${count} ${count === 1 ? 'plugin lästes inte in eftersom den importerar' : 'pluginer lästes inte in eftersom de importerar'} modulsökvägar som togs bort ${date}.`
      : `${count} ${count === 1 ? 'plugin importerar' : 'pluginer importerar'} modulsökvägar som slutar fungera ${date}.`,
  pluginDetail: (list, date, inactive) =>
    inactive
      ? `${list}\n\nUppdatera berörda pluginer eller tvinga inläsning med plugins.allow_deprecated_imports: true i config.yaml (de kommer ändå att sluta fungera när kompatibilitetslagret tas bort).\n\nFullständig lista: hermes plugins compat`
      : `${list}\n\nSök efter pluginuppdateringar eller meddela upphovspersonen före ${date}. Efter det datumet läses dessa pluginer inte in.\n\nFullständig lista: hermes plugins compat`,
  saveImage: 'Spara bild',
  images: 'Bilder',
  allFiles: 'Alla filer',
  saveFile: 'Spara fil',
  addContext: 'Lägg till kontext',
  save: 'Spara',
  chooseProject: 'Välj standardkatalog för projekt',
  update: 'Hermes-uppdatering',
  updateNeedsStep: 'Uppdateringen är klar, men ett steg återstår',
  updateFailed: 'Hermes-uppdateringen slutfördes inte',
  updateDetails: (message, path) => `${message}\n\nDetaljer: ${path}`,
  keepRunning: 'Fortsätt köra',
  quitAnyway: 'Avsluta ändå',
  quitMore: count => `• ${count} till`,
  quitWarning: 'Om du avslutar stoppas agenten mitt i turen. Arbete som inte har skrivits färdigt går förlorat.',
  quitMessage: count => `Hermes arbetar fortfarande i ${count} ${count === 1 ? 'chatt' : 'chattar'}.`
}

export function nativeMessages(locale: unknown): NativeMessages {
  const value = typeof locale === 'string' ? locale.trim().toLowerCase().replaceAll('_', '-') : ''

  return value.split('-')[0] === 'sv' || value === 'swedish' || value === 'svenska' ? sv : en
}

type LocaleSender = { id: number; once: (event: 'destroyed', listener: () => void) => unknown }
type LocaleIpc = {
  on: (channel: string, listener: (event: { sender: LocaleSender }, locale: unknown) => void) => unknown
}

export function registerNativeLocaleIpc(ipc: LocaleIpc) {
  const windows = new Map<number, NativeMessages>()
  ipc.on('hermes:ui-locale', (event, locale) => {
    const { sender } = event

    if (!windows.has(sender.id)) {
      sender.once('destroyed', () => windows.delete(sender.id))
    }

    windows.set(sender.id, nativeMessages(locale))
  })

  return (windowId: number | undefined, systemLocale: unknown): NativeMessages =>
    (windowId === undefined ? undefined : windows.get(windowId)) ?? nativeMessages(systemLocale)
}
