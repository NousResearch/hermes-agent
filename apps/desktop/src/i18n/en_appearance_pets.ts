import type { AppearancePetsCopy } from './types_appearance_pets'

export const enAppearancePets: AppearancePetsCopy = {
  title: 'Pet',
  intro:
    'Adopt an animated petdex mascot that floats over the app and reacts to what Hermes is doing — running while tools execute, celebrating on success, sulking on errors.',
  restartHint:
    'Pets need a quick restart — the running app started before this feature was added. Quit and reopen Hermes, then come back here.',
  scaleTitle: 'Size',
  scaleDesc: 'Resize the floating mascot. Applies everywhere instantly.',
  roamTitle: 'Roam',
  roamDesc: 'Let the pet wander the window on its own while idle.',
  chooseTitle: 'Choose a pet',
  chooseDesc: 'Picking one installs it (if needed) and makes it active.',
  searchPlaceholder: 'Search pets…',
  unreachable: "Couldn't reach the petdex gallery. Check your connection and reopen this page.",
  noMatch: query => `No pets match "${query}".`,
  installedTag: 'installed',
  generatedTag: 'Generated',
  countCapped: (cap, total) => `Showing ${cap} of ${total} — type to narrow it down.`,
  count: n => `${n} pet${n === 1 ? '' : 's'}.`,
  uninstall: name => `Uninstall ${name}`,
  delete: name => `Delete ${name}`,
  deleteTitle: name => `Delete ${name}?`,
  deleteBody: "This permanently deletes the pet — it can't be reinstalled.",
  deleteConfirm: 'Delete',
  rename: name => `Rename ${name}`,
  renameTitle: 'Rename pet',
  renamePlaceholder: 'Name your pet',
  renameSave: 'Save',
  exportPet: name => `Export ${name}`,
  adoptFailed: slug => `Could not adopt ${slug}`,
  uninstallFailed: slug => `Could not uninstall ${slug}`,
  renameFailed: slug => `Could not rename ${slug}`,
  exportFailed: slug => `Could not export ${slug}`,
  noneAvailable: 'No pets available to turn on right now.',
  turnOnFailed: 'Could not turn the pet on.',
  turnOffFailed: 'Could not turn the pet off.'
}
