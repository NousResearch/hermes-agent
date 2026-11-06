/** The Appearance > Pet panel copy. */
export interface AppearancePetsCopy {
  title: string
  intro: string
  restartHint: string
  scaleTitle: string
  scaleDesc: string
  roamTitle: string
  roamDesc: string
  chooseTitle: string
  chooseDesc: string
  searchPlaceholder: string
  unreachable: string
  noMatch: (query: string) => string
  installedTag: string
  generatedTag: string
  countCapped: (cap: number, total: number) => string
  count: (n: number) => string
  uninstall: (name: string) => string
  delete: (name: string) => string
  deleteTitle: (name: string) => string
  deleteBody: string
  deleteConfirm: string
  rename: (name: string) => string
  renameTitle: string
  renamePlaceholder: string
  renameSave: string
  exportPet: (name: string) => string
  adoptFailed: (slug: string) => string
  uninstallFailed: (slug: string) => string
  renameFailed: (slug: string) => string
  exportFailed: (slug: string) => string
  noneAvailable: string
  turnOnFailed: string
  turnOffFailed: string
}
