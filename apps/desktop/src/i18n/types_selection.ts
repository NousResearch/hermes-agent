export interface SelectionTranslateCopy {
  title: string
  providerNote: string
  target: string
  preferredHint: string
  searchLanguages: string
  noLanguages: string
  useLanguageTag: (name: string, tag: string) => string
  languageTagHint: string
  source: string
  translation: string
  translating: string
  failed: string
  emptyResult: string
  tooLong: string
  retry: string
  copy: string
  copied: string
  copyFailed: string
}
export interface SelectionActionCopy {
  readAloud: string
  lookUp: string
  translate: string
  stop: string
}

export interface ContextMenuCopy {
  link: {
    openInApp: string
    openExternal: string
    copyUrl: string
    copyResolvedUrl: string
  }
  image: {
    copyImage: string
    copyImageAddress: string
    saveImageAs: string
  }
  edit: {
    cut: string
    paste: string
    selectAll: string
    addToDictionary: string
  }
  page: {
    copyPageUrl: string
    inspectElement: string
  }
}
