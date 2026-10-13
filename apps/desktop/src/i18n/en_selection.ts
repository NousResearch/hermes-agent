import type { TranslationOverrides } from './define-locale'

export const enSelectionTranslate = {
  title: 'Translate',
  providerNote: 'Uses your configured Hermes model. Selected text may leave this device via that provider.',
  target: 'Preferred language',
  preferredHint:
    'Saved for future translations. If the text already matches a non-English target, Hermes translates it to English instead.',
  searchLanguages: 'Search languages…',
  noLanguages: 'No languages found.',
  useLanguageTag: (name, tag) => `Use ${name} (${tag})`,
  languageTagHint: 'You can also enter a language tag, such as pt-BR or zh-Hant.',
  source: 'Selected text',
  translation: 'Translation',
  translating: 'Translating…',
  failed: 'Translation failed',
  emptyResult: 'The provider returned an empty translation.',
  tooLong: 'Select 4,000 characters or fewer to translate.',
  retry: 'Retry',
  copy: 'Copy',
  copied: 'Translation copied',
  copyFailed: 'Could not copy translation'
} satisfies TranslationOverrides['selectionTranslate']

export const enSelectionActions = { readAloud: 'Read Aloud', lookUp: 'Look Up', translate: 'Translate…', stop: 'Stop' }

export const enContextMenu = {
  link: {
    openInApp: 'Open in in-app browser',
    openExternal: 'Open in external browser',
    copyUrl: 'Copy URL',
    copyResolvedUrl: 'Copy resolved URL'
  },
  image: {
    copyImage: 'Copy image',
    copyImageAddress: 'Copy image address',
    saveImageAs: 'Save image as…'
  },
  edit: {
    cut: 'Cut',
    paste: 'Paste',
    selectAll: 'Select all',
    addToDictionary: 'Add to dictionary'
  },
  page: {
    copyPageUrl: 'Copy page URL',
    inspectElement: 'Inspect element'
  }
} satisfies TranslationOverrides['contextMenu']
