import type { TranslationOverrides } from './define-locale'

export const ruChrome = {
  fileMenu: {
    openWithApp: 'Открыть в приложении по умолчанию',
    revealFinder: 'Показать в Finder',
    revealExplorer: 'Показать в Проводнике',
    revealFileManager: 'Открыть содержащую папку',
    revealInSidebar: 'Показать в дереве файлов',
    copyPath: 'Копировать путь',
    copyRelativePath: 'Копировать относительный путь',
    download: 'Скачать',
    downloadSaved: 'Сохранено',
    downloadFailed: 'Не удалось скачать',
    rename: 'Переименовать…',
    delete: 'Удалить',
    renameTitle: 'Переименование',
    renameLabel: 'Новое имя',
    deleteTitle: (name: string) => `Удалить ${name}?`,
    deleteBody: 'Элемент будет перемещён в корзину — его можно восстановить оттуда.',
    pathCopied: 'Путь скопирован',
    revealMissing: 'Этой папки нет на этом компьютере'
  }
} satisfies TranslationOverrides
