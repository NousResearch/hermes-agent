import type { TranslationOverrides } from './define-locale'

export const ruFileMenu = {
    revealFinder: 'Показать в Finder',
    revealExplorer: 'Показать в Проводнике',
    revealFileManager: 'Открыть содержащую папку',
    revealInSidebar: 'Показать в дереве файлов',
    newFile: 'Новый файл…',
    newFolder: 'Новая папка…',
    createFailed: 'Не удалось создать',
    deleteRemoteBody: 'Файл будет безвозвратно удалён на удалённом компьютере — отменить это действие нельзя.',
    copyPath: 'Копировать путь',
    copyRelativePath: 'Копировать относительный путь',
    download: 'Скачать',
    downloadSaved: 'Сохранено',
    downloadFailed: 'Не удалось скачать',
    rename: 'Переименовать…',
    delete: 'Удалить',
    renameTitle: 'Переименование',
    renameLabel: 'Новое имя',
    deleteTitle: name => `Удалить ${name}?`,
    deleteBody: 'Элемент будет перемещён в корзину — его можно восстановить оттуда.',
    pathCopied: 'Путь скопирован',
    revealMissing: 'Этой папки нет на этом компьютере'
  } satisfies NonNullable<TranslationOverrides['fileMenu']>
