import type { TranslationOverrides } from './define-locale'
import { zhHantChrome } from './zh-hant_chrome'

export const zhHantFileMenu = {
    ...zhHantChrome.fileMenu,
    newFile: '新增檔案…',
    newFolder: '新增資料夾…',
    createFailed: '建立失敗',
    deleteRemoteBody: '檔案將在遠端電腦上永久刪除，且無法復原。'
  } satisfies NonNullable<TranslationOverrides['fileMenu']>
