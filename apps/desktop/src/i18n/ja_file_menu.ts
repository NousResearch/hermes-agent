import type { TranslationOverrides } from './define-locale'

export const jaFileMenu = {
    revealFinder: 'Finder で表示',
    revealExplorer: 'エクスプローラーで表示',
    revealFileManager: '格納フォルダーを開く',
    revealInSidebar: 'ファイルツリーで表示',
    newFile: '新規ファイル…',
    newFolder: '新規フォルダー…',
    createFailed: '作成に失敗しました',
    deleteRemoteBody: 'リモートマシンから完全に削除されます。元に戻せません。',
    copyPath: 'パスをコピー',
    copyRelativePath: '相対パスをコピー',
    download: 'ダウンロード',
    downloadSaved: '保存しました',
    downloadFailed: 'ダウンロードに失敗しました',
    rename: '名前を変更…',
    delete: '削除',
    renameTitle: '名前を変更',
    renameLabel: '新しい名前',
    deleteTitle: name => `${name} を削除しますか？`,
    deleteBody: 'ゴミ箱に移動します。そこから復元できます。',
    pathCopied: 'パスをコピーしました',
    revealMissing: 'そのフォルダーはこのコンピューターにありません'
  } satisfies NonNullable<TranslationOverrides['fileMenu']>
