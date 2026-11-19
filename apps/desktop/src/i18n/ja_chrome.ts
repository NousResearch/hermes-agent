import type { TranslationOverrides } from './define-locale'

export const jaChrome = {
  fileMenu: {
    openWithApp: 'デフォルトのアプリで開く',
    revealFinder: 'Finder で表示',
    revealExplorer: 'エクスプローラーで表示',
    revealFileManager: '格納フォルダーを開く',
    revealInSidebar: 'ファイルツリーで表示',
    copyPath: 'パスをコピー',
    copyRelativePath: '相対パスをコピー',
    download: 'ダウンロード',
    downloadSaved: '保存しました',
    downloadFailed: 'ダウンロードに失敗しました',
    rename: '名前を変更…',
    delete: '削除',
    renameTitle: '名前を変更',
    renameLabel: '新しい名前',
    deleteTitle: (name: string) => `${name} を削除しますか？`,
    deleteBody: 'ゴミ箱に移動します。そこから復元できます。',
    pathCopied: 'パスをコピーしました',
    revealMissing: 'そのフォルダーはこのコンピューターにありません'
  }
} satisfies TranslationOverrides
