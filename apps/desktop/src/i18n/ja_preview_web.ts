import type { TranslationOverride } from '@hermes/shared/i18n'

import type { Translations } from './types'

// The in-app browser pane's copy (`preview.web`), composed by ja.ts.
export const jaPreviewWeb: TranslationOverride<Translations['preview']['web']> = {
  embeddedPreviewHint: '埋め込みプレビューを許可しないサイトもあります。元のページをブラウザーのタブで開いてください。',
  appFailedToBoot: 'プレビューアプリの起動に失敗しました',
  serverNotFound: 'サーバーが見つかりません',
  remoteLoopback:
    'このアドレスはエージェントを実行しているマシンを指しており、このマシンではありません。ブラウザペインはページをローカルで読み込むため、リモートの開発サーバーにはポート転送か到達可能なホスト名が必要です。',
  failedToLoad: 'プレビューの読み込みに失敗しました',
  tryAgain: '再試行',
  restarting: 'Hermes を再起動中...',
  askRestart: 'Hermes にサーバーの再起動を依頼',
  lookingRestart: taskId => `Hermes は再起動するプレビューサーバーを検索中です (${taskId})`,
  restartingTitle: 'プレビューサーバーを再起動中',
  restartingMessage: 'Hermes はバックグラウンドで作業中です。進捗はプレビューコンソールで確認してください。',
  startRestartFailed: message => `サーバー再起動を開始できませんでした: ${message}`,
  restartFailed: 'サーバーの再起動に失敗しました',
  hideConsole: 'プレビューコンソールを非表示',
  showConsole: 'プレビューコンソールを表示',
  hideDevTools: 'プレビュー DevTools を非表示',
  openDevTools: 'プレビュー DevTools を開く',
  goBack: '戻る',
  goForward: '進む',
  reload: 'ページを再読み込み',
  address: 'アドレス',
  addressPlaceholder: 'アドレスを入力',
  blankPageBody: '上のアドレス欄に入力するか、Hermes にページを開くよう頼んでください。',
  finishedRestarting: message => `Hermes がプレビューサーバーの再起動を完了しました${message ? `: ${message}` : ''}`,
  failedRestarting: message => `サーバーの再起動に失敗しました: ${message}`,
  unknownError: '不明なエラー',
  restartedTitle: 'プレビューサーバーが再起動しました',
  reloadingNow: 'プレビューを再読み込み中です。',
  restartFailedTitle: 'プレビューの再起動に失敗しました',
  restartFailedMessage: 'Hermes がサーバーを再起動できませんでした。',
  stillWorking:
    'Hermes はまだ作業中ですが、再起動の結果がまだ届いていません。サーバーコマンドがフォアグラウンドで実行されている可能性があります。',
  workspaceReloading: 'ワークスペースが変更され、プレビューを再読み込み中',
  fileChanged: url => `ファイルが変更され、プレビューを再読み込み中: ${url}`,
  filesChanged: (count, url) => `${count} 件のファイルが変更され、プレビューを再読み込み中: ${url}`,
  watchFailed: message => `プレビューファイルを監視できませんでした: ${message}`,
  moduleMimeDescription:
    'モジュールスクリプトが間違った MIME タイプで提供されています。通常、静的ファイルサーバーがプロジェクトの開発サーバーの代わりに Vite/React アプリを提供していることを意味します。',
  loadFailedConsole: (code, message) => `読み込みに失敗しました${code ? ` (${code})` : ''}: ${message}`,
  unreachableDescription: 'プレビューページに到達できませんでした。',
  openTarget: url => `${url} を開く`,
  fallbackTitle: 'プレビュー'
}
