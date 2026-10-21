import type { TranslationOverrides } from './define-locale'

// The boot screen's copy, composed by ja.ts.
export const jaBoot = {
  boot: {
    ready: 'Hermes Desktop の準備ができました',
    desktopBootFailedWithMessage: message => `デスクトップの起動に失敗しました: ${message}`,
    steps: {
      connectingGateway: 'ライブデスクトップゲートウェイに接続中',
      loadingSettings: 'Hermes の設定を読み込み中',
      loadingSessions: '最近のセッションを読み込み中',
      retryingRemoteBackend: 'リモート Hermes バックエンドに再接続中…',
      startingDesktopConnection: 'デスクトップ接続を開始中',
      startingHermesDesktop: 'Hermes Desktop を起動中…'
    },
    errors: {
      backgroundExited: 'Hermes バックグラウンドプロセスが終了しました。',
      backgroundExitedDuringStartup: '起動中に Hermes バックグラウンドプロセスが終了しました。',
      backendStopped: 'バックエンドが停止しました',
      desktopBootFailed: 'デスクトップの起動に失敗しました',
      gatewayConnectionLost: 'ゲートウェイへの接続が切断されました',
      gatewayConnectionLostDetail:
        'Still retrying in the background. You can keep reading and drafting — open Gateway settings if this persists.',
      gatewaySignInRequired: 'ゲートウェイへのサインインが必要です',
      ipcBridgeUnavailable: 'デスクトップ IPC ブリッジが利用できません。'
    },
    failure: {
      title: 'Hermes を起動できませんでした',
      description:
        'バックグラウンドゲートウェイが起動しませんでした。以下の回復手順をお試しください。チャットや設定は削除されません。',
      remoteTitle: 'リモートゲートウェイへのサインインが必要です',
      remoteDescription:
        'リモートゲートウェイのセッションが期限切れです。再接続するにはもう一度サインインしてください。チャットや設定は削除されません。',
      retry: '再試行',
      repairInstall: 'インストールを修復',
      useLocalGateway: 'ローカルゲートウェイを使用',
      gatewaySettings: 'ゲートウェイ設定',
      back: '戻る',
      openLogs: 'ログを開く',
      repairHint: '修復はインストーラーを再実行します。新しいマシンでは数分かかる場合があります。',
      remoteSignInHint: signInLabel =>
        `保存済みのリモートブラウザセッションからサインアウトし、${signInLabel}を開きます。代わりにバンドルされたバックエンドに切り替えるには「ローカルゲートウェイを使用」を選択してください。`,
      signOutAndSignIn: 'サインアウトして再サインイン',
      remoteFailureHint:
        '「ゲートウェイ設定」でゲートウェイの URL とサインインを確認するか、ローカルゲートウェイに切り替えてください。',
      cloudDownTitle: 'Nous Cloud エージェントが停止しています',
      cloudDownDescription:
        'このゲートウェイが接続している Nous 管理のクラウドエージェントがサーバーエラーを返しています。ここから再起動することはできません。ステータスを確認するか、ローカルゲートウェイに切り替えるか、サポートに連絡してください。',
      cloudDownHint:
        '下のボタンから Nous Portal（インスタンスの状態と操作）を開くか、Discord でサポートを受けられます。',
      cloudDownCheckPortal: 'Portal のステータスを確認',
      cloudDownDiscord: 'Discord でサポートを受ける',
      hideRecentLogs: '最近のログを非表示',
      showRecentLogs: '最近のログを表示',
      signedInTitle: 'サインインしました',
      signedInMessage: 'リモートゲートウェイに再接続中…',
      signInIncompleteTitle: 'サインインが完了していません',
      signInIncompleteMessage: '認証が完了する前にログインウィンドウが閉じられました。',
      signInFailed: 'サインインに失敗しました',
      signInToRemoteGateway: 'リモートゲートウェイにサインイン',
      signInWithProvider: provider => `${provider} でサインイン`,
      identityProvider: 'ID プロバイダー'
    }
  }
} satisfies Pick<TranslationOverrides, 'boot'>
