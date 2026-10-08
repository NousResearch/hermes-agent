import type { TranslationOverrides } from './define-locale'

export const jaOnboarding: TranslationOverrides['onboarding'] = {
  headerTitle: 'Hermes Agent のセットアップをしましょう',
  headerDesc: 'チャットを始めるにはモデルプロバイダーを接続してください。ほとんどのオプションはワンクリックです。',
  preparingInstall: 'Hermes はインストールを完了中です。初回実行では通常 1 分以内に完了します。',
  starting: 'Hermes を起動中…',
  lookingUpProviders: 'プロバイダーを検索中...',
  collapse: '折りたたむ',
  otherProviders: 'その他のプロバイダー',
  haveApiKey: 'API キーをお持ちです',
  chooseLater: '後でプロバイダーを選択します',
  recommended: '推奨',
  connected: '接続済み',
  featuredPitch: '1 つのサブスクリプションで 300 以上の最先端モデル — Hermes を実行するための推奨方法',
  fireworksPitch: '直接モデル API — Fireworks がホストする最先端モデル',
  localModelsTitle: 'モデルをローカルで実行',
  localModelsPitch: 'アカウント不要——モデルをダウンロードしてこのマシンで実行',
  openRouterPitch: '1 つのキーで数百のモデル — 堅実なデフォルト',
  apiKeyOptions: {
    fireworks: {
      short: 'モデル API に直接接続',
      description: 'Fireworks AI がホストするモデルに直接アクセスします。'
    },
    openrouter: {
      short: '1 つのキーで多くのモデル',
      description: '1 つのキーで数百のモデルをホスト。新規インストールのデフォルトとして最適。'
    },
    openai: { short: 'GPT クラスのモデル', description: 'OpenAI モデルへの直接アクセス。' },
    gemini: { short: 'Gemini モデル', description: 'Google Gemini モデルへの直接アクセス。' },
    xai: { short: 'Grok モデル', description: 'xAI Grok モデルへの直接アクセス。' },
    local: {
      short: 'セルフホスト',
      description:
        'ローカルまたはセルフホストの OpenAI 互換エンドポイント（vLLM、llama.cpp、Ollama など）に Hermes を接続。'
    }
  },
  backToSignIn: 'サインインに戻る',
  getKey: 'キーを取得',
  replaceCurrent: '現在の値を置き換え',
  pasteApiKey: 'API キーを貼り付け',
  couldNotSave: '認証情報を保存できませんでした。',
  connecting: '接続中',
  update: '更新',
  flowSubtitles: {
    pkce: 'ブラウザーを開いてサインインし、ここに戻ります',
    device_code: 'ブラウザーで確認ページを開きます — Hermes が自動接続します',
    external: 'ターミナルで一度サインインして、チャットに戻ります'
  },
  startingSignIn: provider => `${provider} のサインインを開始中...`,
  verifyingCode: provider => `${provider} でコードを確認中...`,
  connectedProvider: provider => `${provider} が接続されました`,
  connectedPicking: provider => `${provider} が接続されました。デフォルトモデルを選択中...`,
  signInFailed: 'サインインに失敗しました。再試行してください。',
  signInExpired:
    '承認待ちでタイムアウトしました。多くの場合、開いたタブのサインインページが止まっている（サーバー側の問題）ためです。そのページでサインインを完了してから再試行してください。解決しない場合は API キーまたは CLI を利用してください。',
  pickDifferentProvider: '別のプロバイダーを選択',
  signInWith: provider => `${provider} でサインイン`,
  openedBrowser: provider => `${provider} をブラウザーで開きました。`,
  authorizeThere: 'そこで Hermes を承認してください。',
  copyAuthCode: '認証コードをコピーして以下に貼り付けてください。',
  pasteAuthCode: '認証コードを貼り付け',
  reopenAuthPage: '認証ページを再度開く',
  waitingAuthorize: '承認を待っています...',
  externalPending: provider =>
    `${provider} は独自の CLI からサインインします。ターミナルでこのコマンドを実行してから、戻って「サインインしました」を選択してください:`,
  signedIn: 'サインインしました',
  deviceCodeOpened: provider => `${provider} をブラウザーで開きました。そこにこのコードを入力してください:`,
  reopenVerification: '確認ページを再度開く',
  copy: 'コピー',
  defaultModel: 'デフォルトモデル',
  freeTier: '無料プラン',
  pro: 'Pro',
  free: '無料',
  price: (input, output) => `${input} 入力 / ${output} 出力 per Mtok`,
  change: '変更',
  startChatting: '始める',
  docs: provider => `${provider} ドキュメント`
}
