import type { TranslationOverrides } from './define-locale'

type SettingsOverrides = NonNullable<TranslationOverrides['settings']>

/** Japanese copy for the plugin settings surfaces: Settings ▸ Plugins pages and
 *  the Desktop plugin install flow. */
export const jaPluginSettings = {
  pluginPages: {
    blurb:
      'インストール済みプラグインが追加するオプションです。各プラグインに専用ページがあり、サブページを持つものもあります。',
    empty: '設定を持つプラグインはまだありません。',
    manage: 'プラグインを管理',
    agentSettings: 'エージェント設定',
    pageCount: (n: number) => `${n} ページ`,
    missing: 'このプラグインには設定ページがありません。無効化またはアンインストールされた可能性があります。'
  },
  plugins: {
    openFolder: 'デスクトッププラグインフォルダーを開く',
    installModal: {
      installUncertain:
        'Hermes はインストール結果の待機を終了しましたが、プラグインのインストールはまだ進行中の可能性があります。この画面を閉じ、再インストールする前にプラグイン一覧を再スキャンしてください。',
      installFromGit: 'Git からインストール',
      reviewRepository: 'リポジトリを確認',
      repoPlaceholder: 'https://github.com/owner/repo',
      toolsConnected: n => `${n} 個のツールを接続しました`,
      skillsReady: names =>
        names.length === 1 ? `スキル ${names[0]} の準備ができました` : `${names.length} 個のスキルの準備ができました`,
      nextChat: 'ほかのツールは次のチャットで使えます',
      serverNotConnected: (server, reason) =>
        `MCP サーバー ${server} は接続されていません${reason ? `: ${reason}` : '。'}`,
      verified: '確認済み',
      unverified: '未確認',
      reviewedByNous: 'Nous がレビュー済み。',
      installsAndEnables: profile => `${profile} にインストールし、すぐに有効になります。`,
      installsOnly: profile => `${profile} にインストールします。`,
      advanced: '詳細設定',
      versionLabel: 'バージョン',
      reviewedPin: sha => (sha ? `レビュー済みのピン ${sha}` : 'レビュー済みのピン'),
      latestCommit: '最新のコミット',
      nothingToInstall: 'このパッケージにはここでインストールするものがありません。'
    }
  }
} satisfies Pick<SettingsOverrides, 'pluginPages' | 'plugins'>

/** Japanese copy for the Capabilities ▸ Plugins tab: the page blurb and a plugin's folder access. */
export const jaPluginsTab = {
  pageBlurb: 'プラグインはこのアプリ、エージェント、または両方を拡張できます。それぞれに独立したスイッチがあります。',
  folderAccess: {
    title: 'フォルダーへのアクセス',
    hint: (name: string) =>
      `${name} はこれらのフォルダー内のファイルを読み取り、変更できます。取り消しはすぐに反映されます。`,
    revoke: '取り消す',
    revokeLabel: (path: string) => `${path} へのアクセスを取り消す`,
    revoked: (name: string, path: string) => `${name} は ${path} を使用できなくなりました。`,
    revokeFailed: (name: string) => `${name} のフォルダーアクセスを取り消せませんでした`,
    pickerTitle: (name: string) => `${name} 用のフォルダーを選択`,
    pickerButton: 'アクセスを許可',
    refusedProtected: (name: string) =>
      `${name} はこのフォルダーを使用できません。Hermes の設定、キー、またはプラグインデータが含まれています。それらを含まないフォルダーを選択してください。`,
    loadFailed: (name: string) => `${name} と共有しているフォルダーを読み込めませんでした。`
  }
} satisfies NonNullable<NonNullable<TranslationOverrides['skills']>['plugins']>
