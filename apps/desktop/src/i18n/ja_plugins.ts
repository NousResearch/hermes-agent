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
        `MCP サーバー ${server} は接続されていません${reason ? `: ${reason}` : '。'}`
    }
  }
} satisfies Pick<SettingsOverrides, 'pluginPages' | 'plugins'>
