import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const jaDelegationRouting: DelegationRoutingTranslations = {
  title: 'ルーティング',
  settingsTitle: 'サブエージェントの既定値',
  intro: '各サブエージェントがどのモデルを使うかを決めます。',
  modes: {
    default: { label: '既定', description: 'すべてのサブエージェントが下のサブエージェントモデルで動作します。' },
    model_map: {
      label: 'モデルルート',
      description: 'いくつかのモデルに名前を付けます。エージェントがサブエージェントごとに 1 つを選びます。'
    },
    profile_routes: {
      label: 'プロファイルルート',
      description: 'サブエージェントがいずれかのプロファイルとして、そのモデル、スキル、キーで動作します。'
    }
  },
  inUse: '使用中',
  plugin: 'プラグイン',
  pluginMissing: 'モデルルートとプロファイルルートは Advanced Delegation プラグインが提供します。',
  pluginDisabled: 'Advanced Delegation はインストール済みですが、このプロファイルではオフです。',
  pluginNotLoaded:
    'Advanced Delegation はオンですが、Hermes がまだ読み込んでいません。完了するには Hermes を再起動してください。',
  installPlugin: 'プラグインをインストール',
  installUnavailable:
    'Advanced Delegation はまだプラグインカタログにありません。スキルとツール › プラグイン からインストールしてください。',
  enablePlugin: 'オンにする',
  enableFailed: 'Advanced Delegation をオンにできませんでした',
  modeFailed: 'サブエージェントのルーティングを変更できませんでした',
  loadFailed: 'サブエージェントのルーティングを読み込めませんでした',
  blockedPill: '動作していません',
  blockedDetail: 'このモードが保存されているため、Hermes は黙って既定を使わず、すべてのサブエージェントを拒否します。',
  blockedMissing: 'Advanced Delegation をインストールするまで、サブエージェントは起動しません。',
  blockedDisabled: 'このプロファイルで Advanced Delegation をオンにするまで、サブエージェントは起動しません。',
  blockedNotLoaded:
    'Hermes が Advanced Delegation を読み込むまで、サブエージェントは起動しません。Hermes を再起動して完了してください。',
  blockedUnavailable: label =>
    `${label} がここで動作できるようになるまで、サブエージェントは起動しません。そのプラグインは動作できないと報告しています。`,
  blockedUnregistered: name =>
    `「${name}」を提供するプラグインをインストールしてオンにするまで、サブエージェントは起動しません。`,
  thirdPartyDescription: 'プラグインが提供します。設定はそのプラグイン側にあります。',

  modelRoutesTitle: 'モデルルート',
  modelRoutesEmpty: 'ルートはまだありません。追加するまで、サブエージェントはメインエージェントのモデルを使います。',
  addRoute: 'ルートを追加',
  editRoute: '編集',
  makeDefault: '既定にする',
  deleteRoute: '削除',
  defaultBadge: '既定',
  routeActions: name => `${name} の操作`,
  deleteConfirmTitle: name => `ルート「${name}」を削除しますか？`,
  deleteConfirmBody: 'サブエージェントはこのルートを選べなくなります。',
  deleteConfirmNextDefault: name => `既定のルートのため、「${name}」が新しい既定になります。`,
  deleteConfirmAction: 'ルートを削除',
  saveFailed: 'ルートを保存できませんでした',
  baseUrlRoute: url => `エンドポイント ${url}`,

  dialogNew: '新しいモデルルート',
  dialogEdit: 'モデルルートを編集',
  dialogDesc: 'エージェントは名前と説明を読んで、サブエージェントがどのルートを使うかを決めます。',
  nameLabel: '名前',
  namePlaceholder: '例: fast',
  nameHint: '短く、スペースなしで。エージェントはこの名前でルートを選びます。',
  descriptionLabel: '使う場面',
  descriptionPlaceholder: '例: 安くて速い: 調べもの、要約、簡単な編集。',
  descriptionHint: 'エージェントが選ぶ前に読む 1 行の説明です。',
  providerLabel: 'プロバイダー',
  providerPlaceholder: 'プロバイダーを選択',
  modelLabel: 'モデル',
  effortLabel: '推論強度',
  effortInherit: 'サブエージェントの既定',
  tierLabel: '速度',
  tierNormal: '標準',
  tierFast: '高速',
  tierUltrafast: '超高速',
  tierHint: '対応しているモデルのみ。',
  optional: '任意',
  errors: { required: '必須', spaces: 'スペースは使えません', duplicate: 'この名前は別のルートで使われています' },
  saveRoute: 'ルートを保存',
  createRoute: 'ルートを追加',

  profileRoutesTitle: 'サブエージェントが使えるプロファイル',
  profileRoutesHint: 'チェックした各プロファイルは、それぞれのモデル、スキル、ペルソナ、キーを持ち込みます。',
  profileDescriptionPlaceholder: '使う場面（任意）',
  profileDefault: '既定',
  profileEmpty: 'ほかのプロファイルはまだありません。プロファイルバーから作成してください。',
  profileMissing: name => `${name}（プロファイルは存在しません）`,

  inactiveTitle: 'このモードでは使われません',
  inactiveModelMap: route =>
    `モデルルートが各サブエージェントのモデルを選びます。ルートを選ばなかったサブエージェントは「${route}」を使います。`,
  inactiveProfiles: profile =>
    `プロファイルルートが各サブエージェントのプロファイルを選びます。プロファイルを選ばなかったサブエージェントは「${profile}」として動作します。`,
  inactiveModelMapNoDefault:
    'モデルルートが各サブエージェントのモデルを選びます。ルートを選ばなかったサブエージェントはメインエージェントのモデルを使います。',
  inactiveProfilesNoDefault:
    'プロファイルルートが各サブエージェントのプロファイルを選びます。プロファイルを選ばなかったサブエージェントはメインエージェントのモデルを使います。',
  inactivePlugin: label => `${label} が各サブエージェントのモデルを決めるため、この設定は適用されません。`,
  useDefault: '既定に切り替え'
}
