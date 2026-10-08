import type { Translations } from './types'

export const jaQuestionnaire: Translations['questionnaire'] = {
  skipSetup: 'セットアップをスキップ',
  back: '戻る',
  trailLabel: 'あなたの回答',
  changeAnswer: 'この回答を変更',
  retry: '再試行',
  saveFailed: 'セットアップの完了を保存できませんでした',
  skipped: 'スキップ',
  otherLabel: 'その他の回答',
  kinds: {
    accent: 'アクセントカラー',
    layout: 'レイアウト',
    local: 'ローカルモデル',
    apps: 'このマシンのアプリ',
    connectors: 'コネクタ'
  },
  name: {
    greeting: 'こんにちは、Hermes です。',
    question: 'なんとお呼びすればいいですか？',
    placeholder: 'お名前'
  },
  accent: {
    title: '色を選んでください',
    custom: 'カスタムカラー'
  },
  layout: {
    title: 'どのように使いますか？',
    basic: 'ベーシック',
    basicDetail: 'Hermes と会話するために。',
    elite: 'エリート',
    eliteDetail: '開発者向け：ターミナル、ファイル、差分。'
  },
  local: {
    titleSpark: 'お使いの Spark で Hermes をオフライン実行できます',
    title: kind => `この${kind}ではモデルをローカルで実行できます`,
    download: model => `${model} をダウンロード`,
    downloadDetail: '最初のチャットが始まるとバックグラウンドでダウンロードします。',
    notNow: '今はしない',
    notNowDetail: 'モデルメニューに「ローカルで実行」として残ります。',
    chip: 'チップ',
    gpu: 'GPU',
    memory: 'メモリ',
    os: 'OS',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: 'これらのアプリで Hermes を使いますか？',
    plugin: 'プラグイン',
    pluginNotRunning: 'プラグイン · アプリ未起動',
    approve: '各インストールは最初のチャットで承認します。',
    found: 'このマシンで見つかりました。インストールされていないアプリは表示されません。'
  },
  connectors: {
    title: 'アプリを接続しますか？',
    checking: 'アカウントで接続できるアプリを確認中…',
    offered: '現在アカウントで利用できる一覧です。各アプリへのサインインは最初のチャットで行います。'
  },
  task: {
    title: 'まず何をしましょう？',
    question: '1つ選ぶか、自由に入力してください。',
    blender: 'Blender でシーンを作る',
    nvidia: 'GPU を調整する',
    broadcast: 'マイクとカメラを設定する',
    brief: '今日の予定をまとめる',
    tidy: 'ダウンロードフォルダを整理する'
  },
  tour: {
    title: 'ツアーを見ますか？',
    question: 'このあと少し案内しましょうか？',
    quick: '案内して',
    quickDetail: '数か所だけ、1分以内。',
    none: '自分で見てみる',
    noneDetail: 'ツアーはあとで Hermes に頼めます。'
  },
  start: {
    title: name => (name ? `準備完了、${name}さん。` : '準備完了。'),
    sub: 'これが Hermes が最初に受け取るメッセージです。依頼が先頭に表示され、残りはその下に続きます。',
    aboutMe: '自分について：',
    beforeYouStart: '始める前に：',
    noTask: 'タスク未選択：Hermes が何をしたいか尋ねます。',
    action: '開始',
    waiting: '無料アカウントを準備中…',
    failed: '最初のチャットを開始できませんでした'
  },
  summary: {
    custom: 'カスタム',
    local: model => `ローカル：${model}`,
    localNo: 'ローカル：今はしない',
    noApps: 'アプリなし',
    noConnectors: 'コネクタなし',
    more: (label, count) => `${label} +${count}`,
    tour: 'ツアー',
    noTour: 'ツアーなし'
  },
  status: {
    settingUp: '無料アカウントを設定中',
    stillSettingUp: '無料アカウントをまだ設定中',
    unavailable: '無料アカウントを利用できません',
    ready: 'Nous · 無料プラン',
    downloadingEngine: 'ローカルエンジンをダウンロード中',
    downloading: model => `${model} をダウンロード中`
  },
  settings: {
    title: 'セットアップをやり直す',
    description: '初回起動時の質問をもう一度開きます。回答はデフォルトのプロファイルに適用されます。',
    action: 'セットアップを実行',
    failed: 'セットアップを開始できませんでした'
  }
}
