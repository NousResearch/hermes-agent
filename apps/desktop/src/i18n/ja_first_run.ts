import type { FirstRunCopy } from './types_first_run'

export const jaFirstRun: FirstRunCopy = {
  guide: {
    title: 'Hermes を設定',
    resume: 'セットアップを再開',
    upgradeRequired:
      'セッションなしのセットアップには、このマシン上の Hermes の更新が必要です。そのマシンで hermes update を実行し、Hermes を再起動してから再試行してください。チャットは変更されません。',
    intro: '任意の設定をいくつか選ぶと、いつものチャットを使えます。セットアップで会話が作成されることはありません。',
    appearance: '外観',
    model: 'モデルを選択',
    ready: 'モデルの準備ができました。',
    providerNeeded: 'メッセージを送信する前に、プロバイダーを接続するか無料モデルを使用してください。',
    freeModel: '無料モデルを使用',
    freeUnavailable: 'ここでは無料モデルを利用できません。プロバイダーを接続するか、後で選択してください。',
    providers: 'プロバイダーを接続',
    finish: 'セットアップを完了',
    skip: 'セットアップをスキップ',
    back: '戻る',
    check: '準備状況を確認',
    checking: '確認中…',
    privacy: '使用状況の指標の共有は任意です。'
  },
  beats: {
    name: '名前',
    theme: 'アクセントカラー',
    layout: 'レイアウト',
    connectors: 'アプリ',
    plugins: 'プラグイン',
    tour: 'ツアー',
    build: '最初のタスク'
  },
  namePlaceholder: 'あなたの名前',
  useName: name => `「${name}」を使う`,
  install: 'インストール',
  installed: 'インストール済み',
  loading: '読み込み中…',
  noPlugins: '現在表示できるプラグインはありません。',
  noneChosen: 'なし',
  noAnswer: '回答なし',
  answerRefused: '回答は受け付けられませんでした',
  tours: {
    quick: 'クイックツアー',
    full: 'すべて見せて',
    skip: 'スキップして何か作ろう'
  },
  skippedAll: 'セットアップをスキップしました',
  skip: 'セットアップをスキップ',
  skipTip: 'ガイド付きセットアップを終了し、新しいチャットを開きます。',
  skipFailed: 'セットアップをスキップできませんでした',
  slashOffTitle: 'セットアップ中はスラッシュコマンドを使えません',
  slashOffBody: 'セットアップが終わると通常どおり使えます。',
  localModel: {
    title: 'このコンピューターはモデルをローカルで実行できます',
    text: (model, size) =>
      `${model}（${size}）はお使いのハードウェアに適しています。無料で動作し、チャットはこのコンピューターから出ません。`,
    setUp: 'セットアップする',
    notNow: '今はしない'
  },
  intro: {
    greeting: 'こんにちは、Hermes です。',
    plan: 'まずはあなたに合わせて設定しましょう。そのあと、何かすごいものを作りましょう。'
  },
  rerun: {
    title: 'オンボーディングをもう一度実行',
    description: '現在のモデルで初回セットアップの手順をたどる、新しいセットアップチャットを開きます。',
    action: 'もう一度実行',
    failed: 'オンボーディングを開始できませんでした'
  }
}
