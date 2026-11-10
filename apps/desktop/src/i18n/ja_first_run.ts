import type { FirstRunCopy } from './types_first_run'

export const jaFirstRun: FirstRunCopy = {
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
