import type { FirstRunCopy } from './types_first_run'

export const zhFirstRun: FirstRunCopy = {
  beats: {
    name: '名字',
    theme: '强调色',
    layout: '布局',
    connectors: '应用',
    plugins: '插件',
    tour: '导览',
    build: '第一个任务'
  },
  namePlaceholder: '你的名字',
  useName: name => `使用“${name}”`,
  install: '安装',
  installed: '已安装',
  loading: '正在加载…',
  noPlugins: '暂时没有可显示的插件。',
  noneChosen: '无',
  noAnswer: '未回答',
  answerRefused: '未接受该回答',
  tours: {
    quick: '快速导览',
    full: '全部展示给我',
    skip: '跳过，来做点东西'
  },
  skippedAll: '已跳过设置',
  skip: '跳过设置',
  skipTip: '结束引导设置并打开新对话。',
  skipFailed: '无法跳过设置',
  slashOffTitle: '设置期间无法使用斜杠命令',
  slashOffBody: '设置完成后即可照常使用。',
  localModel: {
    title: '这台电脑可以在本地运行模型',
    text: (model, size) => `${model}（${size}）适合你的硬件。它免费运行，你的对话会留在这台电脑上。`,
    setUp: '设置',
    notNow: '暂不'
  },
  intro: {
    greeting: '你好，我是 Hermes。',
    plan: '我们先帮你把一切设置好，然后一起做点很酷的事。'
  },
  rerun: {
    title: '重新运行新手引导',
    description: '打开一个新的设置对话，用你当前的模型走一遍首次运行的步骤。',
    action: '重新运行',
    failed: '无法启动新手引导'
  }
}
