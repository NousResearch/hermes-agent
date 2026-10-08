import type { Translations } from './types'

export const zhQuestionnaire: Translations['questionnaire'] = {
  skipSetup: '跳过设置',
  back: '返回',
  trailLabel: '你的回答',
  changeAnswer: '修改此回答',
  retry: '重试',
  saveFailed: '无法将设置标记为完成',
  skipped: '已跳过',
  otherLabel: '其他回答',
  kinds: {
    accent: '强调色',
    layout: '布局',
    local: '本地模型',
    apps: '本机应用',
    connectors: '连接器'
  },
  name: {
    greeting: '你好，我是 Hermes。',
    question: '我该怎么称呼你？',
    placeholder: '你的名字'
  },
  accent: {
    title: '选一个颜色',
    custom: '自定义颜色'
  },
  layout: {
    title: '你想怎么使用？',
    basic: '基础',
    basicDetail: '用于和 Hermes 对话。',
    elite: '高级',
    eliteDetail: '面向开发者：终端、文件、差异。'
  },
  local: {
    titleSpark: '你的 Spark 可以离线运行 Hermes',
    title: kind => `这台${kind}可以在本地运行模型`,
    download: model => `下载 ${model}`,
    downloadDetail: '第一次对话开始后在后台下载。',
    notNow: '暂不',
    notNowDetail: '会保留在模型菜单中，显示为“本地运行”。',
    chip: '芯片',
    gpu: 'GPU',
    memory: '内存',
    os: '系统',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: '在这些应用中使用 Hermes？',
    plugin: '插件',
    pluginNotRunning: '插件 · 应用未运行',
    approve: '每次安装都会在第一次对话中由你确认。',
    found: '在本机上找到。未安装的应用不会列出。'
  },
  connectors: {
    title: '连接你的应用？',
    checking: '正在检查你的账户可以连接哪些应用…',
    offered: '这是你的账户目前提供的列表。你会在第一次对话中逐个登录。'
  },
  task: {
    title: '先做什么？',
    question: '选一个，或自己输入。',
    blender: '在 Blender 中搭建场景',
    nvidia: '调优我的 GPU',
    broadcast: '设置麦克风和摄像头',
    brief: '简报今天的安排',
    tidy: '整理下载文件夹'
  },
  tour: {
    title: '要看看导览吗？',
    question: '之后快速看一圈？',
    quick: '带我看看',
    quickDetail: '几个站点，不到一分钟。',
    none: '我自己摸索',
    noneDetail: '之后可以随时让 Hermes 带你导览。'
  },
  start: {
    title: name => (name ? `准备好了，${name}。` : '准备好了。'),
    sub: '这是 Hermes 收到的第一条消息。请求显示在最前，其余内容在下方。',
    aboutMe: '关于我：',
    beforeYouStart: '开始之前：',
    noTask: '未选择任务：Hermes 会问你想做什么。',
    action: '开始',
    waiting: '正在准备你的免费账户…',
    failed: '无法开始第一次对话'
  },
  summary: {
    custom: '自定义',
    local: model => `本地：${model}`,
    localNo: '本地：暂不',
    noApps: '无应用',
    noConnectors: '无连接器',
    more: (label, count) => `${label} +${count}`,
    tour: '导览',
    noTour: '不导览'
  },
  status: {
    settingUp: '正在设置你的免费账户',
    stillSettingUp: '仍在设置你的免费账户',
    unavailable: '免费账户不可用',
    ready: 'Nous · 免费版',
    downloadingEngine: '正在下载本地引擎',
    downloading: model => `正在下载 ${model}`
  },
  settings: {
    title: '重新运行设置',
    description: '再次打开首次启动时的问题。你的回答会应用到默认配置文件。',
    action: '运行设置',
    failed: '无法开始设置'
  }
}
