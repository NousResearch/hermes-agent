import type { TranslationOverrides } from './define-locale'

export const zhCommon = {
  common: {
    apply: '应用',
    back: '返回',
    save: '保存',
    saving: '保存中…',
    cancel: '取消',
    change: '更改',
    choose: '选择',
    clear: '清除',
    close: '关闭',
    collapse: '收起',
    confirm: '确认',
    connect: '连接',
    connecting: '连接中',
    continue: '继续',
    copied: '已复制',
    copy: '复制',
    copyFailed: '复制失败',
    delete: '删除',
    docs: '文档',
    done: '完成',
    error: '错误',
    expand: '展开',
    failed: '失败',
    formatJson: '格式化 JSON',
    free: '免费',
    loading: '加载中…',
    notSet: '未设置',
    refresh: '刷新',
    remove: '移除',
    replace: '替换',
    retry: '重试',
    run: '运行',
    send: '发送',
    set: '设置',
    skip: '跳过',
    update: '更新',
    tryHint: term => `试试“${term}”`,
    on: '开',
    off: '关'
  },

  billingBlock: {
    titleNous: 'Nous 额度已用尽',
    titleProvider: provider => `额度已用尽 — ${provider}`,
    fallbackMessage: '您的账户额度已用尽。请充值以继续使用。',
    openBilling: '打开账单',
    addCredits: '添加额度',
    dismiss: '忽略'
  },

  ui: {
    search: {
      clear: '清除搜索'
    },
    pagination: {
      label: '分页',
      previous: '上一页',
      previousAria: '前往上一页',
      next: '下一页',
      nextAria: '前往下一页'
    },
    sidebar: {
      title: '侧边栏',
      description: '显示移动端侧边栏。',
      toggle: open => `${open ? '显示' : '隐藏'}侧边栏`
    }
  }
} satisfies Pick<TranslationOverrides, 'common' | 'billingBlock' | 'ui'>
