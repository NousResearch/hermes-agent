// Notification settings copy used when the desktop locale loads.
export const zhNotifications = {
  title: '通知',
  intro: '原生桌面通知，区别于应用内提示。设置按设备保存，每台电脑各自独立。',
  enableAll: '启用通知',
  enableAllDesc: '关闭后静音下方所有通知。',
  dockBounce: 'Dock 图标跳动',
  dockBounceDesc:
    'Hermes 在后台时收到通知，让 Dock 图标跳动（Windows/Linux 为任务栏闪烁），直到窗口重新获得焦点。默认关闭 —— 系统通知和 Dock 角标照常显示。',
  focusedHint: '完成提醒仅在 Hermes 处于后台时触发。',
  kinds: {
    approval: {
      label: '需要批准',
      description: '有命令正在等待你批准或拒绝。'
    },
    input: {
      label: '需要输入',
      description: 'Hermes 提出了问题，或需要密码或密钥。'
    },
    turnDone: {
      label: '回复就绪',
      description: 'Hermes 在后台时完成了一轮对话。'
    },
    turnError: {
      label: '本轮失败',
      description: '后台回合错误。'
    },
    backgroundDone: {
      label: '后台任务完成',
      description: '后台终端命令已完成。'
    },
    credits: {
      label: '额度提醒',
      description: '额度访问被暂停或恢复。'
    },
    plugin: {
      label: '插件通知',
      description: 'Hermes 在后台时，桌面插件发送了通知。'
    }
  },
  browserPermission: {
    title: '浏览器通知',
    allow: '允许',
    states: {
      default: '标签页在后台时由此浏览器显示提醒。在你允许之前,只有标签页标题会闪烁。',
      granted: '已允许。此标签页在后台时会显示提醒,标签页标题也会闪烁。',
      denied: '已在此浏览器的网站设置中被阻止。标签页标题仍会闪烁。',
      insecure: '需要 HTTPS,例如 Tailscale serve。使用普通 HTTP 时只有标签页标题会闪烁。',
      unsupported: '此浏览器无法显示通知。标签页标题仍会闪烁。'
    }
  },
  phonePresence: {
    title: '此电脑（手机通知）',
    description: '此窗口告诉 Hermes 你是否在这里。如何处理由手机通知插件决定。',
    unsupported: '在此后端上，手机通知不受你的活动影响。',
    disconnected: '未连接。',
    watchingComputer: (time: string) => `当此窗口中打开对话时，Hermes 会得知你上次使用这台电脑的时间。上次输入在 ${time} 前。`,
    watchingWindow: (time: string) => `当此窗口中打开对话时，Hermes 会得知你上次使用此窗口的时间。上次输入在 ${time} 前。`,
    watchingWindowNoInput: '当此窗口中打开对话时，Hermes 会得知你上次使用此窗口的时间。尚未检测到任何输入。'
  },
  test: '发送测试通知',
  testTitle: 'Hermes',
  testBody: '通知工作正常。',
  testSent: '测试已发送。如果没有出现，请检查系统通知权限和专注模式／勿扰模式。',
  testUnsupported: '此系统不支持原生通知。',
  muteSoundsTitle: '静音界面提示音',
  muteSoundsDesc: '关闭自动完成、唤醒词和思考提示音、触感反馈以及系统通知声音。声音预览和语音回复仍会播放。',
  completionSoundTitle: '完成提示音',
  completionSoundDesc: '智能体回合结束时播放。可在此选择预设并预览。',
  completionSoundPreview: '预览'
}
