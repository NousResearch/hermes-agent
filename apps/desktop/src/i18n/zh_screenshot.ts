import type { ScreenshotTranslations } from './types_screenshot'

export const zhScreenshot: ScreenshotTranslations = {
  leftCommand: '左 ⌘',
  rightCommand: '右 ⌘',
  enabledTitle: '截图快捷键',
  enabledDesc:
    '在任意应用中同时按下左右两个 Command 键，即可截取最前面的窗口并附加到当前 Hermes 草稿。绝不会自动发送。默认关闭，仅适用于这台 Mac。窗口可能包含敏感内容，请在发送前检查附件。',
  statusTitle: '截图快捷键状态',
  checking: '正在检查截图快捷键…',
  disabled: '截图快捷键已关闭。',
  starting: '正在启动快捷键监听，尚未就绪。',
  ready: '快捷键已就绪。截图会附加到当前草稿，不会发送。',
  inputPermission:
    '输入监控权限允许 Hermes 在其他应用处于活动状态时检测左右两个 Command 键。请在系统设置 → 隐私与安全性 → 输入监控中允许 Hermes，然后返回此处重试。',
  screenPermission:
    '屏幕录制权限允许 Hermes 在你使用此快捷键时截取最前面的应用窗口。请在系统设置 → 隐私与安全性 → 屏幕录制中允许 Hermes，然后返回此处重试。如果 macOS 提示，请重启 Hermes。',
  openSettings: '打开系统设置',
  retry: '重试',
  unavailable: '截图快捷键不可用。请重试或将其关闭。',
  errorTitle: '截图快捷键错误',
  loadFailed: '无法读取快捷键状态。请重试以检查当前设置。',
  saveFailed: '无法确认快捷键更改。请重试以检查当前设置。',
  permissionFailed: '无法打开系统设置。请手动打开“隐私与安全性”，然后重试。',
  captureFailed: '无法截取最前面的窗口。未附加或发送任何内容。',
  contextChanged: '截图期间当前草稿发生了变化。截图未附加或发送。'
}
