import type { TranslationOverrides } from './define-locale'

export const zhDiagnostics = {
  notifications: {
    region: '通知',
    hide: '隐藏',
    show: '显示',
    more: count => `另外 ${count} 条通知`,
    clearAll: '全部清除',
    dismiss: '关闭通知',
    details: '详情',
    copyDetail: '复制详情',
    copyDetailFailed: '无法复制通知详情',
    backendOutOfDateTitle: '后端版本过旧',
    backendOutOfDateMessage: '你的 Hermes 后端早于当前桌面构建，可能无法正常工作。请更新以保持一致。',
    installMethodUnsupportedTitle: '不受支持的安装方式',
    updateHermes: '更新 Hermes',
    updateReadyTitle: '有可用更新',
    updateReadyMessage: count => `有 ${count} 项新更改可用。`,
    updateReadyMessageUnknown: '有新更新可用。',
    seeWhatsNew: '查看更新内容',
    mcp: {
      needsAuthTitle: 'MCP 服务器需要重新认证',
      needsAuthMessage: name => `${name} MCP 需要重新认证。`,
      errorTitle: 'MCP 服务器无法连接',
      errorMessage: name => `${name} MCP 健康检查失败。`,
      signIn: '登录',
      view: '查看',
      disable: '禁用',
      disabledMessage: name => `已禁用 ${name} MCP。可随时在「能力 → MCP」中重新启用。`,
      disableFailed: name => `无法禁用 ${name} MCP。`
    },
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs STT 需要 ELEVENLABS_API_KEY。',
      elevenLabsRejectedKey: 'ElevenLabs 拒绝了该 API key (401)。',
      diskFull: '磁盘已满 — 请腾出一些空间后再试。',
      gatewayAuthFailed: '网关认证失败 — 请检查你的 API_SERVER_KEY。',
      methodNotAllowed: '桌面后端拒绝了该请求 (405 Method Not Allowed)。请尝试重启 Hermes Desktop。',
      microphonePermission: '麦克风权限已被拒绝。',
      openaiRejectedApiKey: 'OpenAI 拒绝了该 API key。',
      openaiTtsNeedsKey: 'OpenAI TTS 需要 VOICE_TOOLS_OPENAI_KEY 或 OPENAI_API_KEY。',
      codeSkewRestartRequired: '更新后此后端仍在运行旧代码。请重启以加载新代码。'
    },
    voice: {
      configureSpeechToText: '配置语音转文字后即可使用语音模式。',
      couldNotStartSession: '无法启动语音会话',
      microphoneAccessDenied: '麦克风访问被拒绝。',
      microphoneConstraintsUnsupported: '此设备不支持当前麦克风约束。',
      microphoneFailed: '麦克风出错',
      microphoneInUse: '麦克风正被其他应用占用。',
      microphonePermissionDenied: '麦克风权限被拒绝。',
      microphoneStartFailed: '无法开始麦克风录音。',
      microphoneUnsupported: '当前运行环境不支持麦克风录音。',
      noMicrophone: '未找到麦克风。',
      noSpeechDetected: '没有检测到语音',
      playbackFailed: '语音播放失败',
      recordingFailed: '语音录制失败',
      sayStopToEnd: phrase => `说“${phrase}”即可结束语音对话。`,
      transcriptionFailed: '语音转写失败',
      transcriptionUnavailable: '语音转写暂不可用。',
      tryRecordingAgain: '请再录一次。',
      unavailable: '语音不可用',
      liveEnded: '实时语音会话已结束',
      liveEndedConnectionLost: '实时语音会话连接已断开。',
      liveEndedClosed: '实时语音会话已被服务端关闭。',
      liveError: '实时语音',
      liveDelegationFailed: '无法将请求交给 Hermes',
      liveUnavailable: reason => `GPT-Live 语音聊天不可用：${reason}。已改用语音转文字。`
    },
    native: {
      approvalTitle: '需要批准',
      approvalTitleNamed: session => `需要批准 — ${session}`,
      approveAction: '批准',
      rejectAction: '拒绝',
      inputTitle: '需要输入',
      inputTitleNamed: session => `需要输入 — ${session}`,
      inputBody: 'Hermes 正在等待你的回应。',
      turnDoneTitle: 'Hermes 已完成',
      turnDoneBody: '',
      turnErrorTitle: '本轮失败',
      backgroundDoneTitle: '后台任务已完成',
      backgroundFailedTitle: '后台任务失败',
      creditsTitle: '额度'
    }
  },

  sendDiagnostics: {
    title: '向 Nous 发送诊断信息',
    privacyNotice:
      '这会将调试包上传到 Nous 内部存储（并非公开粘贴板）。内容包括系统信息（操作系统、版本、服务商、已配置的 API 密钥种类 — 绝不包含密钥本身）以及完整的 agent、gateway 和桌面端日志（每个最多 512 KB，很可能包含对话内容、工具输出与文件路径）。上传前会先脱敏。仅 Nous 员工与获准的 Discord 版主可查看，14 天后自动删除。',
    upload: '上传',
    uploading: '上传中…',
    cancel: '取消',
    close: '关闭',
    copyLink: '复制链接',
    uploadIdFallback: id => `未返回查看链接 — 请向支持人员提供上传 ID ${id}`,
    doneTitle: '诊断信息已发送',
    doneDescription: '调试包已私密上传。在您的支持会话中分享以下链接，团队即可查看您的日志。',
    failedTitle: '上传失败',
    failedHint:
      '您也可以在终端运行 `hermes debug share --nous`，或运行 `hermes debug share --local` 在不上传的情况下查看报告。',
    handoffLead: '在以下位置继续讨论:',
    links: {
      github: 'GitHub Issues',
      portal: 'Nous Portal 支持',
      discord: 'Discord'
    }
  },

  errors: {
    genericFailure: '发生错误',
    boundaryTitle: '界面出错了',
    boundaryDesc: '此视图遇到意外错误。你的对话和设置是安全的。',
    reloadWindow: '重新加载窗口',
    openLogs: '打开日志'
  }
} satisfies Pick<TranslationOverrides, 'notifications' | 'sendDiagnostics' | 'errors'>
