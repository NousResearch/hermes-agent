import type { RedesignOverlay } from './types'

// Simplified Chinese wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const zhRedesign: RedesignOverlay = {
  sharedMetrics: {
    collectedMilestones: '设置：完成了哪些步骤、提供商连接，以及技能、插件和例行任务的数量',
    collectedUsage: '智能体质量：未成功的编辑、出错的工具调用、卡住的循环、每个任务的成本',
    sending:
      '如果你选择分享，Hermes 每天向 Nous 发送一次统计数据，并附带此智能体的随机 ID。除了一条 Hermes 已安装的一次性记录之外，你同意之前的统计数据永远不会被发送。你可以随时在设置中更改。',
    sendDesc: '仅包含计数，每天发送一次，并附带此智能体的随机 ID。绝不包含你的消息、文件、路径或错误文本。',
    consentBody:
      'Hermes 可以统计你的使用情况：对话时长、运行了哪些模型和工具，以及何时出错。它绝不会记录你的消息、文件、路径或错误文本。',
    collectedActivity: '对话：时长、结果、错误类型、每天的活跃时间'
  },
  common: {
    bots: '智能体'
  },
  boot: {
    steps: {
      connectingGateway: '正在连接机器',
      loadingSessions: '正在加载最近对话'
    },
    errors: {
      gatewayConnectionLost: '与机器的连接已断开',
      gatewayConnectionLostDetail: '仍在后台重试。你可以继续阅读和起草消息；如果问题持续，请打开机器设置。',
      gatewaySignInRequired: '需要登录机器'
    },
    failure: {
      description: '后台 Hermes 没有启动。请尝试下面的恢复步骤；这里不会删除你的对话或设置。',
      remoteTitle: '需要重新登录远程机器',
      remoteDescription: '你的远程机器会话已过期。请重新登录以恢复连接。这些操作不会删除你的对话或设置。',
      useLocalGateway: '使用本地机器',
      gatewaySettings: '机器设置',
      remoteSignInHint: signInLabel =>
        `先退出已保存的远程浏览器对话，然后打开${signInLabel}。也可以使用本地机器切换到随应用提供的后端。`,
      remoteFailureHint: '在“机器设置”中检查机器 URL 和登录，或切换到本地机器。',
      cloudDownTitle: 'Nous Cloud 智能体已宕机',
      cloudDownDescription:
        '此机器连接的 Nous 托管云智能体正在返回服务器错误。无法在此处重启——请检查其状态、切换到本地机器或获取支持。',
      signedInMessage: '正在重新连接远程机器…',
      signInToRemoteGateway: '登录远程机器'
    }
  },
  notifications: {
    sharedProfileWarning:
      '另一个 Hermes 安装实例正在使用此智能体。两个实例共享该智能体的设置和数据，因此更改可能发生冲突。你可以继续使用，也可以在更改前关闭另一个实例。',
    errors: {
      gatewayAuthFailed: '此 Hermes 不再接受你保存的登录信息。请打开“机器”重新登录（或粘贴新的访问令牌），然后重试。'
    },
    voice: {
      couldNotStartSession: '无法启动语音对话',
      liveEnded: '实时语音对话已结束',
      liveEndedConnectionLost: '实时语音对话连接已断开。',
      liveEndedClosed: '实时语音对话已被服务端关闭。',
      liveDelegationFailed: '无法将请求交给智能体',
      liveUnavailable: reason => `GPT-Live 语音对话不可用：${reason}。已改用语音转文字。`
    },
    native: {
      inputBody: '智能体正在等待你的回应。',
      turnDoneTitle: '智能体已完成'
    }
  },
  promptNotices: {
    desktopCommands: {
      agentsEmpty: '此对话中没有活跃的子智能体或后台进程。',
      subagentsHeader: (count: string) => `子智能体（${count}）：`,
      subagentFallback: '子智能体',
      configFooter: '在“设置”中更改，或在运行 Hermes 的机器上用 `hermes config edit` 编辑 config.yaml。',
      saveLegacyRefused:
        '未保存：此机器无法导出脱敏、指定文件名或非 JSON 的副本。请更新 Hermes，或不带选项运行 /save，在连接的机器上写入原始 JSON。',
      savedOnHost: (path: string) => `已保存到连接的机器：${path}`,
      exportRemote:
        '/export 从运行 Hermes 的机器打包智能体，而该机器不是这台电脑。请在那台机器上运行 `hermes profile export`。',
      importRemote:
        '/import 在运行 Hermes 的机器上安装智能体，而该机器不是这台电脑。请在那台机器上运行 `hermes profile import <archive>`。',
      historyHermes: '智能体',
      checkpointsDisabled: '此对话未启用检查点（在 config.yaml 中设置 checkpoints.enabled）。'
    },
    sessionCommands: {
      busy: (command: string) =>
        `对话忙碌中 — 智能体仍在回复。请先停止当前回复（Stop 按钮，或在终端中按 Ctrl+C），然后再运行 /${command}。`,
      busyCompressing: (command: string) => `对话忙碌中 — 正在运行 /compress；完成后再运行 /${command}。`,
      titleSet: (title, queued) => `已设置对话标题：${title}${queued ? '（对话开始后生效）' : ''}`,
      titleCleared: '已清除对话标题。'
    }
  },
  sendDiagnostics: {
    privacyNotice:
      '这会将调试包上传到 Nous 内部存储（并非公开粘贴板）。内容包括系统信息（操作系统、版本、服务商、已配置的 API 密钥种类 — 绝不包含密钥本身）以及完整的智能体、Hermes 服务和桌面端日志（每个最多 512 KB，很可能包含对话内容、工具输出与文件路径）。上传前会先脱敏。仅 Nous 员工与获准的 Discord 版主可查看，14 天后自动删除。',
    doneDescription: '调试包已私密上传。在你的支持工单中分享以下链接，团队即可查看你的日志。'
  },
  keybinds: {
    categories: {
      profiles: '智能体',
      session: '对话'
    },
    actions: {
      'nav.profiles': '打开智能体',
      'nav.cron': '打开例行任务',
      'view.toggleProfileRail': '切换智能体栏',
      'profile.default': '切换到默认智能体',
      'profile.switch.1': '切换到智能体 1',
      'profile.switch.2': '切换到智能体 2',
      'profile.switch.3': '切换到智能体 3',
      'profile.switch.4': '切换到智能体 4',
      'profile.switch.5': '切换到智能体 5',
      'profile.switch.6': '切换到智能体 6',
      'profile.switch.7': '切换到智能体 7',
      'profile.switch.8': '切换到智能体 8',
      'profile.switch.9': '切换到智能体 9',
      'profile.switch.10': '切换到智能体 10',
      'profile.switch.11': '切换到智能体 11',
      'profile.switch.12': '切换到智能体 12',
      'profile.switch.13': '切换到智能体 13',
      'profile.switch.14': '切换到智能体 14',
      'profile.switch.15': '切换到智能体 15',
      'profile.switch.16': '切换到智能体 16',
      'profile.switch.17': '切换到智能体 17',
      'profile.switch.18': '切换到智能体 18',
      'profile.next': '下一个智能体',
      'profile.prev': '上一个智能体',
      'profile.toggleAll': '切换全部智能体视图',
      'profile.create': '创建智能体',
      'agents.new': '新建智能体（智能体页面）',
      'nav.artifacts': '打开资源库',
      'session.new': '新建对话',
      'session.newTab': '新建对话标签',
      'session.next': '下一个对话',
      'session.prev': '上一个对话',
      'session.slot.1': '切换到最近对话 1',
      'session.slot.2': '切换到最近对话 2',
      'session.slot.3': '切换到最近对话 3',
      'session.slot.4': '切换到最近对话 4',
      'session.slot.5': '切换到最近对话 5',
      'session.slot.6': '切换到最近对话 6',
      'session.slot.7': '切换到最近对话 7',
      'session.slot.8': '切换到最近对话 8',
      'session.slot.9': '切换到最近对话 9',
      'session.focusSearch': '搜索对话',
      'session.togglePin': '固定/取消固定当前对话',
      'session.archive': '归档当前对话',
      'conversation.scrollPageUp': '向上翻动一页对话',
      'conversation.scrollPageDown': '向下翻动一页对话',
      'view.toggleSidebar': '切换对话侧边栏'
    }
  },
  settings: {
    subpages: {
      gatewayConnection: '主机器',
      appearanceChatDisplay: '对话显示'
    },
    nav: {
      gateway: '机器',
      sessions: '对话'
    },
    plugins: {
      installModal: {
        partialInstall: sha =>
          sha
            ? `插件只安装了一部分：智能体部分已安装在提交 ${sha}，但由于仓库在你审阅后有变更，桌面部分未能随之安装。在“高级”中打开强制重装，可用显示的版本替换两部分；或取消，仅保留智能体部分。`
            : '插件只安装了一部分：智能体部分已安装，但由于仓库在你审阅后有变更，桌面部分未能随之安装。在“高级”中打开强制重装，可用显示的版本替换两部分；或取消，仅保留智能体部分。',
        profileLabel: '安装到智能体',
        desktopTargetFromPackage: '从上方的包加载到本应用 — 所有智能体共用',
        nextChat: '更多工具将在下一次对话中可用'
      }
    },
    vault: {
      addDescription: '加密保存在本机。智能体永远不会看到密码。',
      otpHint: '启用两步验证时网站显示的“设置密钥”。保存后智能体会自动生成验证码。'
    },
    searchPlaceholder: {
      gateway: '机器连接…',
      sessions: '搜索已归档对话…'
    },
    appearance: {
      themeProfileNote: profile => `已为智能体「${profile}」保存——每个智能体保留各自的主题。`,
      chatTextScaleTitle: '对话文字大小',
      sessionDensityTitle: '对话列表密度',
      sessionDensityDesc: '选择侧边栏对话标题下方显示的信息量。',
      tabStripDesc: '在分区上方显示标签。自动模式会在分区只有一个面板时隐藏标签，除非还开着其他对话或磁贴分区。',
      chatFontTitle: '对话字体',
      chatFontDesc: '为对话及应用界面选择已安装的字体，适合 OpenDyslexic 等易读字体；留空则使用主题字体。',
      backdropTitle: '对话背景',
      textDirectionDesc:
        '设置对话消息和输入框的文字方向。“自动”根据每段的第一个字母判断；混合文本排列不对时，可手动选择方向。代码始终从左到右显示。',
      reactionsDesc: 'iMessage 风格的表情回应 — 你可以给消息添加回应，智能体也能回应你的消息。',
      tipsDesc: '偶尔显示来自应用和智能体的提示，每条提示只出现一次。开始使用满30天后自动关闭，你可以重新开启。',
      toursDesc: '让智能体逐步高亮每个位置，带你熟悉应用。开始使用满30天后自动关闭，你可以重新开启。',
      fileBrowserDesc: '打开工作区时，在对话旁显示文件浏览器。标题栏的切换按钮也会更改此设置。',
      resumeLastSessionTitle: '启动时恢复上次对话',
      resumeLastSessionDesc: '开启后，应用冷启动时重新打开最近的对话。关闭则始终从空白新对话开始。',
      pet: {
        intro:
          '领养一只悬浮在应用上的 petdex 动画宠物，它会根据智能体的状态做出反应——工具执行时奔跑、成功时欢呼、出错时沮丧。'
      }
    },
    uninstallSection: {
      options: {
        full: {
          description: '移除应用、智能体和所有用户数据——设置、对话记录、例行任务、密钥和日志。',
          consequence: '全部内容——聊天图形界面、Hermes 智能体以及你的所有配置、对话记录、密钥和日志'
        },
        gui: {
          description: '仅移除此桌面应用。Hermes 智能体、你的配置和对话记录都会保留。'
        },
        lite: {
          description: '移除应用和 Hermes 智能体，但保留配置、对话记录和密钥，以便将来重新安装。',
          consequence: '聊天图形界面和 Hermes 智能体（配置、对话记录和密钥会保留）'
        }
      }
    },
    about: {
      channel: {
        heldBody: (v: string) =>
          `此版本比稳定版 ${v} 更新，且你的对话已在使用 ${v} 无法打开的 Hermes 运行时。在更新的稳定版发布前，Hermes 将保持当前版本。`
      }
    },
    connections: {
      title: '已注册机器',
      intro: '管理本机以及通过远程、SSH 或 Hermes Cloud 连接可访问的每台运行 Hermes 的机器。',
      stagedNote: '所有机器都保持连接。智能体、对话、消息和例行任务归属于各自机器，“对话”可以同时显示全部机器。',
      launchModeTitle: '启动时返回上次使用机器上的对话',
      launchModeDesc: '关闭时，“对话”将在主机器上打开。',
      searchPlaceholder: '搜索机器…',
      noSearchResults: '没有与搜索匹配的机器。',
      kindRemote: '远程机器',
      kindRemoteDesc: '可通过 HTTP(S) 访问的 Hermes 机器——局域网、Tailscale 或互联网。',
      urlTitle: '机器 URL',
      headersTitle: '额外机器请求头',
      headersDesc:
        '随发往此机器的每个 HTTP 和 WebSocket 请求一起发送——用于 Cloudflare Access 等访问代理（CF-Access-Client-Id / CF-Access-Client-Secret）。值加密存储。由 Hermes 管理的请求头（Authorization、Cookie、Host 等）会被忽略。',
      duplicateUrl: (label: string) => `已存在指向此机器 URL 的连接（“${label}”）。`
    },
    managedUpdates: {
      intro:
        '以事务方式更新由桌面端托管的 SSH 安装：先完成正在运行的对话，再更新远端检出，最后恢复每个智能体，并生成关联回执。',
      progress: '正在完成运行中的对话、更新远端安装并恢复智能体…',
      scopesRestored: (profiles: string) => `已恢复的智能体：${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `智能体“${profile}”未恢复：${error}`
    },
    gateway: {
      loading: '正在加载机器设置...',
      unavailableTitle: '机器设置不可用',
      unavailableDesc: '桌面 IPC 桥未暴露机器设置。',
      webappHostDesc:
        'Webapp 始终运行在提供它的机器上。下方已保存的连接是这台机器的列表，与 Hermes Desktop 应用共享；其他机器通过这台机器访问。',
      webappOauthPaste: {
        description: '如果登录标签页停在无法打开的地址（此浏览器不在连接的机器上），请将该地址粘贴到这里。'
      },
      webAccess: {
        pendingOn: '已保存。Hermes 重启后将提供网页应用。',
        pendingOff: '已关闭，但 Hermes 在重启前仍会继续提供网页应用。',
        unavailableDetail: reason => `Hermes 无法提供网页应用：${reason}`,
        restartBusy: '已保存。有对话正在进行，因此 Hermes 没有重启。对话结束后请选择“重启 Hermes”。',
        restartFailed: '已保存，但 Hermes 无法重启。请选择“重启 Hermes”重试。',
        restartFailedBecause: reason => `Hermes 无法重启：${reason}`,
        startTimedOut: 'Hermes 未能在 90 秒内提供网页应用。',
        stopTimedOut: 'Hermes 未能在 90 秒内停止提供网页应用。',
        startStages: {
          restarting: '正在重启 Hermes 以提供网页应用。'
        },
        restartGateway: '重启 Hermes'
      },
      title: '机器连接',
      intro: '默认使用本机。需要让此应用在其他机器上运行 Hermes 时，请使用远程。连接机器即可发现其中的智能体。',
      localTitle: '本地机器',
      remoteTitle: '远程机器',
      remoteAuthHint: '托管机器使用 OAuth 或用户名密码；自托管机器也可能使用会话 token。',
      cloudSavedTitle: '已保存的云机器',
      cloudSavedDesc: '使用已保存的机器，不更改默认机器。在下方登录以添加实例。在已保存的连接列表中管理名称和登录。',
      cloudUseSaved: '使用机器',
      probing: '正在检查此机器的认证方式…',
      probeError: '暂时无法访问此机器。请检查 URL；机器响应后会显示认证方式。',
      authSignedInPassword: '此机器使用用户名和密码。你已登录，会话会自动刷新。',
      authSignedInOauth: '此机器使用 OAuth。你已登录，会话会自动刷新。',
      authNeedsPassword: '此机器使用用户名和密码。请登录以授权此桌面应用。',
      authNeedsOauth: provider => `此机器使用 OAuth。请使用 ${provider} 登录以授权此桌面应用。`,
      plainTextConfirmTitle: '以明文存储访问令牌？',
      keychainEncryptionDesc:
        '默认关闭。开启后，机器 token 和登录凭据将使用系统钥匙串（Keychain Access、GNOME Keyring 或 Windows DPAPI）加密——系统可能会请求授权或密码。关闭时，它们以仅当前用户可读的普通文件形式存储。',
      diagnosticsDesc: '在文件管理器中显示 desktop.log， Hermes 启动失败时很有用。',
      incompleteTitle: '远程机器配置不完整',
      restartingTitle: '正在重新连接机器',
      savedTitle: '机器设置已保存',
      reachableTitle: '远程机器可访问',
      signedOutMessage: '已清除远程机器会话。',
      failedLoad: '机器设置加载失败',
      testFailed: '远程机器测试失败',
      applyFailed: '无法应用机器设置',
      saveFailed: '无法保存机器设置',
      webappHostTitle: 'Hermes 机器',
      envOverrideTitle: '此连接由 Hermes 的启动方式固定。',
      incompleteToken: '切换到远程前，请输入远程 URL 和对话 token。',
      incompleteTokenTest: '测试前，请输入远程 URL 和对话 token。'
    },
    profileScope: {
      editsProfile: profile => `此页面的更改将应用于智能体“${profile}”。`
    },
    model: {
      tasks: {
        profile_describer: {
          label: '智能体描述',
          hint: '自动生成智能体描述'
        },
        title_generation: {
          hint: '对话标题'
        },
        voice_chat: {
          label: '语音对话'
        }
      },
      subagents: {
        routing: {
          modes: {
            profile_routes: {
              label: '智能体路由',
              description: '使用你已有的智能体作为子智能体，沿用它自己的模型、技能和密钥。'
            }
          },
          pluginMissing: '模型路由和智能体路由来自 Advanced Delegation 插件。',
          pluginDisabled: 'Advanced Delegation 已安装，但在此智能体中处于关闭状态。',
          blockedDisabled: '为此智能体开启 Advanced Delegation 之前，子智能体不会启动。',
          profileRoutesTitle: '可用作子智能体的智能体',
          profileRoutesHint: '每个勾选的智能体都会使用自己的模型、技能、人设和密钥。',
          profileEmpty: '还没有其他智能体。可在智能体栏中创建。',
          profileMissing: name => `${name}（智能体已不存在）`,
          inactiveProfiles: profile => `智能体路由决定使用哪个已有智能体。未指定时，子智能体以“${profile}”运行。`,
          inactiveProfilesNoDefault: '智能体路由决定使用哪个已有智能体。未指定时，子智能体使用主智能体的模型。'
        }
      },
      appliesDesc: '应用于新对话。可在输入框的模型选择器中临时切换当前对话。'
    },
    billing: {
      errors: {
        remoteSpendingReconnect: who => `${who} 请从“设置 → 机器”重新连接以重新授权此设备。`,
        sessionRevoked: {
          message: '你的会话已登出。请从“设置 → 机器”重新登录。'
        },
        transport: {
          message: '账单请求在到达 Hermes 前失败'
        }
      }
    },
    toolsets: {
      browserRealProfile: {
        description:
          '将默认浏览器的登录信息和 Cookie 复制到托管快照中，智能体使用该快照进行浏览，应用内浏览器也会用它们登录。绝不会直接打开你的真实配置文件。将应用于新对话。',
        prompt: {
          body: '让智能体使用默认浏览器配置文件的快照进行浏览，无论是智能体自身还是应用内浏览器标签页，网站打开时即已登录。'
        },
        enabledMessage: '新对话将使用默认浏览器配置文件的快照进行浏览。',
        disabledMessage: '配置文件快照将被删除，应用内浏览器也会退出这些登录；新对话使用干净的浏览器。'
      },
      modelSelectedMessage: model => `${model} 将应用于新对话。`,
      terminalBackend: {
        selectedMessage: backend => `终端命令现在通过 ${backend} 运行。将应用于新对话。`,
        needsSetupConfirmDescription: detail => `${detail} 此更改生效后启动的对话在设置完成前将没有终端或文件工具。`,
        needsSetupConfirmDescriptionGeneric:
          '此后端尚未完成设置。此更改生效后启动的对话在设置完成前将没有终端或文件工具。'
      }
    },
    notifications: {
      kinds: {
        input: {
          description: '智能体提出了问题，或需要密码或密钥。'
        }
      }
    },
    fieldLabels: {
      'voice.voiceChatMode': '语音对话模式'
    },
    fieldDescriptions: {
      'display.personality': '新对话的默认助手风格。',
      'memory.memoryEnabled': '保存有助于未来对话的持久记忆。',
      'stt.echoTranscripts': '将语音消息的原始 🎙️ 转写文本发回对话。'
    },
    memoryLayers: {
      selectedDesc: current => `尚未生效：在保存此选择之前，对话仍使用 ${current}。`,
      noProviderDesc: '智能体仅使用下方的内置层。如需添加，请在 设置 › 记忆与上下文 中选择外部提供方。',
      providerDesc: name => `${name} 会在内置层之外，为每个对话添加自己的回忆。`,
      builtinDesc:
        'MEMORY.md 和 USER.md 是 Hermes 自己的记忆，独立于任何外部提供方。两者默认都会载入对话；可在 设置 › 记忆与上下文 中用记忆和用户档案开关分别关闭。提供方是对这一层的补充，绝不会取代它。'
    },
    localModels: {
      quickstartDetail: (model, size) =>
        `一键完成所有设置：本地引擎、${model}（需下载 ${size}），并设为新对话的默认模型。数据不会离开这台电脑。`,
      quickstartDetailReady: model => `一键将 ${model} 设为新对话的默认模型。所有内容都在本机运行。`,
      quickstartDoneToast: model => `${model} 已就绪——新对话将在本机运行。`
    },
    sessions: {
      loading: '正在加载已归档对话…',
      archivedTitle: '已归档对话',
      autoArchiveTitle: '自动归档闲置对话',
      autoArchiveDesc:
        '自动归档你一段时间未使用的对话。已置顶的对话永远不会被归档，也不会删除任何内容——归档的对话会移动到这里。',
      defaultDirDesc: '新对话默认从此文件夹开始，除非你选择其他目录。留空则使用你的 home 目录。',
      failedLoad: '无法加载已归档对话'
    }
  },
  skills: {
    plugins: {
      agentTitle: '智能体插件',
      agentBlurb: '为所选智能体添加工具、钩子和模型提供方。重启 Hermes 后生效。',
      pageBlurb: '插件可以扩展本应用、智能体，或两者 — 每个部分都有自己的开关。',
      halfDesktopHint: '本应用，所有智能体共用',
      halfAgent: '智能体',
      halfAgentIn: (profile: string) => `智能体 ${profile}`,
      kindAgent: '智能体',
      kindBoth: '智能体 + 桌面',
      installAgentHereTip: (profile: string) =>
        `桌面部分已加载到本应用，但智能体部分尚未安装到 ${profile}。在那里安装它。`,
      installAgentHereNoOrigin:
        '此智能体尚未安装该插件的智能体部分，且该包是手动复制的（无目录条目或 git 远程），无法从此处安装。请将其文件夹复制到该智能体的目录或从 Git 重新安装。',
      empty: '尚未为此智能体安装插件。',
      loadFailed: '无法加载智能体插件',
      alreadyInstalled: (name: string) => `${name} 已安装到此智能体。`,
      updated: (name: string) => `${name} 已更新到当前目录固定提交。重启 Hermes 后生效。`,
      autoUpdateTip: (hours: number) =>
        `Hermes 检查时（每 ${hours} 小时）自动应用插件更新。更新后的代码在新对话中加载。`,
      uninstallConfirmBody: (name: string, profile: string) =>
        `这将从智能体 ${profile} 中删除该插件的文件。它附带的桌面部分也会一并移除。之后可随时从目录或 Git 重新安装。`,
      uninstalled: (name: string) => `${name} 已卸载。重启 Hermes 后完全卸载。`,
      defaultProfile: '默认智能体'
    },
    remotePluginHint: '在此智能体所属的机器上打开它以管理插件。插件安装使用已连接的机器。',
    openAgentSettings: '智能体设置',
    appliesToNewSessions: name => `${name} 将应用于新对话。`,
    bulkUpdated: count => `已为新对话更新 ${count} 项。`,
    changesApplyNewSessions: '更改将应用于新对话。',
    openToolLibrary: '浏览工具目录'
  },
  gatewayGroupOpen: {
    lookupFailed: '无法在机器上查找此群组，请重试。'
  },
  agents: {
    close: '关闭智能体',
    subtitle: '当前回合的子智能体实时活动。',
    emptyTitle: '暂无活跃子智能体',
    emptyDesc: '当某个回合派发任务时，子智能体会在此实时显示进度。',
    moreAgents: count => `还有 ${count} 个子智能体`,
    steerPlaceholder: '此子智能体的指令',
    requestRejected: '子智能体未接受请求',
    agentsCount: count => `${count} 个智能体`
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: '机器状态、日志、重启/更新'
      },
      sessions: {
        title: '对话面板',
        detail: '搜索、置顶与管理对话'
      }
    },
    gatewayRunning: '消息服务运行中',
    gatewayStopped: '消息服务已停止',
    restartGateway: '重启 Hermes',
    gatewayRestartFailed: 'Hermes 重启失败。',
    sharedGatewayRestartTitle: '为所有智能体重启 Hermes？',
    sharedGatewayRestartDescription: bots => `此设备上的所有智能体都会重新连接：${bots}`,
    sharedGatewayRestarted: count => `Hermes 已重启（${count} 个智能体）`,
    searchPlaceholder: '搜索对话、视图与操作',
    goToSession: '前往对话',
    newSessionInProject: project => `在 ${project} 中新建对话`,
    sections: {
      sessions: '对话'
    },
    nav: {
      newChat: {
        title: '新建对话',
        detail: '开始一个新对话'
      },
      artifacts: {
        title: '资源库'
      }
    },
    providerSessions: '对话',
    pinSession: '置顶对话',
    exportSession: '导出对话',
    deleteSession: '删除对话',
    noSessions: '暂无对话。',
    hermesActiveSessions: (version, count) => `Hermes ${version} · 活跃对话 ${count}`,
    statSessions: '对话',
    maintenance: {
      backupDesc: '打包配置、记忆、技能与对话'
    }
  },
  messaging: {
    states: {
      gateway_stopped: '消息服务已停止'
    },
    hintPendingRestart: '在状态栏重启 Hermes 以应用此更改。',
    sharedListenerUrl: '可通过以下地址访问',
    hintGatewayStopped: '在状态栏启动 Hermes 以建立连接。',
    gatewayStopped: '消息服务已停止',
    restartToApply: '此更改将在 Hermes 重启后生效。',
    restartToReconnect: '新凭据将在 Hermes 重启后生效。',
    appliedLive: '已应用，无需重启 Hermes。',
    connectingLive: '正在使用新凭据连接。',
    restartNeeded: '已保存。请重启 Hermes 以应用你的消息设置。',
    restartFailedManual: 'Hermes 重启失败 — 请手动重启并检查 Hermes 日志。',
    relayLink: {
      managedNote: (platform: string) =>
        `此实例的消息由 Team Gateway 承载，因此 Hermes 每次启动都会关闭 ${platform} 的直连。若要与中继同时使用，请在 Hermes 环境中设置 GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true（直连会使实例无法缩容到零，并可能重复投递消息）。`,
      managedIntro: '此智能体的账号在 Nous Portal 中关联。'
    },
    telegramQr: {
      savedRestarting: 'Telegram 已保存；Hermes 正在重启…',
      savedRestartFailed: detail => `Telegram 已保存；Hermes 重启失败${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: 'bot 主动发送消息的频道（例行任务输出、提醒等）。'
      },
      QQBOT_HOME_CHANNEL: {
        help: '例行任务投递的默认频道或群组。'
      }
    }
  },
  webhooks: {
    disabledBody:
      'Webhook 独立于聊天渠道运作。在此启用以接受传入的 HTTP 事件；仅当订阅投递到 Telegram、Discord、Slack 或其他渠道时才需要聊天渠道。',
    restarting: 'Hermes 正在重启…',
    restartNeeded: 'Webhook 已启用。请重启 Hermes 以开始接收事件。',
    restartGateway: '重启 Hermes',
    restartFailed: (detail: string) => `Hermes 重启失败${detail}`,
    enabledRestarting: 'Webhook 已启用；Hermes 正在重启…'
  },
  profiles: {
    close: '关闭智能体',
    title: '智能体',
    count: count => `${count} 个智能体`,
    search: '搜索智能体…',
    loading: '正在加载智能体…',
    newProfile: '新建智能体',
    importProfile: '导入智能体…',
    exportProfile: '导出智能体…',
    imported: '智能体已导入',
    exported: '智能体已导出',
    failedImport: '导入智能体失败',
    failedExport: '导出智能体失败',
    exportNeedsHost: name => `浏览器无法保存智能体归档。请在运行 Hermes 的机器上运行：hermes profile export ${name}`,
    allProfiles: '全部智能体',
    showAllProfiles: '显示全部智能体',
    manageProfiles: '管理智能体…',
    connectGateway: '管理机器…',
    fleet: {
      allOnGateway: '此机器上的全部智能体',
      gateway: gateway => `${gateway} 上的智能体`,
      localDevice: '此设备（本地后端——若未安装 Hermes 则会安装，否则打开一个新对话）',
      installDeviceDesc: '这将在本地安装 Hermes，然后在这台电脑上打开一个新对话。确认之前不会开始安装。'
    },
    remoteOverride: {
      description: '此智能体的对话将在你指定的远程 Hermes 上运行，而不是这台电脑。',
      collisionWarning: (label: string) => `设置中已存在名为“${label}”的机器。此智能体的连接是独立的，不会更改它。`,
      confirmTitle: '将此智能体连接到远程机器？',
      savedTitle: '智能体已连接',
      menuItem: '连接到远程机器…',
      title: (profile: string) => `将 ${profile} 连接到远程机器`,
      confirmNote: (profile: string, host: string) =>
        `${profile} 中的新对话将在 ${host} 上运行。命令和文件读取都发生在那台电脑上，而不是这台。请只连接你信任的机器。`,
      authFailedTitle: '远程机器拒绝了已保存的令牌'
    },
    defaultProfile: '默认智能体',
    defaultDescription: '用于 Hermes 启动和新建对话。现有对话仍归属于各自的智能体。',
    failedSetDefault: '无法设置默认智能体',
    noProfiles: '暂无智能体。',
    selectPrompt: '选择一个智能体以查看其详情。',
    refresh: '刷新智能体',
    refreshing: '正在刷新智能体',
    soulDesc: '此智能体的系统提示词与人格指令。',
    soulMissing: '此智能体尚无 SOUL.md 文件。在下方输入指令并保存即可创建。config.yaml 中的人格预设需单独管理。',
    soulPlaceholder: mode => `此智能体的系统提示词 / 人格说明。\n留空则保留${mode}默认值。`,
    deleteTitle: '删除智能体？',
    createDesc: '每个智能体都有自己的指令、设置和对话，也可以使用不同的技能。',
    cloneFromDesc: '从选中的智能体复制设置、技能和 SOUL.md。',
    cloneFromDefault: '从默认智能体克隆',
    cloneFromDefaultDesc: '从你的默认智能体复制设置、技能和 SOUL.md。',
    createAction: '创建智能体',
    renameTitle: '重命名智能体',
    renameDescPrefix: '重命名会更新智能体目录以及 ',
    displayNameDesc: '设置一个在整个应用中显示的名称。内部 ID 仍为 "default"。',
    created: '智能体已创建',
    renamed: '智能体已重命名',
    deleted: '智能体已删除',
    failedLoad: '加载智能体失败',
    failedDelete: '删除智能体失败',
    failedCreate: '创建智能体失败',
    failedRename: '重命名智能体失败'
  },
  cron: {
    close: '关闭例行任务',
    title: '例行任务',
    count: count => `${count} 个例行任务`,
    search: '搜索例行任务…',
    loading: '正在加载例行任务…',
    states: {
      scheduled: '已安排'
    },
    newCron: '新建例行任务',
    emptyDescNew: '设置运行时间，智能体就会自动执行提示词，并把结果发送到你选择的目的地。',
    emptyTitleNew: '暂无例行任务',
    actionsTitle: '例行任务操作',
    resume: '恢复例行任务',
    pause: '暂停例行任务',
    edit: '编辑例行任务',
    deleteTitle: '删除例行任务？',
    resumed: '例行任务已恢复',
    paused: '例行任务已暂停',
    triggered: '例行任务已触发',
    deleted: '例行任务已删除',
    created: '例行任务已创建',
    updated: '例行任务已更新',
    failedLoad: '加载例行任务失败',
    failedUpdate: '更新例行任务失败',
    failedTrigger: '触发例行任务失败',
    failedDelete: '删除例行任务失败',
    failedSave: '保存例行任务失败',
    editTitle: '编辑例行任务',
    createTitle: '新建例行任务',
    editDesc: '更新运行时间、提示词或投递目标。更改将在下次运行时生效。',
    createDesc: '设置自动运行的提示词。运行时间可使用 Cron 语法或“每 15 分钟”这样的自然语言。',
    customScheduleLabel: '自定义运行时间',
    promptScheduleRequired: '提示词和运行时间为必填项。',
    scheduleRequired: '运行时间为必填项。',
    scriptOnlyEditHint: '此例行任务仅运行脚本（无 AI 提示词）。例行任务 ID：',
    createAction: '创建例行任务',
    tabs: {
      jobs: '例行任务'
    },
    blueprints: {
      subtitle: '现成的例行任务模板',
      dialogDesc: '填写详细信息并设置自动运行时间。',
      scheduled: '已从蓝图创建例行任务',
      emptyDesc: '此后端上没有可用的例行任务蓝图。'
    }
  },
  sidebar: {
    filter: {
      profile: '智能体'
    },
    gatewayGroups: {
      grouping: '机器与智能体',
      aliasHint: '仅更改显示名称；机器和智能体名称保持不变。'
    },
    profileRail: '智能体栏',
    nav: {
      cron: '例行任务',
      'new-session': '新建对话',
      artifacts: '资源库'
    },
    cronJobs: '例行任务',
    projects: {
      createdInPreviousContext: '项目已在之前的连接或智能体中创建。请切换回去；IDEA.md 尚未写入。',
      showAllSessions: '显示所有对话',
      moveFailed: '无法移动对话',
      moveBusy: '请先停止正在运行的回合，再移动该对话',
      toggle: (label, open) => `${open ? '展开' : '收起'} ${label} 对话`,
      showAllCount: count => `显示全部 ${count} 个对话`
    },
    row: {
      ownedByProfile: profile => `智能体：${profile}`,
      copyIdFailed: '无法复制对话 ID',
      sessionActions: '对话操作',
      sessionRunning: '对话运行中',
      renameTitle: '重命名对话',
      untitledPlaceholder: '无标题对话',
      deleteTitle: '删除对话？',
      deleted: '对话已删除',
      untitledChat: id => `对话 ${id}`
    },
    searchAria: '搜索对话',
    searchPlaceholder: '搜索对话…',
    noMatch: query => `没有对话匹配"${query}"。`,
    sessions: '对话',
    groupAriaGrouped: '以单一列表显示对话',
    groupAriaUngrouped: '按工作区分组对话',
    showSessions: '显示对话',
    projectEmpty: '暂无对话',
    projectLoadFailed: '对话加载失败',
    noSessions: '暂无对话',
    noFilterMatches: '没有对话符合这些筛选条件',
    newSessionIn: label => `在 ${label} 中新建对话`
  },
  composer: {
    commandDescs: {
      '/retry': '重试最后一条消息（重新发送给智能体）',
      '/export': '将智能体（设置、技能、主题）导出为可分享的归档文件',
      '/import': '将共享的智能体归档文件导入为新的智能体',
      '/agents': '显示活跃智能体与正在运行的任务',
      '/review': '生成独立子智能体，审查刚才讨论的工作（PR、代码、文档）',
      '/profile': '切换活跃的智能体',
      '/blueprint': '使用 blueprint 模板创建例行任务',
      '/platform': '暂停、恢复或列出故障的消息平台',
      '/clear': '开始新对话',
      '/resume': '恢复已保存的对话',
      '/topic': '启用或查看 Telegram 私聊话题对话',
      '/title': '重命名当前对话',
      '/handoff': '将此对话移交到消息平台',
      '/bg': '在独立的后台对话中运行提示词',
      '/goal': '设置持续目标，让智能体跨回合工作直到完成',
      '/heartbeat': '设置周期性提示词，在空闲时重新进入此对话',
      '/loop': '在此对话中以固定间隔重新运行提示词',
      '/status': '显示当前对话状态',
      '/learn': '从你描述的任何内容（目录、URL、此对话、笔记）学习可复用的技能',
      '/model': '切换此对话的模型'
    },
    queueLostDiscardTip: 'Hermes 在该轮次进行中重启，无法完成。丢弃后其后的排队轮次将继续执行。',
    newSessionPlaceholders: [
      '我们要构建什么？',
      '给智能体一个任务',
      '你在想什么？',
      '描述你需要什么',
      '我们该处理什么？',
      '随便问点什么',
      '从一个目标开始'
    ],
    voiceEngine: '语音对话引擎',
    voiceEngineChained: '语音转文字 + 智能体语音',
    voiceEngineLive: 'GPT-Live（全双工，委托给智能体）',
    voiceEngineChangeFailed: '无法更改语音对话引擎',
    attachUrlDesc: '智能体将抓取该页面并作为本回合的上下文。',
    queueDroppedBody: '该后台队列条目因对话多次尝试后仍无法恢复而被丢弃。队列中的其他内容不受影响。',
    browseAllSessions: '浏览所有对话…'
  },
  statusStack: {
    agents: '智能体',
    subagents: count => `${count} 个子智能体`,
    control: {
      waitSession: target => `等待对话 ${target}`,
      loopDeferredNotice: '活跃目标当前正在控制对话。',
      dueWaitingForIdle: '已到期，等待对话空闲',
      continuationBusy: '目标已恢复 — 对话正忙，请先停止当前回复（Stop 按钮或 Esc）以继续',
      controlUnavailable: msg => `对话控制不可用: ${msg}`
    },
    coding: {
      agentShip: '让智能体提交并开 PR',
      agentShipUnavailable: '拥有这些更改的对话当前不在屏幕上。'
    }
  },
  install: {
    setupChoiceDesc: '将此应用连接到已运行 Hermes 的机器，或在这台电脑上本地安装 Hermes。',
    setupChoiceDescLocal: '在这台电脑上安装 Hermes，或连接到已运行 Hermes 的机器。',
    remoteSetupDesc: '输入机器 URL。Hermes Desktop 会检测需要令牌还是浏览器登录。',
    remoteUrlTitle: '机器 URL',
    remoteUrlDesc: '使用 Hermes 机器的基础 URL；远程地址请包含 https://。',
    probing: '正在检测机器认证方式...',
    probeError: '无法连接到该 Hermes 机器。',
    authNeedsOauth: provider => `测试此机器前请先使用 ${provider} 登录。`,
    enterUrlFirst: '请先输入机器 URL。',
    tokenDesc: '粘贴远程机器 .env 文件中的会话令牌。',
    incompleteSignInTest: '测试此受 OAuth 保护的机器前请先登录。',
    incompleteTokenTest: '测试此机器前请输入会话令牌。'
  },
  shell: {
    gatewayMenu: {
      gateway: '机器',
      reconnectGateway: '重连机器',
      machineSettings: '机器',
      restartTitle: '在这台机器上重启 Hermes？'
    },
    statusbar: {
      gateway: '机器',
      messagingStopped: '消息服务已停止',
      gatewayTitle: '机器',
      agents: '智能体',
      closeAgents: '关闭智能体',
      openAgents: '打开智能体',
      subagents: count => `${count} 个子智能体`,
      cron: '例行任务',
      openCron: '打开例行任务',
      contextUsagePanel: {
        categories: {
          subagent_definitions: '子智能体定义'
        }
      },
      toggleSessionTimer: '对话计时',
      cacheHitRateTitle: '本对话的提示缓存命中率 — 缓存 token 更便宜，越高越省'
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: '已移除的机器',
    terminalGatewayRemovedBody: '此终端所在的机器已被移除，无法重新打开。请关闭此标签页。',
    terminalGatewayNoShellBody: '此机器不提供 Shell 访问，此终端无法重新打开。请关闭此标签页。'
  },
  preview: {
    saveScopeChanged: '请切换回原来的连接和智能体以保存此草稿。',
    web: {
      remoteLoopback:
        '该地址指向运行智能体的那台机器，而不是本机。浏览器面板在本地加载页面，因此远程开发服务器需要端口转发或可访问的主机名。',
      restarting: '智能体正在重启服务器...',
      askRestart: '让智能体重启服务器',
      lookingRestart: taskId => `智能体正在查找要重启的预览服务器 (${taskId})`,
      restartingMessage: '智能体正在后台工作。可在预览控制台查看进度。',
      blankPageBody: '在上方输入地址开始浏览，或让智能体打开一个页面。',
      finishedRestarting: message => `智能体已完成预览服务器重启${message ? `: ${message}` : ''}`,
      restartFailedMessage: '智能体无法重启服务器。',
      stillWorking: '智能体仍在工作，但还没有收到重启结果。服务器命令可能正在前台运行。'
    }
  },
  assistant: {
    thread: {
      errorCodes: {
        cron_session_read_only: {
          title: '此对话属于一个例行任务',
          body: '它只用于自动运行该例行任务，因此你的消息未被执行。请新建对话后再发送。'
        },
        no_reply: {
          body: '智能体已完成本轮，但回复未送达此窗口。请重试以重新发送。'
        },
        context_overflow: {
          title: '此对话过长'
        },
        format_error: {
          body: provider => `${provider} 无法接受此对话的结构。请切换服务商；若问题持续，请新建对话。`
        },
        loop_error: {
          title: '智能体无法完成此步骤'
        },
        session_read_only: {
          title: '此对话为只读',
          body: '它是 Hermes 只能显示、无法继续的已保存历史记录，因此你的消息未被执行。请新建对话后再发送。'
        },
        workspace_unavailable: {
          title: '此对话的文件夹已不存在',
          body: '此对话使用的文件夹已被移动或删除，因此你的消息未被执行。请重新创建该文件夹，或使用“移动到项目”为对话指定其他文件夹，然后重试。'
        },
        launch_credentials_unavailable: {
          title: '此对话的 API 密钥已更改',
          body: '此对话开始时使用的 API 密钥后来已被更改或删除，因此你的消息未发送。打开的对话不会自行切换到其他密钥。请新建对话以使用当前密钥。'
        },
        agent_init_failed: {
          title: '无法开始此对话',
          body: provider => `Hermes 无法为此对话连接到 ${provider}。请重试；若持续失败，请在设置中检查该服务商。`
        },
        SESSION_NOT_OWNED: {
          title: '对话已在别处打开',
          body: '此对话已在另一个 Hermes 窗口或终端中打开。请在这里新建对话，或先在那边关闭它。'
        },
        free_tier_disabled: {
          title: '登录以继续对话',
          body: '使用 Nous 账户登录即可继续对话，完全免费。'
        },
        free_tier_refused: {
          title: '登录以继续对话'
        }
      },
      loadingSession: '正在加载对话',
      loadingResponse: '智能体正在加载回复',
      errorLayers: {
        generic: '智能体无法完成此次回复',
        runtime: '智能体遇到了问题'
      },
      errorLayerBodies: {
        generic: '智能体回复时出现问题。请重试；若问题持续，请复制错误详情。'
      },
      errorStartNewChatLink: '新建对话',
      errorStartNewSession: '开始新对话',
      errorOauthExpired: provider => `你的 ${provider} 登录已过期或被撤销。请重新登录以继续对话。`
    },
    approval: {
      gatewayDisconnected: '机器未连接',
      allowSession: '允许本对话',
      alwaysDescription: pattern =>
        `这会将“${pattern}”模式加入永久允许列表 (~/.hermes/config.yaml)。智能体对类似命令将不再询问，包括当前对话和未来对话。`
    },
    clarify: {
      gatewayDisconnected: '机器未连接',
      notDelivered: '此问题未送达应用，无法在此回答。请按停止结束本轮，然后在对话中回复。'
    },
    catalogInstall: {
      targetProfile: profile => `安装到你的智能体 ${profile}`
    },
    mcpSetup: {
      gatewayDisconnected: '机器未连接',
      reloadFailed: '服务器已保存，但重新加载 MCP 工具失败 — 将在下个对话加载'
    },
    tool: {
      titles: {
        cronjob: {
          done: '例行任务',
          pending: '正在设置例行任务',
          pendingAction: '正在设置'
        },
        session_search_recall: {
          done: '已搜索对话历史',
          pending: '正在搜索对话历史'
        }
      }
    },
    themePicker: {
      basicDetail: '用于与智能体对话。'
    }
  },
  prompts: {
    gatewayDisconnected: '机器未连接',
    sudoDesc: '输入 sudo 密码前，请先检查命令。密码会发送给执行命令的智能体，并在本次对话中缓存。',
    sudoCommandUnavailable: '此智能体未提供命令。如果无法在对话中确认，请取消。',
    sudoInstallDesc:
      '智能体需要你的 sudo 密码，以在其所在机器上安装智能体屏幕软件包（TigerVNC + Xfce）。密码只会发送到那台机器。',
    secretDesc: '智能体需要一个凭据才能继续。',
    vaultSaveDesc: origin =>
      `智能体到达了 ${origin} 的登录页，但没有为它保存的登录信息。在此输入一次；它将在本机加密保存并直接填入页面，模型永远看不到密码。`,
    vaultCodeDesc: site =>
      `${site} 要求输入一次性验证码（短信、邮件或验证器应用）。在此输入，智能体会将其填入页面；模型永远看不到它。`,
    vaultCodeFootnote: '提示：在“设置 → 密码与登录”中为该登录保存验证器密钥后，智能体会自动填写验证码。'
  },
  desktop: {
    profileStatus: current => `智能体：${current}。使用 /profile <name> 或“新建对话”选择器与其他智能体开始对话。`,
    unknownProfile: '未知智能体',
    noProfileNamed: (target, available) => `没有名为“${target}”的智能体。可用：${available}`,
    newChatsProfile: name => `新对话将使用智能体 ${name}。`,
    setProfileFailed: '设置智能体失败',
    resumeStrandedBody: '与此对话的连接失败，自动重试已停止。请确认 Hermes 正在运行，然后重试。',
    poolSlotTimeoutBody:
      '所有本地智能体后端槽位都在使用中。请在“设置”→“高级”中增加智能体后端预热数量，或等待空闲后端被驱逐后重试。',
    handoff: {
      timedOut: '等待 Hermes 超时。`hermes gateway` 是否正在运行？'
    },
    sessionUnavailable: '对话不可用',
    createSessionFailed: '无法创建新对话',
    yoloSystem: active => `此对话 YOLO ${active ? '已开启' : '已关闭'}`,
    readOnlyTranscriptBody:
      '尚无已连接的后端认领这个较早的对话，因此它以只读记录方式打开。历史记录完好；在有后端认领之前无法发送消息。',
    readOnlyTranscriptSendBlocked: '该对话目前以只读记录方式打开——发送已禁用。',
    resumeStrandedTitle: '无法加载此对话',
    sessionBusy: '对话忙碌中',
    cwdStagedMessage: '重启桌面后端后，工作目录更改才会应用到当前活跃对话。',
    sessionExported: '对话已导出',
    sessionExportFailed: '无法导出对话',
    sessionOpenFailed: '无法打开此对话'
  },
  tips: {
    items: {
      cron: {
        title: '自动运行的例行任务',
        text: '设置一条提示词，让它每小时、每晚或按自定义时间自动运行。'
      },
      profiles: {
        title: '智能体彼此独立',
        text: '每个智能体都有自己的指令、设置、记忆和对话。'
      },
      skills: {
        text: '技能是一组说明文件，智能体会在需要时自行加载。'
      },
      artifacts: {
        title: '智能体做过的一切',
        text: '所有对话里的图片、文件和链接，都汇总在资源库中。'
      },
      'command-palette': {
        text: '对话、设置、技能和命令都能从命令面板找到。'
      }
    }
  },
  intro: {
    stock: {
      none: [
        '提个问题、贴出错误，或指向仓库。我能读代码、运行工具，帮你完成交付。',
        '用自己的话描述任务。我会选择合适的工具、解释计划，并在有风险的步骤前向你确认。',
        '放入文件路径、错误跟踪，或初步想法。我会调查、建议下一步，并让操作保持可撤销。',
        '搜索仓库、编辑文件、运行测试、创建 PR。告诉我目标，机械性的工作交给我。',
        '输入任务、问题或片段。我会记住对话、引用来源，并在不确定时停下来询问。'
      ]
    }
  },
  simplifiedAgents: {
    description: '使用全新的导航、侧边栏和对话布局。关闭后恢复经典界面。',
    chats: '对话',
    hideChats: '隐藏对话',
    showChats: '显示对话',
    feedbackReasons: {
      missing_chats: '找不到我的对话'
    }
  },
  firstRun: {
    guide: {
      upgradeRequired:
        '此机器上的 Hermes 需要更新才能进行无对话设置。请在该机器上运行 hermes update，重启 Hermes，然后重试。你的对话保持不变。',
      intro: '完成几项可选偏好设置后，即可正常对话。设置过程不会创建对话。'
    },
    intro: {
      greeting: '你好，我是你的智能体。'
    }
  },
  connectors: {
    unavailable: '此对话暂时无法使用连接器。',
    disclaimer: '连接为可选操作。请仅授权你希望智能体使用的应用。',
    namedAccount: (app, alias) => `智能体已将这个 ${app} 账户命名为 ${alias}。`
  },
  catalog: {
    installDescription: '此技能将在新对话中可用。请仅安装可信来源的内容。',
    installHint: '安装前请检查源代码。更改将在新对话中生效。'
  },
  sessionImport: {
    action: '导入对话',
    search: '搜索已加载的对话',
    scanError: '无法查找对话',
    emptyHelp: '此后端上的 Claude Code 和 Codex 对话将显示在这里。',
    searchHelp: '尝试其他标题或文件夹，或加载更多对话。',
    more: '加载更多对话',
    chooseHelp: '选择对话，在导入 Hermes 前查看历史记录。'
  },
  titlebar: {
    searchTitle: '搜索对话、视图与操作',
    unreadSessions: count => (count === 1 ? '1 个未读对话' : `${count} 个未读对话`)
  },
  skillDeepLink: {
    installDescription: '此技能将在新对话中可用。请仅安装可信来源的内容。'
  },
  starmap: {
    emptyDesc: '当智能体为你的工作构建技能和记忆时，会显示在这里。'
  },
  artifacts: {
    search: '搜索资源库…',
    refresh: '刷新资源库',
    refreshing: '正在刷新资源库',
    indexing: '正在索引最近对话的产出',
    noArtifactsTitle: '资源库中还没有内容',
    noArtifactsDesc: '当对话生成图片和文件输出时，它们会显示在这里。',
    failedLoad: '资源库加载失败',
    colSession: '对话',
    partialIndex: (failed, total) => `索引资源库时跳过了 ${total} 个最近对话中的 ${failed} 个。`
  },
  artifactPreview: {
    missingTitle: '项目不可用',
    missingBody: '此项目已不在资源库中。'
  },
  updates: {
    bundleSwapPendingDesc: '更新后的应用已安装完成，只需重启 Hermes 即可加载新版本。对话记录和设置不会受到影响。',
    applyStatus: {
      noReturn: '后端未恢复在线。更新可能未完成——请检查运行 Hermes 的机器。'
    }
  },
  onboarding: {
    headerTitle: '开始设置你的智能体'
  },
  interfaceMode: {
    hint: '只改变显示的内容，不改变智能体的能力。',
    simple: {
      description: '用于与智能体对话。只有侧边栏和对话；没有终端、文件或差异面板。'
    }
  },
  zones: {
    closeRunningBody: '此对话仍在运行（或正在等待你的输入）。关闭标签只会隐藏它——对话将保留进度，可从侧边栏重新打开。',
    newSessionTab: '新建对话标签'
  }
}
