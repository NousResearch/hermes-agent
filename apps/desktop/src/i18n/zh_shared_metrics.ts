export const zhSharedMetrics = {
  consentTitle: '帮助改进 Hermes？',
  consentBody:
    '共享指标只包含有上限的计数，绝不包含提示词、文件、路径或错误文本。收集仅在本地进行；发送给 Nous 需要另行同意。',
  whatIsCollected: '收集哪些内容',
  collectedIntro: '仅限有上限的计数：',
  collectedActivity:
    '活动、会话时长、结果和错误类别，包括记忆写入或上下文压缩被拒绝、失败或跳过时的原因（来自固定列表）',
  collectedModels: '模型路由和 token 总量',
  collectedNames: '内置工具、命令和目录项名称',
  collectedMilestones: '分桶的设置计数',
  collectedReliability: '更新和安装的结果与耗时（失败时包括来自固定列表的原因和所在阶段；全新安装记录在本机，仅在你同意后计入）、崩溃、启动与回复速度、消息平台状态',
  collectedUsage:
    'Hermes 的使用方式：代理的准确度与效率（编辑是否成功、循环、错误后的恢复、每个任务的 token 与工具调用数、缓存中断），各界面与 Desktop 模式的活跃时间，哪些应用区域、操作与设置被使用、很快关闭或被关闭，以及提供商设置的结果',
  collectedMachine:
    '概略的机器信息：内存范围、GPU 类型、Hermes 版本新旧与发布通道、落后的更新数、是否使用本地模型服务器',
  installId:
    '发送会把每日数据包上传到 Nous 遥测服务。数据包带有此配置文件的安装 ID：一个不含个人信息的固定随机 UUID，删除共享指标目录即可重置。',
  consentWindow:
    '只有整个收集周期都落在已记录同意时段内的数据包才会被发送。除全新安装记录（记录在本机，仅在你同意后计入）外，你同意之前的数据，或发送关闭期间的数据，都会留在本机。你可以随时再次关闭发送。',
  readDocs: '查看完整说明',
  share: '收集并发送给 Nous',
  local: '仅在本地收集',
  off: '不用了',
  changeLater: '你可以随时在 设置 → 安全 中更改。',
  saveFailed: '无法保存你的选择',
  collectLabel: '收集使用统计',
  collectDesc: '在此设备上保存有上限的计数。绝不包含提示词、文件、路径或错误文本。',
  sendLabel: '向 Nous 发送使用统计',
  sendDesc: '将每日数据包上传到 Nous 遥测服务。只发送同意时段内的数据。需要先开启收集。',
  unavailable: '请更新 Hermes 后端以更改此设置。',
  stripBody: '仅限有界计数器，绝不包含提示词或文件。',
  stripReaskBody: '再次询问：旧版本可能在你看到此问题之前就已保存了“不用了”。',
  stripChoices: { share: '发送给 Nous', local: '仅本地', off: '不用了' },
  stripDetails: '详情'
}
