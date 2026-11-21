import type { AuxTaskCopyMap } from './types_aux_tasks'

export const zhAuxTasks: AuxTaskCopyMap = {
  vision: { label: '视觉', hint: '图片分析' },
  compression: { label: '压缩', hint: '上下文压缩' },
  approval: { label: '审批', hint: '智能自动批准' },
  mcp: { label: 'MCP', hint: 'MCP 工具路由' },
  title_generation: { label: '标题生成', hint: '会话标题' },
  review: { label: '评审', hint: '/review 评审子智能体' },
  voice_chat: { label: '语音聊天', hint: '语音模式回复' },
  triage_specifier: { label: '分类指定', hint: '看板任务规格补全' },
  kanban_decomposer: { label: '看板分解', hint: '任务拆解' },
  profile_describer: { label: '配置描述', hint: '自动生成配置描述' },
  curator: { label: '维护器', hint: '技能使用审查' },
  memory_query_rewrite: { label: '记忆查询改写', hint: '记忆检索查询' },
  tts_audio_tags: { label: 'TTS 音频标签', hint: 'Gemini TTS 标签插入' },
  group_router: { label: '群组路由', hint: '决定自动路由群组中由谁回复' },
  goal_judge: { label: '目标评判', hint: '/goal 契约起草与完成判定' },
  background_review: { label: '后台复盘', hint: '每轮结束后更新记忆与技能' },
  monitor: { label: '监控', hint: '重要邮件评分' },
  moa_reference: { label: 'MoA 参考', hint: '混合代理参考模型' },
  moa_aggregator: { label: 'MoA 聚合', hint: '混合代理聚合步骤' }
}
