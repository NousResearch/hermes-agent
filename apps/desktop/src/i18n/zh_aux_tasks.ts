import type { AuxTaskCopyMap } from './types_aux_tasks'

export const zhAuxTasks: AuxTaskCopyMap = {
  vision: { label: '视觉', hint: '图片分析' },
  compression: { label: '压缩', hint: '上下文压缩' },
  skills_hub: { label: '技能中心', hint: '技能搜索' },
  approval: { label: '审批', hint: '智能自动批准' },
  mcp: { label: 'MCP', hint: 'MCP 工具路由' },
  title_generation: { label: '标题生成', hint: '会话标题' },
  review: { label: '评审', hint: '/review 评审子智能体' },
  voice_chat: { label: '语音聊天', hint: '语音模式回复' },
  triage_specifier: { label: '分类指定', hint: '看板任务规格补全' },
  kanban_decomposer: { label: '看板分解', hint: '任务拆解' },
  profile_describer: { label: '配置描述', hint: '自动生成配置描述' },
  curator: { label: '维护器', hint: '技能使用审查' }
}
