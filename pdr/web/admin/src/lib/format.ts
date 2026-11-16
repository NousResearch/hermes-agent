import type { Asset, AssetStatus, ConfigValue, Modality, Stage } from '../api'

export const STAGES: { key: Stage; name: string }[] = [
  { key: 'upload', name: '上传' }, { key: 'parse', name: '解析' }, { key: 'review', name: '人工核对' }, { key: 'label', name: '标注' },
  { key: 'classify', name: '分类' }, { key: 'store', name: '入库' }, { key: 'apply', name: '应用' }
]
export const stageName = (s: Stage) => STAGES.find((x) => x.key === s)?.name ?? s

export const STATUS: Record<AssetStatus, [string, string]> = {
  running: ['进行中', 'blue'],
  pending: ['待处理', 'purple'],
  confirm: ['待确认', 'amber'],
  retrying: ['待重试', 'amber'],
  failed: ['失败', 'pink'],
  done: ['已完成', 'green']
}

export const MODALITY_TINT: Record<Modality, string> = {
  PDF: 'purple', Word: 'blue', Excel: 'green', Markdown: 'green', 图片: 'amber', 视频: 'blue', 音频: 'pink'
}

export const MODALITIES: Modality[] = ['PDF', 'Word', 'Excel', 'Markdown', '图片', '视频', '音频']

/** 任务列表里“操作”列：按阶段与状态给出下一步要去的页面 */
export function nextAction(a: Asset): { label: string; path?: string; retry?: boolean } {
  if (a.status === 'failed') return { label: '重新执行', retry: true }
  if (a.stage === 'review') return { label: '去核对', path: ['视频', '音频'].includes(a.modality) ? '/review/media' : '/review/doc' }
  if (a.stage === 'label') return { label: '去标注', path: '/review/label' }
  if (a.stage === 'classify') return { label: '确认分类', path: '/review/classify' }
  return { label: '查看' }
}

export function showValue(v: ConfigValue): string {
  if (v === null || v === undefined || v === '') return '（空）'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

export const RELOAD_TINT: Record<string, string> = { 热加载: 'green', 下一会话: 'purple', 重启: 'amber' }
