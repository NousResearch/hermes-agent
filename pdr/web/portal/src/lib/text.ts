/** 把回答文本切成段落 / 列表项，并识别 **加粗** 与 [n] 出处角标；不用 v-html，避免注入 */

export type Part = { t: 'text'; v: string } | { t: 'bold'; v: string } | { t: 'cite'; n: number }
export interface Block {
  kind: 'p' | 'li'
  marker: string
  parts: Part[]
}

const TOKEN = /(\*\*[^*]+\*\*|\[\d+\])/g

function parts(line: string): Part[] {
  return line.split(TOKEN).filter(Boolean).map((seg): Part => {
    if (seg.startsWith('**') && seg.endsWith('**')) return { t: 'bold', v: seg.slice(2, -2) }
    const cite = /^\[(\d+)\]$/.exec(seg)
    return cite ? { t: 'cite', n: Number(cite[1]) } : { t: 'text', v: seg }
  })
}

export function toBlocks(text: string): Block[] {
  return text.split('\n').filter((l) => l.trim()).map((line) => {
    const li = /^\s*(\d+[.、]|[-•])\s*(.*)$/.exec(line)
    return li ? { kind: 'li', marker: li[1], parts: parts(li[2]) } : { kind: 'p', marker: '', parts: parts(line) }
  })
}

export function greeting(date = new Date()): string {
  const h = date.getHours()
  if (h < 5) return '夜深了'
  if (h < 11) return '早上好'
  if (h < 13) return '中午好'
  if (h < 18) return '下午好'
  return '晚上好'
}

export function groupByTime<T extends { updatedAt: string }>(items: T[], now = Date.now()): { label: string; items: T[] }[] {
  const startToday = new Date(now)
  startToday.setHours(0, 0, 0, 0)
  const t0 = startToday.getTime()
  const groups: Record<string, T[]> = { 今天: [], '7 天内': [], 更早: [] }
  for (const it of items) {
    const t = new Date(it.updatedAt).getTime()
    groups[t >= t0 ? '今天' : t >= t0 - 6 * 86400000 ? '7 天内' : '更早'].push(it)
  }
  return Object.entries(groups).filter(([, v]) => v.length).map(([label, v]) => ({ label, items: v }))
}
