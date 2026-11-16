/**
 * 模拟接口：后端未就绪时用于体验与联调界面。数据是示例，保存在本机浏览器（localStorage）。
 * 行为尽量贴近真实接口：流式输出、检索条数、出处、追问建议、收藏、反馈都会走一遍。
 */
import type { AskEvent, ChatMessage, Citation, PortalApi, PortalUser, SavedAnswer, Scope, SessionDetail, SessionSummary, SuggestedQuestion } from './types'

const KEY = 'pdr-portal-mock-v1'
const USER: PortalUser = { userCode: 'chenxiao', userName: '陈晓', account: 'chenxiao@demo-travel', tenantName: '示例旅行社' }

const SCOPES: Scope[] = [
  { code: 'all', name: '全部可用资料', desc: '你有权限查看的全部知识库' },
  { code: 'product', name: '产品资料', desc: '线路手册、价格表、宣传视频' },
  { code: 'contract', name: '供应商合同', desc: '酒店、地接、景区合同条款' },
  { code: 'service', name: '服务规范', desc: '导游服务、投诉处理、出行须知' }
]

const QUESTIONS: SuggestedQuestion[] = [
  { id: 'q1', text: '黄山宏村四日游，1.15 米的孩子要买门票吗？', category: '产品资料', scopeCode: 'product' },
  { id: 'q2', text: '外滩店的退改规则是什么？', category: '供应商合同', scopeCode: 'contract' },
  { id: 'q3', text: '春季华东线路有哪些出发日期？', category: '产品资料', scopeCode: 'product' },
  { id: 'q4', text: '游客投诉导游服务，门店该怎么处理？', category: '服务规范', scopeCode: 'service' },
  { id: 'q5', text: '老人参团需要注意什么？', category: '服务规范', scopeCode: 'service' },
  { id: 'q6', text: '外滩店早餐几点开始？', category: '供应商合同', scopeCode: 'contract' }
]

const C = {
  video: { chunkNo: 'K-0012', title: '黄山线路介绍.mp4', kind: 'video', location: '01:04–01:28', startMs: 64000, endMs: 88000, quote: '“身高一米二以下的儿童免门票，索道费用按景区规定另付。”（语音转写，已人工核对）' },
  manual: { chunkNo: 'K-0031', title: '2026春季华东线路产品手册.pdf', kind: 'pdf', location: '第 3 页 · 线路价格表', quote: '黄山 + 宏村 4 天，成人价、儿童价分列；价格以出团通知为准。' },
  dates: { chunkNo: 'K-0034', title: '2026春季华东线路产品手册.pdf', kind: 'pdf', location: '第 5 页 · 出发日期', quote: '黄山 + 宏村：3 月 14 日起每周六出发；杭州 + 乌镇：3 月起每周三、周六出发，5 月 31 日截止。' },
  guide: { chunkNo: 'K-0102', title: '导游服务规范.md', kind: 'markdown', location: '「儿童与老人」小节', quote: '儿童随团出行须由监护人全程陪同，报名时登记身高；70 岁以上游客须签署健康承诺书，建议有家属陪同。' },
  complaint: { chunkNo: 'K-0108', title: '导游服务规范.md', kind: 'markdown', location: '「投诉处理」小节', quote: '门店收到投诉后 2 小时内登记并反馈给计调，24 小时内给游客初步答复；涉及安全的投诉立即上报。' },
  contract: { chunkNo: 'K-0210', title: '外滩店酒店供应商合同.docx', kind: 'word', location: '第 7 条 · 退改规则', quote: '入住前 72 小时以上取消免费；72 小时内取消收取首晚房费；节假日按附件二执行。' },
  breakfast: { chunkNo: 'K-0214', title: '外滩店酒店供应商合同.docx', kind: 'word', location: '附件一 · 服务标准', quote: '早餐时间 6:30–10:00，含双早，加早按门市价 5 折结算。' }
} satisfies Record<string, Omit<Citation, 'n'>>

interface Canned { keys: string[]; answer: string; cites: (keyof typeof C)[]; followUps: string[] }

const CANNED: Canned[] = [
  {
    keys: ['门票', '儿童', '孩子', '1.15', '黄山'],
    answer: '不需要买门票。黄山景区对身高 1.2 米以下的儿童免门票，孩子 1.15 米符合条件[1]。\n需要注意两点：\n1. 索道费用不在免票范围内，按景区规定另付[1]；\n2. 团费中的儿童价单独列出，以出团通知为准[2]；报名时需登记孩子身高，并由监护人全程陪同[3]。',
    cites: ['video', 'manual', 'guide'],
    followUps: ['索道费用是多少？', '儿童需要带什么证件？']
  },
  {
    keys: ['退改', '取消', '退订'],
    answer: '外滩店的退改规则按供应商合同第 7 条执行[1]：\n1. 入住前 72 小时以上取消：**免费**；\n2. 入住前 72 小时内取消：收取**首晚房费**；\n3. 节假日另按合同附件二执行[1]。\n门店给游客答复时，建议同时说明以出团通知为准。',
    cites: ['contract'],
    followUps: ['节假日的退改规则是什么？', '外滩店早餐几点开始？']
  },
  {
    keys: ['出发', '日期', '班期', '春季'],
    answer: '春季华东线路的出发日期如下[1]：\n1. 黄山 + 宏村：3 月 14 日起，每周六出发；\n2. 杭州 + 乌镇：3 月起，每周三、周六出发，5 月 31 日截止。\n价格与余位以出团通知为准[2]。',
    cites: ['dates', 'manual'],
    followUps: ['黄山宏村线路的儿童价怎么算？', '有没有五一专线？']
  },
  {
    keys: ['投诉', '导游'],
    answer: '按《导游服务规范》的投诉处理流程[1]：\n1. 门店收到投诉后 **2 小时内**登记，并反馈给计调；\n2. **24 小时内**给游客初步答复；\n3. 涉及人身安全的投诉，立即上报，不等待流程。',
    cites: ['complaint'],
    followUps: ['投诉登记要填哪些信息？']
  },
  {
    keys: ['老人', '老年', '70'],
    answer: '老人参团的要求[1]：\n1. 70 岁以上游客须签署**健康承诺书**；\n2. 建议有家属陪同；\n3. 行程中如有登山项目，导游会提前说明强度，可选择在山下等候。',
    cites: ['guide'],
    followUps: ['健康承诺书在哪里下载？']
  },
  {
    keys: ['早餐'],
    answer: '外滩店早餐时间为 **6:30–10:00**，房价含双早；需要加早的，按门市价 5 折结算[1]。',
    cites: ['breakfast'],
    followUps: ['外滩店的退改规则是什么？']
  }
]

interface Store {
  sessions: SessionSummary[]
  messages: Record<string, ChatMessage[]>
  saved: SavedAnswer[]
  seq: number
}

const now = () => new Date().toISOString()
const daysAgo = (d: number) => new Date(Date.now() - d * 86400000).toISOString()

function seed(): Store {
  const id = 's-demo-1'
  return {
    seq: 100,
    sessions: [
      { id, title: '外滩店早餐时间', updatedAt: daysAgo(0.1) },
      { id: 's-demo-2', title: '签证材料清单', updatedAt: daysAgo(3) },
      { id: 's-demo-3', title: '导游服务投诉处理流程', updatedAt: daysAgo(5) },
      { id: 's-demo-4', title: '五一专线报名截止时间', updatedAt: daysAgo(12) }
    ],
    messages: {
      [id]: [
        { id: 'm-1', role: 'user', content: '外滩店早餐几点开始？', createdAt: daysAgo(0.1) },
        { id: 'm-2', role: 'assistant', status: 'done', retrievedQty: 1, content: CANNED[5].answer, createdAt: daysAgo(0.1), citations: cite(CANNED[5]), followUps: CANNED[5].followUps, feedback: 0 }
      ]
    },
    saved: []
  }
}

function cite(c: Canned): Citation[] {
  return c.cites.map((k, i) => ({ ...C[k], n: i + 1 }) as Citation)
}

function load(): Store {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return JSON.parse(raw) as Store
  } catch {
    // 隐私模式或存储被禁用：退回内存数据
  }
  return seed()
}

const db: Store = load()

function persist(): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(db))
  } catch {
    // 同上，仅影响刷新后是否保留
  }
}

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(t)
      reject(new DOMException('已停止', 'AbortError'))
    })
  })

function findMessage(id: string): { sessionId: string; msg: ChatMessage } | null {
  for (const [sessionId, list] of Object.entries(db.messages)) {
    const msg = list.find((m) => m.id === id)
    if (msg) return { sessionId, msg }
  }
  return null
}

function match(question: string, scope: string): Canned | null {
  const hit = CANNED.map((c) => ({ c, score: c.keys.filter((k) => question.includes(k)).length })).sort((a, b) => b.score - a.score)[0]
  if (!hit || hit.score === 0) return null
  if (scope === 'all') return hit.c
  const scopeOf = (k: keyof typeof C) => (k === 'contract' || k === 'breakfast' ? 'contract' : k === 'guide' || k === 'complaint' ? 'service' : 'product')
  return hit.c.cites.some((k) => scopeOf(k) === scope) ? hit.c : null
}

export const mockApi: PortalApi = {
  mode: 'mock',
  async login(account, password) {
    await wait(400)
    if (!account.trim() || !password) throw new Error('请输入账号和密码')
    return { token: 'mock-portal-token', user: { ...USER, account: account.trim() } }
  },
  async me() {
    return USER
  },
  async logout() {},
  async scopes() {
    return SCOPES
  },
  async suggestedQuestions() {
    return QUESTIONS
  },
  async sessions() {
    return [...db.sessions].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  },
  async session(id): Promise<SessionDetail> {
    const summary = db.sessions.find((s) => s.id === id)
    if (!summary) throw new Error('会话不存在或已删除')
    return { summary, messages: structuredClone(db.messages[id] || []) }
  },
  async createSession(title) {
    const s = { id: `s-${++db.seq}`, title: title.slice(0, 40) || '新对话', updatedAt: now() }
    db.sessions.unshift(s)
    db.messages[s.id] = []
    persist()
    return s
  },
  async renameSession(id, title) {
    const s = db.sessions.find((x) => x.id === id)
    if (s) s.title = title
    persist()
  },
  async deleteSession(id) {
    db.sessions = db.sessions.filter((s) => s.id !== id)
    delete db.messages[id]
    db.saved = db.saved.filter((x) => x.sessionId !== id)
    persist()
  },
  async ask(sessionId, question, opts, onEvent, signal) {
    const list = (db.messages[sessionId] ||= [])
    list.push({ id: `m-${++db.seq}`, role: 'user', content: question, createdAt: now() })
    const answer: ChatMessage = { id: `m-${++db.seq}`, role: 'assistant', content: '', createdAt: now(), status: 'streaming', feedback: 0 }
    list.push(answer)
    const session = db.sessions.find((s) => s.id === sessionId)
    if (session) session.updatedAt = now()
    persist()

    const emit = (e: AskEvent) => onEvent(e)
    const hit = match(question, opts.scopeCode)
    await wait(opts.deep ? 900 : 450, signal)
    const citations = hit ? cite(hit) : []
    emit({ type: 'retrieval', qty: hit ? citations.length + (opts.deep ? 3 : 0) : 0 })
    if (citations.length) emit({ type: 'citations', citations })
    const scopeName = SCOPES.find((s) => s.code === opts.scopeCode)?.name || '全部可用资料'
    const text = hit
      ? hit.answer
      : `在「${scopeName}」中没有找到与“${question}”直接相关的内容。\n可以换个说法再问，或把资料范围切换到“全部可用资料”后重试。`
    try {
      for (let i = 0; i < text.length; ) {
        const step = 2 + Math.floor(Math.random() * 3)
        const piece = text.slice(i, i + step)
        i += step
        answer.content += piece
        emit({ type: 'delta', text: piece })
        await wait(28, signal)
      }
    } finally {
      answer.status = signal.aborted ? 'stopped' : 'done'
      answer.citations = citations
      persist()
    }
    const followUps = hit ? hit.followUps : ['有哪些资料可以查？']
    answer.followUps = followUps
    answer.retrievedQty = citations.length
    persist()
    emit({ type: 'follow_ups', items: followUps })
    emit({ type: 'done', messageId: answer.id })
  },
  async feedback(messageId, value) {
    const f = findMessage(messageId)
    if (f) f.msg.feedback = value
    persist()
  },
  async setSaved(messageId, saved) {
    const f = findMessage(messageId)
    if (!f) return
    f.msg.saved = saved
    db.saved = db.saved.filter((x) => x.messageId !== messageId)
    if (saved) {
      const list = db.messages[f.sessionId]
      const idx = list.findIndex((m) => m.id === messageId)
      const question = [...list.slice(0, idx)].reverse().find((m) => m.role === 'user')?.content || ''
      db.saved.unshift({ messageId, sessionId: f.sessionId, question, answer: f.msg.content, savedAt: now() })
    }
    persist()
  },
  async savedAnswers() {
    return db.saved
  },
  async uploadAttachment(file) {
    await wait(300)
    return { id: `att-${++db.seq}`, name: file.name, size: file.size }
  }
}
