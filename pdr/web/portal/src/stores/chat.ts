import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'
import { api, type Attachment, type ChatMessage, type Citation, type Scope, type SessionSummary, type SuggestedQuestion } from '../api'

let localSeq = 0
const tempId = (p: string) => `${p}-local-${Date.now()}-${++localSeq}`

export const useChat = defineStore('chat', () => {
  const sessions = ref<SessionSummary[]>([])
  const scopes = ref<Scope[]>([])
  const questions = ref<SuggestedQuestion[]>([])
  const currentId = ref<string | null>(null)
  const messages = ref<ChatMessage[]>([])
  const loadingSession = ref(false)
  const error = ref('')

  const scopeCode = ref('all')
  const deep = ref(false)
  const attachments = ref<Attachment[]>([])
  const citation = ref<Citation | null>(null)

  let controller: AbortController | null = null
  const busy = computed(() => messages.value.some((m) => m.status === 'searching' || m.status === 'streaming'))
  const scope = computed(() => scopes.value.find((s) => s.code === scopeCode.value) || scopes.value[0])

  async function bootstrap(): Promise<void> {
    const [s, sc, q] = await Promise.all([api.sessions(), api.scopes(), api.suggestedQuestions()])
    sessions.value = s
    scopes.value = sc
    questions.value = q
  }

  function reset(): void {
    stop()
    currentId.value = null
    messages.value = []
    attachments.value = []
    citation.value = null
    error.value = ''
  }

  async function open(id: string): Promise<void> {
    if (currentId.value === id) return
    stop()
    citation.value = null
    currentId.value = id
    loadingSession.value = true
    error.value = ''
    try {
      const detail = await api.session(id)
      if (currentId.value === id) messages.value = detail.messages
    } catch (e) {
      error.value = (e as Error).message
      messages.value = []
    } finally {
      loadingSession.value = false
    }
  }

  /** 发送问题；返回会话 ID（新会话时由调用方跳转路由） */
  async function ask(question: string): Promise<string> {
    const text = question.trim()
    if (!text || busy.value) return currentId.value || ''
    let id = currentId.value
    if (!id) {
      const s = await api.createSession(text)
      sessions.value.unshift(s)
      id = s.id
      currentId.value = id
      messages.value = []
    }
    messages.value.push({ id: tempId('u'), role: 'user', content: text, createdAt: new Date().toISOString() })
    const answer = reactive<ChatMessage>({ id: tempId('a'), role: 'assistant', content: '', status: 'searching', createdAt: new Date().toISOString(), feedback: 0 })
    messages.value.push(answer)
    const sessionId = id
    controller = new AbortController()
    const opts = { scopeCode: scopeCode.value, deep: deep.value, attachmentIds: attachments.value.map((a) => a.id) }
    attachments.value = []
    run(sessionId, text, answer, opts, controller)
    return sessionId
  }

  async function run(sessionId: string, text: string, answer: ChatMessage, opts: { scopeCode: string; deep: boolean; attachmentIds: string[] }, ctl: AbortController): Promise<void> {
    try {
      await api.ask(sessionId, text, opts, (e) => {
        if (e.type === 'retrieval') answer.retrievedQty = e.qty
        else if (e.type === 'citations') answer.citations = e.citations
        else if (e.type === 'delta') {
          answer.status = 'streaming'
          answer.content += e.text
        } else if (e.type === 'follow_ups') answer.followUps = e.items
        else if (e.type === 'done') {
          answer.id = e.messageId
          answer.status = 'done'
        } else if (e.type === 'error') {
          answer.status = 'error'
          answer.content = answer.content || e.message
        }
      }, ctl.signal)
      if (answer.status === 'streaming' || answer.status === 'searching') answer.status = 'done'
    } catch (e) {
      if ((e as Error).name === 'AbortError') answer.status = 'stopped'
      else {
        answer.status = 'error'
        answer.content = answer.content || `回答失败：${(e as Error).message}`
      }
    } finally {
      if (controller === ctl) controller = null
      const s = sessions.value.find((x) => x.id === sessionId)
      if (s) {
        s.updatedAt = new Date().toISOString()
        sessions.value = [s, ...sessions.value.filter((x) => x !== s)]
      }
    }
  }

  function stop(): void {
    controller?.abort()
    controller = null
  }

  async function regenerate(messageId: string): Promise<void> {
    const idx = messages.value.findIndex((m) => m.id === messageId)
    const question = messages.value.slice(0, idx).reverse().find((m) => m.role === 'user')
    if (question) await ask(question.content)
  }

  async function setFeedback(m: ChatMessage, value: -1 | 0 | 1): Promise<void> {
    m.feedback = m.feedback === value ? 0 : value
    await api.feedback(m.id, m.feedback)
  }

  async function toggleSaved(m: ChatMessage): Promise<void> {
    m.saved = !m.saved
    await api.setSaved(m.id, m.saved)
  }

  async function remove(id: string): Promise<void> {
    await api.deleteSession(id)
    sessions.value = sessions.value.filter((s) => s.id !== id)
    if (currentId.value === id) reset()
  }

  async function attach(file: File): Promise<void> {
    attachments.value.push(await api.uploadAttachment(file))
  }

  return {
    sessions, scopes, questions, currentId, messages, loadingSession, error, scopeCode, deep, attachments, citation, busy, scope,
    bootstrap, reset, open, ask, stop, regenerate, setFeedback, toggleSaved, remove, attach
  }
})
