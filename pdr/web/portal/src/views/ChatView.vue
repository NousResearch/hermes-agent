<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuth } from '../stores/auth'
import { useChat } from '../stores/chat'
import { greeting } from '../lib/text'
import ShellLayout from '../components/ShellLayout.vue'
import GlowOrb from '../components/GlowOrb.vue'
import ChatComposer from '../components/ChatComposer.vue'
import AnswerMessage from '../components/AnswerMessage.vue'
import CitationDrawer from '../components/CitationDrawer.vue'
import AppIcon from '../components/AppIcon.vue'

const route = useRoute()
const router = useRouter()
const auth = useAuth()
const chat = useChat()
const thread = ref<HTMLElement | null>(null)

const id = computed(() => (route.params.id as string | undefined) || '')
const hero = computed(() => !id.value && chat.messages.length === 0)
const hello = greeting()
const suggestions = computed(() => chat.questions.slice(0, 4))
const lastAnswerId = computed(() => [...chat.messages].reverse().find((m) => m.role === 'assistant')?.id)

watch(id, async (v) => {
  if (v) await chat.open(v)
  else if (!chat.busy) chat.reset()
  scrollToEnd(true)
}, { immediate: true })

watch(() => chat.messages.map((m) => m.content.length).join(','), () => scrollToEnd(false))

function scrollToEnd(force: boolean): void {
  nextTick(() => {
    const el = thread.value
    if (!el) return
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 160
    if (force || near) el.scrollTop = el.scrollHeight
  })
}

async function send(text: string): Promise<void> {
  const sessionId = await chat.ask(text)
  if (sessionId && sessionId !== id.value) router.push(`/c/${sessionId}`)
  scrollToEnd(true)
}
</script>

<template>
  <ShellLayout show-scope>
    <div class="layout">
      <section class="col">
        <div v-if="hero" class="home-hero">
          <GlowOrb :size="76" />
          <h1>
            {{ hello }}，{{ auth.user?.userName }}<br />
            今天想<span class="grad">了解些什么？</span>
          </h1>
          <ChatComposer variant="hero" @send="send" />
          <div v-if="suggestions.length" class="suggest">
            <p>你可以试着问我</p>
            <div class="grid">
              <button v-for="q in suggestions" :key="q.id" class="q" type="button" @click="send(q.text)">
                <span>“{{ q.text }}”</span>
                <span class="cat">{{ q.category }}</span>
              </button>
            </div>
          </div>
        </div>

        <template v-else>
          <div ref="thread" class="thread" aria-live="polite">
            <div class="inner">
              <p v-if="chat.loadingSession" class="loading">正在加载会话…</p>
              <p v-else-if="chat.error" class="err" role="alert">{{ chat.error }}</p>
              <template v-for="m in chat.messages" :key="m.id">
                <div v-if="m.role === 'user'" class="user-msg">{{ m.content }}</div>
                <AnswerMessage v-else :message="m" :last="m.id === lastAnswerId" @follow-up="send" />
              </template>
            </div>
          </div>
          <div class="dock">
            <ChatComposer variant="dock" @send="send" />
            <p class="disclaimer"><AppIcon name="book" :size="12" />回答依据已审核入库的资料生成，重要事项请以正式通知为准</p>
          </div>
        </template>
      </section>
      <CitationDrawer />
    </div>
  </ShellLayout>
</template>

<style scoped>
.layout { position: relative; z-index: 1; flex: 1; min-height: 0; display: flex; gap: 16px; padding: 0 20px 16px; }
.col { flex: 1; min-width: 0; display: flex; flex-direction: column; min-height: 0; }
.home-hero { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 28px; padding: 24px 0 40px; overflow-y: auto; }
.home-hero > :deep(.composer) { max-width: 720px; }
h1 { margin: 0; text-align: center; font-size: 30px; line-height: 1.35; font-weight: 600; letter-spacing: 0.5px; color: #20202a; }
.grad { background: linear-gradient(90deg, #5b5bea, #8a6cf6); -webkit-background-clip: text; background-clip: text; color: transparent; }
.suggest { width: 100%; max-width: 720px; display: flex; flex-direction: column; gap: 10px; }
.suggest p { margin: 0; font-size: 12.5px; color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.q {
  display: flex; flex-direction: column; align-items: flex-start; gap: 4px; padding: 12px 14px; text-align: left;
  background: rgba(255, 255, 255, 0.7); border: 1px solid var(--line); border-radius: 12px; font-size: 13.5px; color: var(--text);
}
.q:hover { background: #fff; border-color: #cfcdf5; }
.cat { font-size: 11.5px; color: var(--muted); }
.thread { flex: 1; min-height: 0; overflow-y: auto; }
.inner { max-width: 780px; margin: 0 auto; padding: 12px 4px 24px; display: flex; flex-direction: column; gap: 22px; }
.user-msg {
  align-self: flex-end; max-width: 78%; padding: 10px 14px; border-radius: 14px 14px 4px 14px;
  background: #ecebfd; color: #26265a; font-size: 14.5px; line-height: 1.7; white-space: pre-wrap;
}
.loading, .err { margin: 40px auto; color: var(--muted); }
.err { color: var(--danger); }
.dock { max-width: 780px; width: 100%; margin: 0 auto; padding-top: 8px; }
.disclaimer { display: flex; align-items: center; justify-content: center; gap: 6px; margin: 8px 0 0; font-size: 11.5px; color: var(--muted); }
@media (max-width: 820px) {
  .layout { padding: 0 10px 10px; }
  h1 { font-size: 23px; }
  .grid { grid-template-columns: 1fr; }
  .home-hero { justify-content: flex-start; padding-top: 32px; }
}
</style>
