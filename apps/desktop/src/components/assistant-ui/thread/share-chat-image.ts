import {
  requestComposerAttachImages,
  requestComposerFocus,
  requestComposerInsertAcked,
  requestComposerSetDraft
} from '@/app/chat/composer/focus'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey, requestFreshSession } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions, rememberedSessionProfile } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $focusedRuntimeId } from '@/store/session-states'

const STYLE_ID = 'hermes-share-chat-image-styles'
export const SHARE_SELECTABLE_ATTR = 'data-share-selectable'
export const SHARE_SELECTED_ATTR = 'data-share-selected'
export const SHARE_BLOCK_ROLE_ATTR = 'data-share-block-role'
export const SHARE_CHECK_ATTR = 'data-share-check'
export const SHARE_BAR_ID = 'hermes-share-chat-bottom-bar'
export const SHARE_MODE_ATTR = 'data-share-chat-mode'

const CHECK_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`

const VISION_PROMPT_PATH_RE = /vision_analyze (?:with|using) image_url:\s*([^\n~\]\r]+?\.(?:png|jpe?g|webp|gif))/gi

const VISION_BLOCK_STRIP_RE =
  /\[The user sent an image[\s\S]*?vision_analyze (?:with|using) image_url:\s*[^\n~\]\r]+?\.(?:png|jpe?g|webp|gif)\s*(?:~)?\]\s*/gi

interface ShareCopy {
  cancel: string
  copied: string
  copyFailed: string
  copyImage: string
  download: string
  failed: string
  noMessages: string
  noSelection: string
  quotedNewChat: string
  quoteNewChat: string
  rendering: string
  saved: string
  selectAll: string
  selectedCount: (count: number) => string
  shareTip: string
  title: string
}

const COPY_EN: ShareCopy = {
  shareTip: 'Share conversation as image',
  selectAll: 'Select all',
  title: 'Share conversation',
  selectedCount: n => `${n} message${n === 1 ? '' : 's'} selected`,
  cancel: 'Cancel',
  download: 'Download locally',
  copyImage: 'Copy as image',
  quoteNewChat: 'Quote to new chat',
  rendering: 'Generating…',
  copied: 'Image copied to clipboard — ready to paste (Ctrl+V)',
  quotedNewChat: 'Opened new chat with selected messages & session link',
  saved: 'Image saved',
  noSelection: 'Select at least one message first',
  noMessages: 'No messages found in the current conversation',
  failed: 'Failed to generate image',
  copyFailed: 'Failed to copy image to clipboard'
}

const COPY_ZH: ShareCopy = {
  shareTip: '分享对话为图片',
  selectAll: '全选',
  title: '分享对话',
  selectedCount: n => `已选 ${n} 条消息`,
  cancel: '取消',
  download: '下载到本地',
  copyImage: '复制为图片',
  quoteNewChat: '引至新会话',
  rendering: '生成中…',
  copied: '图片已复制到剪贴板，可直接粘贴 (Ctrl+V)',
  quotedNewChat: '已开启新会话并附带所选对话图片与原会话链接',
  saved: '图片已保存',
  noSelection: '请先至少勾选一条消息',
  noMessages: '当前会话没有可分享的消息',
  failed: '生成分享图片失败',
  copyFailed: '复制图片到剪贴板失败'
}

export function getShareChatCopy(): ShareCopy {
  const copyLabel = translateNow('assistant.thread.copy')

  if (copyLabel === '复制' || copyLabel === '複製') {
    return COPY_ZH
  }

  const navLang = typeof navigator !== 'undefined' ? (navigator.language || '').toLowerCase() : ''

  return navLang.startsWith('zh') ? COPY_ZH : COPY_EN
}

let active = false
let busy = false
let activeViewport: HTMLElement | null = null
let selectableItems: HTMLElement[] = []
let selectedSet = new Set<HTMLElement>()
let bottomBar: HTMLDivElement | null = null
let domObserver: MutationObserver | null = null
let listenersAttached = false

export function isShareChatModeActive(): boolean {
  return active
}

function injectStyles(): void {
  if (typeof document === 'undefined' || document.getElementById(STYLE_ID)) {
    return
  }

  const style = document.createElement('style')
  style.id = STYLE_ID
  style.textContent = `
/* Selection mode on thread viewport */
[${SHARE_MODE_ATTR}="true"] [data-slot="aui_thread-content"] {
  padding-left: 3.5rem !important;
  padding-bottom: 8rem !important;
}

/* Disable sticky pinning on user messages during selection mode so user and agent blocks are cleanly separated */
[${SHARE_MODE_ATTR}="true"] [data-slot="aui_user-message-root"] {
  position: relative !important;
  top: auto !important;
  z-index: 1 !important;
}

[${SHARE_MODE_ATTR}="true"] [data-slot="aui_turn-pair"] {
  display: flex !important;
  flex-direction: column !important;
  gap: 0.85rem !important;
}

/* Distinct visual block for each selectable User or Agent message */
[${SHARE_MODE_ATTR}="true"] [${SHARE_SELECTABLE_ATTR}] {
  position: relative !important;
  cursor: pointer !important;
  border-radius: 0.75rem !important;
  padding: 0.65rem 0.85rem !important;
  margin-top: 0.25rem !important;
  margin-bottom: 0.25rem !important;
  border: 1px solid var(--ui-stroke-tertiary, rgba(255, 255, 255, 0.08)) !important;
  background-color: color-mix(in srgb, var(--color-foreground, #ffffff) 1.5%, transparent) !important;
  transition: background-color 0.14s ease, border-color 0.14s ease, box-shadow 0.14s ease;
}

[${SHARE_MODE_ATTR}="true"] [${SHARE_SELECTABLE_ATTR}]:hover {
  border-color: var(--ui-stroke-secondary, rgba(255, 255, 255, 0.22)) !important;
  background-color: color-mix(in srgb, var(--color-foreground, #ffffff) 3.5%, transparent) !important;
}

[${SHARE_MODE_ATTR}="true"] [${SHARE_SELECTABLE_ATTR}][${SHARE_SELECTED_ATTR}="true"] {
  border-color: color-mix(in srgb, var(--color-foreground, #ffffff) 38%, transparent) !important;
  background-color: color-mix(in srgb, var(--color-foreground, #ffffff) 5%, transparent) !important;
}

/* Keep user bubble right-aligned look inside its selectable block */
[${SHARE_MODE_ATTR}="true"] [${SHARE_SELECTABLE_ATTR}][${SHARE_BLOCK_ROLE_ATTR}="user"] {
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
}

/* Unclamp user messages and hide sticky-prompt-clip while in share selection mode */
[${SHARE_MODE_ATTR}="true"] .sticky-human-clamp {
  max-height: none !important;
  overflow: visible !important;
  -webkit-mask-image: none !important;
  mask-image: none !important;
}

[${SHARE_MODE_ATTR}="true"] [data-sticky-prompt-clip] {
  clip-path: none !important;
}

/* Left circular checkbox */
[${SHARE_CHECK_ATTR}] {
  position: absolute;
  left: -2.35rem;
  top: 0.85rem;
  width: 1.35rem;
  height: 1.35rem;
  border-radius: 9999px;
  border: 1.75px solid var(--ui-stroke-secondary, rgba(255, 255, 255, 0.32));
  background: transparent;
  display: grid;
  place-items: center;
  color: transparent;
  cursor: pointer;
  z-index: 45;
  transition: all 0.14s ease;
  padding: 0;
  outline: none;
}

[${SHARE_SELECTABLE_ATTR}]:hover [${SHARE_CHECK_ATTR}] {
  border-color: var(--color-foreground, #ffffff);
}

[${SHARE_SELECTABLE_ATTR}][${SHARE_SELECTED_ATTR}="true"] [${SHARE_CHECK_ATTR}] {
  background: var(--color-foreground, #ffffff);
  border-color: var(--color-foreground, #ffffff);
  color: var(--color-background, #121212);
}

/* Fixed bottom action bar */
#${SHARE_BAR_ID} {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9990;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.25rem;
  padding: 0.9rem 2rem;
  background: color-mix(in srgb, var(--color-background, #181818) 95%, #000000);
  backdrop-filter: blur(16px);
  border-top: 1px solid var(--ui-stroke-tertiary, rgba(255, 255, 255, 0.1));
  box-shadow: 0 -8px 30px rgba(0, 0, 0, 0.45);
  animation: hermes-share-bar-slide-up 0.16s cubic-bezier(0.16, 1, 0.3, 1);
  font-family: inherit;
}

@keyframes hermes-share-bar-slide-up {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

#${SHARE_BAR_ID} .share-bar-left {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

#${SHARE_BAR_ID} .share-toggle-row {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  cursor: pointer;
  user-select: none;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--color-foreground, #ededed);
}

#${SHARE_BAR_ID} .share-switch {
  position: relative;
  width: 2.25rem;
  height: 1.25rem;
  border-radius: 9999px;
  border: 1.5px solid var(--ui-stroke-secondary, rgba(255, 255, 255, 0.3));
  background: rgba(255, 255, 255, 0.08);
  transition: all 0.16s ease;
  padding: 0;
  cursor: pointer;
}

#${SHARE_BAR_ID} .share-switch[data-checked="true"] {
  background: var(--color-foreground, #ffffff);
  border-color: var(--color-foreground, #ffffff);
}

#${SHARE_BAR_ID} .share-switch-thumb {
  position: absolute;
  top: 50%;
  left: 2px;
  width: 0.85rem;
  height: 0.85rem;
  border-radius: 9999px;
  background: var(--color-foreground, #ffffff);
  transform: translateY(-50%);
  transition: transform 0.16s ease, background-color 0.16s ease;
}

#${SHARE_BAR_ID} .share-switch[data-checked="true"] .share-switch-thumb {
  transform: translate(0.95rem, -50%);
  background: var(--color-background, #141414);
}

#${SHARE_BAR_ID} .share-meta-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--color-foreground, #ededed);
  line-height: 1.25;
}

#${SHARE_BAR_ID} .share-meta-sub {
  font-size: 0.76rem;
  color: var(--color-muted-foreground, #8a8a8a);
  line-height: 1.2;
}

#${SHARE_BAR_ID} .share-bar-actions {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}

#${SHARE_BAR_ID} .share-pill-btn {
  appearance: none;
  border: 1px solid var(--ui-stroke-tertiary, rgba(255, 255, 255, 0.12));
  background: color-mix(in srgb, var(--color-foreground, #ffffff) 9%, transparent);
  color: var(--color-foreground, #ededed);
  font: inherit;
  font-size: 0.825rem;
  font-weight: 500;
  padding: 0.48rem 1.05rem;
  border-radius: 9999px;
  cursor: pointer;
  transition: background-color 0.14s ease, opacity 0.14s ease, transform 0.1s ease;
  white-space: nowrap;
}

#${SHARE_BAR_ID} .share-pill-btn:hover:not(:disabled) {
  background: color-mix(in srgb, var(--color-foreground, #ffffff) 15%, transparent);
}

#${SHARE_BAR_ID} .share-pill-btn:active:not(:disabled) {
  transform: scale(0.98);
}

#${SHARE_BAR_ID} .share-pill-btn[data-primary="true"] {
  background: var(--color-foreground, #ffffff);
  color: var(--color-background, #121212);
  border-color: transparent;
  font-weight: 600;
}

#${SHARE_BAR_ID} .share-pill-btn[data-primary="true"]:hover:not(:disabled) {
  opacity: 0.9;
}

#${SHARE_BAR_ID} .share-pill-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
`
  document.head.appendChild(style)
}

function findActiveViewport(triggerEl?: Element | null): HTMLElement | null {
  if (triggerEl && typeof triggerEl.closest === 'function') {
    const vp = triggerEl.closest<HTMLElement>('[data-slot="aui_thread-viewport"]')

    if (vp) {
      return vp
    }
  }

  return typeof document === 'undefined' ? null : document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')
}

export function collectMessageBlocks(viewport: HTMLElement): HTMLElement[] {
  const content = viewport.querySelector<HTMLElement>('[data-slot="aui_thread-content"]')

  if (!content) {
    return []
  }

  const raw = Array.from(
    content.querySelectorAll<HTMLElement>(
      '[data-slot="aui_user-message-root"], [data-slot="aui_response-group"], [data-slot="aui_assistant-message-root"]'
    )
  )

  return raw.filter(el => {
    const slot = el.getAttribute('data-slot')

    if (slot === 'aui_assistant-message-root' && el.closest('[data-slot="aui_response-group"]')) {
      return false
    }

    const cs = window.getComputedStyle(el)

    return cs.display !== 'none' && cs.visibility !== 'hidden'
  })
}

function getBlockRole(el: Element): 'assistant' | 'user' {
  if (el.getAttribute('data-slot') === 'aui_user-message-root' || el.getAttribute('data-role') === 'user') {
    return 'user'
  }

  return 'assistant'
}

export function findTurnPeers(clickedTarget: Element | null | undefined, allBlocks: HTMLElement[]): HTMLElement[] {
  if (!clickedTarget) {
    return allBlocks.slice(-2)
  }

  let idx = allBlocks.indexOf(clickedTarget as HTMLElement)

  if (idx === -1) {
    idx = allBlocks.findIndex(b => b.contains(clickedTarget) || clickedTarget.contains(b))
  }

  if (idx === -1) {
    return allBlocks.slice(-2)
  }

  const picked: HTMLElement[] = [allBlocks[idx]!]

  for (let i = idx - 1; i >= 0; i--) {
    const prev = allBlocks[i]!
    picked.unshift(prev)

    if (getBlockRole(prev) === 'user') {
      break
    }
  }

  return picked
}

function syncSelectableBlocks(): void {
  if (!active || !activeViewport) {
    return
  }

  const virtualRows = activeViewport.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]')

  for (const row of virtualRows) {
    row.style.contentVisibility = 'visible'
  }

  const blocks = collectMessageBlocks(activeViewport)
  const wasAllSelected = selectedSet.size === selectableItems.length && selectableItems.length > 0

  selectableItems = blocks

  blocks.forEach((el, index) => {
    const role = getBlockRole(el)
    el.setAttribute(SHARE_SELECTABLE_ATTR, String(index))
    el.setAttribute(SHARE_BLOCK_ROLE_ATTR, role)

    if (wasAllSelected) {
      selectedSet.add(el)
    }

    const isSel = selectedSet.has(el)
    el.setAttribute(SHARE_SELECTED_ATTR, isSel ? 'true' : 'false')

    if (!el.querySelector(`[${SHARE_CHECK_ATTR}]`)) {
      const checkBtn = document.createElement('button')
      checkBtn.type = 'button'
      checkBtn.setAttribute(SHARE_CHECK_ATTR, String(index))
      checkBtn.innerHTML = CHECK_ICON_SVG
      checkBtn.addEventListener('click', event => {
        event.preventDefault()
        event.stopPropagation()
        toggleBlock(el)
      })
      el.appendChild(checkBtn)
    }
  })

  updateBottomBar()
}

function ensureGlobalListeners(): void {
  if (listenersAttached || typeof document === 'undefined') {
    return
  }

  listenersAttached = true
  document.addEventListener('click', onViewportClickCapture, true)
  document.addEventListener('keydown', onKeyDown, true)
}

function removeGlobalListeners(): void {
  if (!listenersAttached || typeof document === 'undefined') {
    return
  }

  listenersAttached = false
  document.removeEventListener('click', onViewportClickCapture, true)
  document.removeEventListener('keydown', onKeyDown, true)
}

export function enterShareChatMode(clickedTarget?: Element | null): void {
  const copy = getShareChatCopy()

  if (active) {
    exitShareChatMode()
  }

  injectStyles()

  const viewport = findActiveViewport(clickedTarget)

  if (!viewport) {
    notify({ kind: 'info', message: copy.noMessages })

    return
  }

  const virtualRows = viewport.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]')

  for (const row of virtualRows) {
    row.style.contentVisibility = 'visible'
  }

  const blocks = collectMessageBlocks(viewport)

  if (blocks.length === 0) {
    notify({ kind: 'info', message: copy.noMessages })

    return
  }

  active = true
  activeViewport = viewport
  selectableItems = blocks
  selectedSet = new Set(findTurnPeers(clickedTarget, blocks))

  viewport.setAttribute(SHARE_MODE_ATTR, 'true')

  blocks.forEach((el, index) => {
    const role = getBlockRole(el)
    el.setAttribute(SHARE_SELECTABLE_ATTR, String(index))
    el.setAttribute(SHARE_BLOCK_ROLE_ATTR, role)
    const isSel = selectedSet.has(el)
    el.setAttribute(SHARE_SELECTED_ATTR, isSel ? 'true' : 'false')

    const oldCheck = el.querySelector(`[${SHARE_CHECK_ATTR}]`)

    if (oldCheck) {
      oldCheck.remove()
    }

    const checkBtn = document.createElement('button')
    checkBtn.type = 'button'
    checkBtn.setAttribute(SHARE_CHECK_ATTR, String(index))
    checkBtn.innerHTML = CHECK_ICON_SVG
    checkBtn.addEventListener('click', event => {
      event.preventDefault()
      event.stopPropagation()
      toggleBlock(el)
    })

    el.appendChild(checkBtn)
  })

  let syncTimer: ReturnType<typeof setTimeout> | null = null

  domObserver = new MutationObserver(() => {
    if (active && !busy) {
      if (syncTimer) {
        clearTimeout(syncTimer)
      }

      syncTimer = setTimeout(() => {
        syncTimer = null
        syncSelectableBlocks()
      }, 120)
    }
  })

  domObserver.observe(viewport, {
    childList: true,
    subtree: true
  })

  ensureGlobalListeners()
  mountBottomBar()
  updateBottomBar()
}

export function toggleBlock(el: HTMLElement): void {
  if (busy) {
    return
  }

  if (selectedSet.has(el)) {
    selectedSet.delete(el)
    el.setAttribute(SHARE_SELECTED_ATTR, 'false')
  } else {
    selectedSet.add(el)
    el.setAttribute(SHARE_SELECTED_ATTR, 'true')
  }

  updateBottomBar()
}

export function toggleSelectAll(): void {
  if (busy) {
    return
  }

  syncSelectableBlocks()
  const allSelected = selectedSet.size === selectableItems.length && selectableItems.length > 0

  if (allSelected) {
    selectedSet.clear()
    selectableItems.forEach(el => el.setAttribute(SHARE_SELECTED_ATTR, 'false'))
  } else {
    selectableItems.forEach(el => {
      selectedSet.add(el)
      el.setAttribute(SHARE_SELECTED_ATTR, 'true')
    })
  }

  updateBottomBar()
}

export function exitShareChatMode(): void {
  active = false
  busy = false

  if (domObserver) {
    domObserver.disconnect()
    domObserver = null
  }

  removeGlobalListeners()

  if (activeViewport) {
    activeViewport.removeAttribute(SHARE_MODE_ATTR)
    const virtualRows = activeViewport.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]')

    for (const row of virtualRows) {
      row.style.contentVisibility = ''
    }
  }

  for (const el of selectableItems) {
    el.removeAttribute(SHARE_SELECTABLE_ATTR)
    el.removeAttribute(SHARE_SELECTED_ATTR)
    el.removeAttribute(SHARE_BLOCK_ROLE_ATTR)
    const chk = el.querySelector(`[${SHARE_CHECK_ATTR}]`)

    if (chk) {
      chk.remove()
    }
  }

  selectableItems = []
  selectedSet.clear()
  activeViewport = null

  if (bottomBar) {
    bottomBar.remove()
    bottomBar = null
  }
}

function onViewportClickCapture(event: MouseEvent): void {
  if (!active || busy) {
    return
  }

  const target = event.target instanceof Element ? event.target : null

  if (!target) {
    return
  }

  if (target.closest(`#${SHARE_BAR_ID}`)) {
    return
  }

  if (target.closest('[data-conversation-scaffold] button, details summary')) {
    return
  }

  const block = target.closest<HTMLElement>(`[${SHARE_SELECTABLE_ATTR}]`)

  if (block && activeViewport && activeViewport.contains(block)) {
    event.preventDefault()
    event.stopPropagation()
    toggleBlock(block)
  }
}

function onKeyDown(event: KeyboardEvent): void {
  if (active && event.key === 'Escape') {
    event.preventDefault()
    event.stopPropagation()
    exitShareChatMode()
  }
}

function mountBottomBar(): void {
  const copy = getShareChatCopy()

  if (bottomBar) {
    bottomBar.remove()
  }

  const bar = document.createElement('div')
  bar.id = SHARE_BAR_ID
  bar.innerHTML = `
    <div class="share-bar-left">
      <div class="share-toggle-row" data-role="select-all-row">
        <button type="button" class="share-switch" data-role="select-all-switch" aria-label="${copy.selectAll}">
          <span class="share-switch-thumb"></span>
        </button>
        <span>${copy.selectAll}</span>
      </div>
      <div class="share-meta-title">${copy.title}</div>
      <div class="share-meta-sub" data-role="selected-count"></div>
    </div>
    <div class="share-bar-actions">
      <button type="button" class="share-pill-btn" data-role="cancel-btn">${copy.cancel}</button>
      <button type="button" class="share-pill-btn" data-role="download-btn">${copy.download}</button>
      <button type="button" class="share-pill-btn" data-role="quote-new-btn">${copy.quoteNewChat}</button>
      <button type="button" class="share-pill-btn" data-primary="true" data-role="copy-btn">${copy.copyImage}</button>
    </div>
  `

  bar.querySelector('[data-role="select-all-row"]')?.addEventListener('click', toggleSelectAll)
  bar.querySelector('[data-role="cancel-btn"]')?.addEventListener('click', exitShareChatMode)
  bar.querySelector('[data-role="download-btn"]')?.addEventListener('click', () => void runShareExport('download'))
  bar.querySelector('[data-role="quote-new-btn"]')?.addEventListener('click', () => void runShareExport('quote-new'))
  bar.querySelector('[data-role="copy-btn"]')?.addEventListener('click', () => void runShareExport('copy'))

  document.body.appendChild(bar)
  bottomBar = bar
}

function updateBottomBar(): void {
  if (!bottomBar) {
    return
  }

  const copy = getShareChatCopy()
  const count = selectedSet.size
  const allSelected = count === selectableItems.length && selectableItems.length > 0

  const sw = bottomBar.querySelector('[data-role="select-all-switch"]')

  if (sw) {
    sw.setAttribute('data-checked', allSelected ? 'true' : 'false')
  }

  const countEl = bottomBar.querySelector('[data-role="selected-count"]')

  if (countEl) {
    countEl.textContent = copy.selectedCount(count)
  }

  const downloadBtn = bottomBar.querySelector<HTMLButtonElement>('[data-role="download-btn"]')
  const quoteNewBtn = bottomBar.querySelector<HTMLButtonElement>('[data-role="quote-new-btn"]')
  const copyBtn = bottomBar.querySelector<HTMLButtonElement>('[data-role="copy-btn"]')

  if (downloadBtn) {
    downloadBtn.disabled = busy || count === 0
    downloadBtn.textContent = busy ? copy.rendering : copy.download
  }

  if (quoteNewBtn) {
    quoteNewBtn.disabled = busy || count === 0
    quoteNewBtn.textContent = busy ? copy.rendering : copy.quoteNewChat
  }

  if (copyBtn) {
    copyBtn.disabled = busy || count === 0
    copyBtn.textContent = busy ? copy.rendering : copy.copyImage
  }
}

async function imageToDataUrl(imgEl: HTMLImageElement): Promise<string> {
  const src = imgEl.currentSrc || imgEl.src

  if (!src || src.startsWith('data:')) {
    return src
  }

  try {
    const res = await fetch(src)
    const blob = await res.blob()

    return await new Promise<string>(resolve => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(typeof reader.result === 'string' ? reader.result : src)
      reader.onerror = () => resolve(src)
      reader.readAsDataURL(blob)
    })
  } catch {
    return src
  }
}

function resolveThemeColors(): { bg: string; fg: string; font: string } {
  const contentContainer = activeViewport?.querySelector<HTMLElement>('[data-slot="aui_thread-content"]')
  const chatSurface = activeViewport?.closest<HTMLElement>('[data-chat-surface]')
  const rootCs = window.getComputedStyle(document.documentElement)
  const bodyCs = window.getComputedStyle(document.body)

  const candidates = [
    chatSurface ? window.getComputedStyle(chatSurface).backgroundColor : '',
    activeViewport ? window.getComputedStyle(activeViewport).backgroundColor : '',
    rootCs.getPropertyValue('--ui-chat-surface-background').trim(),
    rootCs.getPropertyValue('--dt-background').trim(),
    rootCs.getPropertyValue('--color-background').trim(),
    bodyCs.backgroundColor
  ]

  let bg = '#141414'

  for (const c of candidates) {
    if (c && c !== 'transparent' && c !== 'rgba(0, 0, 0, 0)') {
      bg = c

      break
    }
  }

  const fg =
    (contentContainer ? window.getComputedStyle(contentContainer).color : '') ||
    rootCs.getPropertyValue('--ui-text-primary').trim() ||
    bodyCs.color ||
    '#ededed'

  const font =
    (contentContainer ? window.getComputedStyle(contentContainer).fontFamily : '') ||
    bodyCs.fontFamily ||
    'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'

  return { bg, fg, font }
}

export function collectDocumentCssText(): string {
  const chunks: string[] = []

  const htmlStyle = document.documentElement.getAttribute('style') || ''
  const bodyStyle = document.body.getAttribute('style') || ''

  if (htmlStyle || bodyStyle) {
    chunks.push(`:root, [data-share-stage] { ${htmlStyle} } body, [data-share-stage] { ${bodyStyle} }`)
  }

  for (const sheet of Array.from(document.styleSheets)) {
    if (sheet.ownerNode && (sheet.ownerNode as Element).id === STYLE_ID) {
      continue
    }

    try {
      const rules = sheet.cssRules || []

      for (const rule of Array.from(rules)) {
        if (rule.type === CSSRule.FONT_FACE_RULE || rule.type === CSSRule.IMPORT_RULE) {
          continue
        }

        const cssText = rule.cssText
          .replace(/:root\.dark\b/g, ':root.dark, [data-share-stage].dark')
          .replace(/:root\[/g, '[data-share-stage][')
          .replace(/:root\b/g, ':root, [data-share-stage]')

        chunks.push(cssText)
      }
    } catch {
      // Cross-origin stylesheet, ignore
    }
  }

  chunks.push(`
    [data-share-stage],
    [data-share-stage] * {
      content-visibility: visible !important;
      contain-intrinsic-size: auto !important;
    }
    [data-share-stage] [data-sticky-prompt-clip] {
      clip-path: none !important;
    }
    [data-share-stage] [data-slot="aui_user-message-root"] {
      position: relative !important;
      top: auto !important;
      margin-left: 0 !important;
      margin-right: 0 !important;
      width: 100% !important;
      padding-left: 0 !important;
      padding-right: 0 !important;
    }
    [data-share-stage] .sticky-human-clamp {
      max-height: none !important;
      overflow: visible !important;
      -webkit-mask-image: none !important;
      mask-image: none !important;
    }
    [data-share-stage] [data-slot="aui_user-message-text"],
    [data-share-stage] [data-slot="aui_user-inline-text"] {
      overflow-wrap: anywhere !important;
      word-break: break-word !important;
      white-space: pre-wrap !important;
    }
    [data-share-stage] [data-slot="aui_msg-actions"],
    [data-share-stage] [data-slot="aui_msg-reactions"]:not([data-reacted]),
    [data-share-stage] .checkpoint-container,
    [data-share-stage] .codicon,
    [data-share-stage] [${SHARE_CHECK_ATTR}] {
      display: none !important;
    }
    [data-share-stage] code,
    [data-share-stage] pre {
      font-family: Consolas, "Cascadia Code", "Courier New", monospace !important;
    }
  `)

  return chunks.join('\n')
}

async function localPathToDataUrl(filePath: string): Promise<null | string> {
  const clean = String(filePath || '').trim()

  if (!clean) {
    return null
  }

  try {
    if (window.hermesDesktop && typeof window.hermesDesktop.readFileDataUrl === 'function') {
      const res = await window.hermesDesktop.readFileDataUrl(clean)

      if (typeof res === 'string' && res.startsWith('data:image/')) {
        return res
      }

      if (
        res &&
        typeof res === 'object' &&
        'dataUrl' in res &&
        typeof (res as { dataUrl?: unknown }).dataUrl === 'string' &&
        ((res as { dataUrl: string }).dataUrl).startsWith('data:image/')
      ) {
        return (res as { dataUrl: string }).dataUrl
      }
    }
  } catch {
    // ignore read error
  }

  return null
}

export async function prepareCloneForSnapshot(sourceNode: HTMLElement): Promise<HTMLElement> {
  const clone = sourceNode.cloneNode(true) as HTMLElement

  for (const el of [clone, ...Array.from(clone.querySelectorAll<HTMLElement>('[style]'))]) {
    if (el.style) {
      if (el.style.contentVisibility) {
        el.style.contentVisibility = 'visible'
      }

      if (el.style.containIntrinsicSize) {
        el.style.containIntrinsicSize = 'auto'
      }

      if (el.style.clipPath) {
        el.style.clipPath = 'none'
      }
    }
  }

  let attachmentClone: HTMLElement | null = null

  const hasExistingAttachmentImgs = Boolean(
    sourceNode.querySelector('[data-slot="aui_embedded-images"] img, [data-slot="aui_directive-image"] img') ||
      (sourceNode.nextElementSibling &&
        !sourceNode.nextElementSibling.hasAttribute('data-slot') &&
        sourceNode.nextElementSibling.querySelector('img'))
  )

  if (
    sourceNode.getAttribute('data-slot') === 'aui_user-message-root' &&
    sourceNode.nextElementSibling &&
    !sourceNode.nextElementSibling.hasAttribute('data-slot') &&
    !sourceNode.nextElementSibling.hasAttribute(SHARE_SELECTABLE_ATTR)
  ) {
    attachmentClone = sourceNode.nextElementSibling.cloneNode(true) as HTMLElement
    const srcAttImgs = Array.from(sourceNode.nextElementSibling.querySelectorAll('img'))
    const dstAttImgs = Array.from(attachmentClone.querySelectorAll('img'))

    for (let i = 0; i < srcAttImgs.length && i < dstAttImgs.length; i++) {
      const dataUrl = await imageToDataUrl(srcAttImgs[i]!)

      if (dataUrl) {
        dstAttImgs[i]!.setAttribute('src', dataUrl)
      }
    }

    attachmentClone.querySelectorAll('.opacity-0.pointer-events-none, button.opacity-0').forEach(n => n.remove())
  }

  if (sourceNode.getAttribute('data-slot') === 'aui_user-message-root' && !hasExistingAttachmentImgs) {
    const rawText = sourceNode.textContent || ''
    const matches = Array.from(rawText.matchAll(VISION_PROMPT_PATH_RE)).map(m => (m[1] || '').trim())

    if (matches.length > 0) {
      const dataUrls = (await Promise.all(matches.map(p => localPathToDataUrl(p)))).filter(
        (u): u is string => Boolean(u)
      )

      if (dataUrls.length > 0) {
        if (!attachmentClone) {
          attachmentClone = document.createElement('div')
          attachmentClone.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end;margin-top:6px'
        }

        for (const url of dataUrls) {
          const thumb = document.createElement('img')
          thumb.src = url
          thumb.style.cssText =
            'max-height:180px;max-width:48%;border-radius:8px;border:1px solid rgba(255,255,255,0.14);object-fit:contain'
          attachmentClone.appendChild(thumb)
        }

        const textSpan =
          clone.querySelector('[data-slot="aui_directive-text"]') ||
          clone.querySelector('[data-slot="aui_user-inline-text"]') ||
          clone.querySelector('[data-slot="aui_user-message-text"]')

        if (textSpan && textSpan.textContent) {
          const cleaned = textSpan.textContent.replace(VISION_BLOCK_STRIP_RE, '').trim()

          if (cleaned) {
            textSpan.textContent = cleaned
          } else {
            const bubbleBtn = clone.querySelector<HTMLElement>('.composer-human-message')

            if (bubbleBtn) {
              bubbleBtn.style.display = 'none'
            }
          }
        }
      }
    }
  }

  clone.removeAttribute(SHARE_SELECTABLE_ATTR)
  clone.removeAttribute(SHARE_SELECTED_ATTR)
  clone.removeAttribute(SHARE_BLOCK_ROLE_ATTR)

  const toRemove = clone.querySelectorAll(
    `[${SHARE_CHECK_ATTR}], [data-slot="aui_msg-actions"], [data-slot="aui_message-streaming-marker"], .checkpoint-container, [data-slot="aui_msg-reactions"]:not([data-reacted])`
  )

  toRemove.forEach(n => n.remove())
  clone.querySelectorAll('.opacity-0.pointer-events-none, button.opacity-0').forEach(n => n.remove())

  const srcImgs = Array.from(sourceNode.querySelectorAll('img'))
  const dstImgs = Array.from(clone.querySelectorAll('img'))

  for (let i = 0; i < srcImgs.length && i < dstImgs.length; i++) {
    const dataUrl = await imageToDataUrl(srcImgs[i]!)

    if (dataUrl) {
      dstImgs[i]!.setAttribute('src', dataUrl)
    }
  }

  const srcCanvases = Array.from(sourceNode.querySelectorAll('canvas'))
  const dstCanvases = Array.from(clone.querySelectorAll('canvas'))

  for (let i = 0; i < srcCanvases.length && i < dstCanvases.length; i++) {
    try {
      const img = document.createElement('img')
      img.src = srcCanvases[i]!.toDataURL('image/png')
      img.className = dstCanvases[i]!.className
      img.style.cssText = dstCanvases[i]!.style.cssText
      dstCanvases[i]!.replaceWith(img)
    } catch {
      // ignore tainted canvas
    }
  }

  const srcSvgs = Array.from(sourceNode.querySelectorAll('svg'))
  const dstSvgs = Array.from(clone.querySelectorAll('svg'))

  for (let i = 0; i < srcSvgs.length && i < dstSvgs.length; i++) {
    const rect = srcSvgs[i]!.getBoundingClientRect()
    const cs = window.getComputedStyle(srcSvgs[i]!)

    if (rect.width > 0 && rect.height > 0) {
      dstSvgs[i]!.setAttribute('width', String(rect.width))
      dstSvgs[i]!.setAttribute('height', String(rect.height))
    }

    if (cs.color) {
      dstSvgs[i]!.style.color = cs.color
    }
  }

  if (attachmentClone) {
    const wrap = document.createElement('div')
    wrap.style.cssText = 'display:flex;flex-direction:column;gap:0.5rem;width:100%'
    wrap.appendChild(clone)
    wrap.appendChild(attachmentClone)

    return wrap
  }

  return clone
}

export async function renderSelectedToDataUrl(): Promise<{ canvas: HTMLCanvasElement; dataUrl: string }> {
  const copy = getShareChatCopy()
  const ordered = selectableItems.filter(el => selectedSet.has(el))

  if (ordered.length === 0) {
    throw new Error(copy.noSelection)
  }

  const { bg, fg, font } = resolveThemeColors()
  const cssText = collectDocumentCssText()

  const contentContainer = activeViewport?.querySelector<HTMLElement>('[data-slot="aui_thread-content"]')
  let innerWidth = 720

  if (contentContainer) {
    const cs = window.getComputedStyle(contentContainer)
    const pl = parseFloat(cs.paddingLeft) || 0
    const pr = parseFloat(cs.paddingRight) || 0
    const w = Math.round(contentContainer.getBoundingClientRect().width - pl - pr)

    if (w > 200) {
      innerWidth = w
    }
  }

  const padX = 32
  const padY = 28
  const totalCardWidth = innerWidth + padX * 2

  const rootWrapper = document.createElement('div')
  rootWrapper.className = `${document.documentElement.className || ''} ${document.body.className || ''}`.trim()

  for (const attr of Array.from(document.documentElement.attributes)) {
    if (attr.name.startsWith('data-')) {
      rootWrapper.setAttribute(attr.name, attr.value)
    }
  }

  rootWrapper.setAttribute('data-share-stage', 'true')
  rootWrapper.style.cssText = [
    'position:fixed',
    'left:-99999px',
    'top:0',
    `width:${totalCardWidth}px`,
    `padding:${padY}px ${padX}px`,
    'box-sizing:border-box',
    `background:${bg}`,
    `color:${fg}`,
    `font-family:${font}`,
    'z-index:-1'
  ].join(';')

  const threadCol = document.createElement('div')
  threadCol.className = contentContainer ? contentContainer.className : 'flex w-full min-w-0 flex-col'
  threadCol.setAttribute('data-slot', 'aui_thread-content')
  threadCol.style.cssText = [
    `width:${innerWidth}px`,
    `max-width:${innerWidth}px`,
    'padding:0',
    'margin:0',
    'box-sizing:border-box',
    'display:flex',
    'flex-direction:column',
    'gap:1.25rem'
  ].join(';')

  for (const block of ordered) {
    const cloned = await prepareCloneForSnapshot(block)
    threadCol.appendChild(cloned)
  }

  const styleEl = document.createElement('style')
  styleEl.textContent = cssText
  rootWrapper.appendChild(styleEl)
  rootWrapper.appendChild(threadCol)
  document.body.appendChild(rootWrapper)

  const stageImgs = Array.from(rootWrapper.querySelectorAll('img'))

  if (stageImgs.length > 0) {
    await Promise.all(
      stageImgs.map(
        im =>
          new Promise<void>(res => {
            if (im.complete) {
              res()

              return
            }

            im.onload = () => res()
            im.onerror = () => res()
          })
      )
    )
  }

  const totalWidth = Math.ceil(rootWrapper.offsetWidth || totalCardWidth)
  const totalHeight = Math.ceil(Math.max(rootWrapper.offsetHeight, rootWrapper.scrollHeight) || 400)

  rootWrapper.style.position = 'static'
  rootWrapper.style.left = '0'
  rootWrapper.style.top = '0'
  rootWrapper.style.zIndex = 'auto'
  rootWrapper.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml')

  for (const node of rootWrapper.querySelectorAll('[xmlns]')) {
    if (node !== rootWrapper && node.tagName.toLowerCase() !== 'svg') {
      node.removeAttribute('xmlns')
    }
  }

  const xhtml = new XMLSerializer().serializeToString(rootWrapper)
  rootWrapper.remove()

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${totalWidth}" height="${totalHeight}" viewBox="0 0 ${totalWidth} ${totalHeight}">
    <foreignObject x="0" y="0" width="100%" height="100%">
      ${xhtml}
    </foreignObject>
  </svg>`

  const desiredScale = Math.min(2.5, Math.max(2, window.devicePixelRatio || 2))
  const maxSafeDim = 15800

  const scale = Math.min(
    desiredScale,
    maxSafeDim / Math.max(1, totalHeight),
    maxSafeDim / Math.max(1, totalWidth)
  )

  return await new Promise<{ canvas: HTMLCanvasElement; dataUrl: string }>((resolve, reject) => {
    const img = new Image()

    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = Math.round(totalWidth * scale)
        canvas.height = Math.round(totalHeight * scale)
        const ctx2d = canvas.getContext('2d')

        if (!ctx2d) {
          reject(new Error('Canvas 2D context unavailable'))

          return
        }

        ctx2d.scale(scale, scale)
        ctx2d.fillStyle = bg
        ctx2d.fillRect(0, 0, totalWidth, totalHeight)
        ctx2d.drawImage(img, 0, 0, totalWidth, totalHeight)
        resolve({
          dataUrl: canvas.toDataURL('image/png'),
          canvas
        })
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)))
      }
    }

    img.onerror = () => reject(new Error('SVG rasterization failed'))
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
  })
}

export async function copyImageToSystemClipboard(dataUrl: string, canvas?: HTMLCanvasElement): Promise<boolean> {
  let copied = false

  try {
    const holder = document.createElement('div')
    holder.contentEditable = 'true'
    holder.setAttribute('data-share-copy-holder', 'true')
    holder.style.cssText = [
      'position:fixed',
      'left:-9999px',
      'top:0',
      'width:10px',
      'height:10px',
      'opacity:0.01',
      'overflow:hidden',
      'pointer-events:none',
      'z-index:-1'
    ].join(';')

    const img = document.createElement('img')
    img.src = dataUrl
    img.alt = 'Shared Conversation'
    holder.appendChild(img)
    document.body.appendChild(holder)

    await new Promise<void>(resolve => {
      if (img.complete && img.naturalWidth > 0) {
        resolve()
      } else {
        img.onload = () => resolve()
        img.onerror = () => resolve()
      }
    })

    const onCopyCapture = (event: ClipboardEvent) => {
      event.stopImmediatePropagation()

      if (event.clipboardData) {
        event.clipboardData.setData('text/plain', '')
        event.clipboardData.setData('text/html', `<img src="${dataUrl}" alt="Shared Conversation" />`)
        event.preventDefault()
        copied = true
      }
    }

    window.addEventListener('copy', onCopyCapture, true)

    try {
      const sel = window.getSelection()

      if (sel) {
        sel.removeAllRanges()
        const range = document.createRange()
        range.selectNode(img)
        sel.addRange(range)
      }

      const execOk = document.execCommand('copy')

      if (execOk) {
        copied = true
      }

      if (sel) {
        sel.removeAllRanges()
      }
    } finally {
      window.removeEventListener('copy', onCopyCapture, true)
      holder.remove()
    }
  } catch (err) {
    console.warn('[share-chat-image] execCommand copy path failed', err)
  }

  if (
    canvas &&
    navigator.clipboard &&
    typeof navigator.clipboard.write === 'function' &&
    typeof ClipboardItem !== 'undefined'
  ) {
    try {
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(b => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png')
      })

      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
      copied = true
    } catch {
      // Ignored if permission denied; Strategy 1 already populated clipboardData
    }
  }

  return copied
}

export async function downloadDataUrl(dataUrl: string): Promise<boolean> {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const filename = `hermes-chat-${stamp}.png`

  if (window.hermesDesktop && typeof window.hermesDesktop.saveImageFromUrl === 'function') {
    return await window.hermesDesktop.saveImageFromUrl(dataUrl)
  }

  const link = document.createElement('a')
  link.href = dataUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()

  return true
}

export function originSessionRef(): string {
  const sid =
    $focusedStoredSessionId.get() ||
    $selectedStoredSessionId.get() ||
    $focusedRuntimeId.get() ||
    $activeSessionId.get() ||
    ''

  const cleanId = String(sid || '').trim()

  if (!cleanId || cleanId === 'new' || cleanId.startsWith('__new__')) {
    return ''
  }

  const profile =
    normalizeProfileKey(rememberedSessionProfile($sessions.get(), cleanId, $activeGatewayProfile.get())) || 'default'

  const target = cleanId.includes('/') ? cleanId : `${profile}/${cleanId}`

  return `@session:${formatRefValue(target)}`
}

export function dataUrlToBlob(dataUrl: string): Blob | null {
  const match = /^data:([\w./+-]+);base64,(.*)$/i.exec(String(dataUrl || '').trim())

  if (!match || typeof atob !== 'function' || typeof Blob === 'undefined') {
    return null
  }

  try {
    const bytes = atob(match[2] || '')
    const buffer = new Uint8Array(bytes.length)

    for (let i = 0; i < bytes.length; i += 1) {
      buffer[i] = bytes.charCodeAt(i)
    }

    return new Blob([buffer], { type: match[1] || 'image/png' })
  } catch {
    return null
  }
}

export async function quoteSelectedToNewChat(dataUrl: string, canvas?: HTMLCanvasElement): Promise<void> {
  const copy = getShareChatCopy()
  const sessionRef = originSessionRef()
  let blob = dataUrlToBlob(dataUrl)

  if (!blob && canvas && typeof canvas.toBlob === 'function') {
    blob = await new Promise<Blob | null>(resolve => canvas.toBlob(b => resolve(b), 'image/png'))
  }

  exitShareChatMode()
  requestFreshSession()

  await new Promise<void>(resolve => setTimeout(resolve, 32))

  if (blob) {
    requestComposerAttachImages([blob], { target: 'main' })
  }

  if (sessionRef) {
    const draftText = `${sessionRef}\n`

    const written =
      (await requestComposerSetDraft([], draftText, { active: true })) ||
      (await requestComposerInsertAcked(sessionRef, { mode: 'block', target: 'main' }))

    if (written) {
      requestComposerFocus('main')
    }
  } else {
    requestComposerFocus('main')
  }

  notify({ kind: 'success', message: copy.quotedNewChat })
}

export async function runShareExport(action: 'copy' | 'download' | 'quote-new'): Promise<void> {
  const copy = getShareChatCopy()

  if (busy) {
    return
  }

  if (selectedSet.size === 0) {
    notify({ kind: 'info', message: copy.noSelection })

    return
  }

  busy = true
  updateBottomBar()

  try {
    const { dataUrl, canvas } = await renderSelectedToDataUrl()

    if (action === 'quote-new') {
      await quoteSelectedToNewChat(dataUrl, canvas)
    } else if (action === 'copy') {
      const ok = await copyImageToSystemClipboard(dataUrl, canvas)

      if (ok) {
        notify({ kind: 'success', message: copy.copied })
        exitShareChatMode()
      } else {
        notify({ kind: 'error', message: copy.copyFailed })
      }
    } else {
      const saved = await downloadDataUrl(dataUrl)

      if (saved) {
        notify({ kind: 'success', message: copy.saved })
        exitShareChatMode()
      }
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    notify({ kind: 'error', message: `${copy.failed}: ${detail}` })
  } finally {
    busy = false
    updateBottomBar()
  }
}
