/**
 * Hermes 提示词优化插件 (prompt-optimizer)
 * ==========================================
 *
 * 功能
 *   - 在会话输入框右侧（模型选择下拉菜单左侧）增加「优化提示词」按钮。
 *   - 点击后使用「当前会话已选择的模型」对输入框中的提示词做无状态单次优化
 *     （gateway RPC: llm.oneshot，不写入会话历史、不破坏 prompt cache）。
 *   - 优化完成后自动将结果填充回输入框，并提供一键「撤回」恢复原始文本。
 *
 * 架构（低耦合 / 更新不覆盖 / 一键导入）
 *   - 源码主副本 = 本仓库根目录的 plugin.js（独立于 Hermes 核心代码，更新不覆盖）
 *   - 运行时部署:  <hermes home>/desktop-plugins/prompt-optimizer/plugin.js
 *     （桌面 App 自动 watch 该目录，文件落地数秒内自动加载/热更新）
 *   - 更新后重新导入:  运行仓库内 install.ps1 / install.sh（幂等）或手动复制 plugin.js
 *
 * 实现要点
 *   - 只允许 import @hermes/plugin-sdk / react / react/jsx-runtime（runtime
 *     加载器的 specifier rewrite 白名单），因此本文件为纯 ESM + jsx() 调用。
 *   - 输入框读写走 composer 的 contentEditable（data-slot="composer-rich-input"）。
 *     composer 以 DOM 为唯一数据源（onInput -> flushEditorToDraft），因此
 *     写入 DOM 后派发 InputEvent 即可驱动官方状态同步，无需 React 内部 hack。
 *   - 会话绑定：点击时同步捕获编辑器引用与会话 ID，异步返回后写入发起时的
 *     编辑器（绝不重新 findEditor），切换/关闭会话不会串写其他会话输入框。
 *   - 序列化规则与 apps/desktop/src/app/chat/composer/rich-editor.ts 的
 *     composerPlainText / composerHtml 保持一致（<br> 换行、data-ref-text chip）。
 *   - 性能：调用本插件后端 REST（plugin_api.py → run_oneshot，继承会话模型），
 *     前端 timeoutMs 300s 兜底，规避桌面 App RPC 层 30s 硬超时。
 */

import {
  Button,
  COMPOSER_AREAS,
  cn,
  haptic,
  host,
  icons,
  Tip,
  usePluginI18n,
  useValue
} from '@hermes/plugin-sdk'
import { jsx } from 'react/jsx-runtime'
import { useCallback, useEffect, useRef, useState } from 'react'

const ID = 'prompt-optimizer'
const RICH_INPUT_SLOT = 'composer-rich-input'

// 模块级 REST door:register 时由 ctx 注入（SDK 的 host 对象无 rest，官方
// 插件统一从 ctx.rest 取，作用域固定为 /api/plugins/<id>）。
let restRequest = null

// 优化引擎提示词模板 —— 融合 Qoder 官方 4 大增强维度（需求明确化/场景上下文化/
// 约束条件完善化/结构化输出）+ Trae 任务叙事式组织 + 开源实现要点
// (jodykwong/Prompt-Enhancement MIT: 意图保真/具体化/验证标准/长度控制)。
const OPTIMIZE_INSTRUCTIONS = `你是一位资深提示词优化专家，擅长把开发者口语化的需求改写为可直接执行的高质量任务提示词。

【核心原则】
1. 意图保真：完整保留原文的目标、问题、专有名词与领域术语（如系统名、模块名、业务术语），绝不编造、替换或丢失关键信息。
2. 任务叙事：以"给 AI 助手派活"的口吻组织内容——先一句话说清要解决的问题，再按任务逻辑分块展开（可编号、分点），最后给出完成标准。结构跟随内容自然组织，绝不使用【任务目标】【问题说明】【约束条件】【输出要求】这类文档式标题硬套模板。
3. 细节补全：把模糊表述转化为具体行动步骤；合理补全易遗漏的关键细节（如错误处理、边界条件、性能要求、兼容性、安全与代码规范），但只补与任务直接相关的合理推断，不添加无关需求。
4. 验证标准：把原文隐含的验收要求（"确保""修复后""需要"等）显式写成可检查的完成标准。
5. 语言与长度：与用户输入保持同一语言；短输入补足细节、长输入提炼组织；删冗余、不注水，长度与任务复杂度相称。
6. 长文本提速：当输入内容较长时，优先保证结构完整与关键信息齐全，直接输出优化结果，避免逐句复述原文或无谓扩写，控制输出长度以缩短响应时间。

【输出要求】
直接输出优化后的提示词正文，不要任何解释、前言、后记、标题或 Markdown 代码块包装。`

// ---------------------------------------------------------------------------
// 编辑器读写 —— 与 rich-editor.ts 的序列化规则保持一致
// ---------------------------------------------------------------------------

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** 对齐 composerPlainText：DOM 子树 -> 纯文本（chip 还原为 @kind:value）。 */
function editorPlainText(node) {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent || ''
  }
  if (node.nodeType !== Node.ELEMENT_NODE) {
    return ''
  }
  const el = node
  if (el.dataset.refText) {
    return el.dataset.refText
  }
  // 仅剩占位 <br> 的编辑器视为空。
  if (el.dataset.slot === RICH_INPUT_SLOT && el.childNodes.length === 1 && el.firstChild?.nodeName === 'BR') {
    return ''
  }
  if (el.tagName === 'BR') {
    return '\n'
  }
  let text = ''
  for (const child of Array.from(node.childNodes)) {
    text += editorPlainText(child)
  }
  const block = el.tagName === 'DIV' || el.tagName === 'P'
  return block && text && el.dataset.slot !== RICH_INPUT_SLOT ? `${text}\n` : text
}

/** 对齐 composerHtml：纯文本 -> <br> 分行 HTML。 */
function editorHtml(text) {
  return escapeHtml(text).replace(/\n/g, '<br>')
}

/** 写回输入框：重绘 contentEditable DOM 并派发 input 事件驱动官方状态同步。
 *  光标/聚焦仅在编辑器仍可见时执行——异步返回时用户可能已切到其他会话，
 *  隐藏 tab 不抢焦点、不影响新会话的输入体验。 */
function writeEditorText(editor, text) {
  editor.innerHTML = editorHtml(text)
  editor.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: text }))
  requestAnimationFrame(() => {
    const style = window.getComputedStyle(editor)
    const visible =
      editor.isConnected &&
      style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      editor.getClientRects().length > 0
    if (!visible) return
    const range = document.createRange()
    range.selectNodeContents(editor)
    range.collapse(false)
    const selection = window.getSelection()
    if (selection) {
      selection.removeAllRanges()
      selection.addRange(range)
    }
    editor.focus()
  })
}

function findEditor() {
  // 桌面 App 多会话 tab 采用 keep-alive：每个曾激活的 tab 都保持 MOUNTED，
  // DOM 中可能同时存在多个 composer-rich-input。必须选中「可见」的那个，
  // 否则会读到隐藏 tab 的空输入框，误报"请先输入内容"。
  const editors = document.querySelectorAll(`[data-slot="${RICH_INPUT_SLOT}"]`)
  for (const el of editors) {
    if (!el.isConnected || el.closest('[hidden]')) continue
    const style = window.getComputedStyle(el)
    if (style.display === 'none' || style.visibility === 'hidden') continue
    if (el.getClientRects().length === 0) continue
    return el
  }
  return editors[0] ?? null
}

// ---------------------------------------------------------------------------
// 优化调用：本插件后端 REST（/api/plugins/prompt-optimizer/optimize）
// ---------------------------------------------------------------------------

async function optimizePrompt(raw, sessionId) {
  // 全链路超时统一 300s（5 分钟）：
  //  - 前端 ctx.rest timeoutMs: 300000（本处，Electron main → HTTP 透传）
  //  - 后端 plugin_api.py run_oneshot timeout: 300
  // 实测当前会话模型单次调用 12.6~52s，长文本/高负载时可能达到数分钟，
  // 300s 窗口覆盖长耗时任务，超过则按超时错误处理并提示。
  // 注意：不走 host.request('llm.oneshot')——桌面 App RPC 层有 30s 硬超时
  // （apps/desktop/src/hermes.ts DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS=30_000），
  // 模型响应经常超过 30s，必失败。改用 ctx.rest 调本插件后端
  // /api/plugins/prompt-optimizer/optimize（plugin_api.py 内直接跑
  // run_oneshot，可等待 5 分钟，且继承会话模型）。
  if (!restRequest) {
    throw new Error('plugin rest door not ready')
  }
  const result = await restRequest('/optimize', {
    method: 'POST',
    body: {
      input: raw,
      instructions: OPTIMIZE_INSTRUCTIONS,
      session_id: sessionId ?? undefined,
      max_tokens: 2000,
      temperature: 0.3,
      timeout: 300
    },
    timeoutMs: 300000
  })
  return (result?.text ?? '').trim()
}

// ---------------------------------------------------------------------------
// 按钮组件 —— 状态机: idle -> running -> optimized(可撤回/可再次优化)
// ---------------------------------------------------------------------------

function OptimizeButton() {
  const t = usePluginI18n(ID)
  const sessionId = useValue(host.state.activeSessionId)

  const [busy, setBusy] = useState(false)
  const [optimized, setOptimized] = useState(false)
  // 撤回目标：本次优化前的原始文本。appliedRef: 最后一次写入的优化文本。
  const originalRef = useRef('')
  const appliedRef = useRef('')

  // 切换会话时清空优化状态，防止跨会话撤回错乱。
  useEffect(() => {
    setBusy(false)
    setOptimized(false)
    originalRef.current = ''
    appliedRef.current = ''
  }, [sessionId])

  const readDraft = useCallback(() => {
    const editor = findEditor()
    return editor ? editorPlainText(editor).trim() : ''
  }, [])

  // 写入目标显式绑定「发起操作时」的编辑器引用：优化是异步的，返回时用户
  // 可能已切换会话，绝不能重新 findEditor()（那会写入新会话的输入框）。
  // 编辑器已从 DOM 断开（会话被关闭）时放弃写入，返回 false 由调用方提示。
  const writeDraft = useCallback((editor, text) => {
    if (!editor || !editor.isConnected) {
      return false
    }
    writeEditorText(editor, text)
    return true
  }, [])

  const runOptimize = useCallback(
    async (draft, editor, sessionId) => {
      setBusy(true)
      try {
        const result = await optimizePrompt(draft, sessionId)
        if (!result) {
          throw new Error('empty result')
        }
        if (!writeDraft(editor, result)) {
          throw new Error('editor not found')
        }
        originalRef.current = draft
        appliedRef.current = result
        setOptimized(true)
        haptic('tap')
        // 写入目标是发起时的会话；若用户已切走，提示结果所在位置。
        const style = editor ? window.getComputedStyle(editor) : null
        const stillVisible =
          editor &&
          editor.isConnected &&
          style.display !== 'none' &&
          style.visibility !== 'hidden' &&
          editor.getClientRects().length > 0
        host.notify({ kind: 'success', message: stillVisible ? t('done') : t('doneElsewhere') })
      } catch (err) {
        // 区分超时与模型异常，给出可操作的提示。
        const timedOut = err instanceof Error && err.message.includes('timed out')
        host.notify({ kind: 'error', message: timedOut ? t('timedOut') : t('failed') })
      } finally {
        setBusy(false)
      }
    },
    [t]
  )

  const onClick = useCallback(() => {
    // 同步阶段捕获「发起操作时」的编辑器引用与会话 ID——优化异步返回时
    // 用户可能已切换会话，写入与模型路由都必须锚定发起时刻。
    const editor = findEditor()
    const draft = editor ? editorPlainText(editor).trim() : ''
    const sessionId = host.state.activeSessionId.get() ?? undefined
    if (busy) {
      return
    }
    if (!draft) {
      host.notify({ kind: 'warning', message: t('empty') })
      return
    }
    if (optimized) {
      // 文本已被用户改过（不同于上次优化结果）-> 视为再次优化；否则撤回。
      if (draft !== appliedRef.current) {
        void runOptimize(draft, editor, sessionId)
        return
      }
      if (originalRef.current) {
        // 撤回：发生在当前会话内，写入当前可见编辑器（findEditor）。
        if (writeDraft(findEditor(), originalRef.current)) {
          originalRef.current = ''
          appliedRef.current = ''
          setOptimized(false)
          haptic('tap')
          host.notify({ kind: 'info', message: t('reverted') })
        }
      }
      return
    }
    void runOptimize(draft, editor, sessionId)
  }, [busy, optimized, readDraft, runOptimize, t])

  // 渲染时判定当前按钮语义：optimized 且文本未变 -> 撤回；否则 -> 优化。
  const draftNow = optimized ? readDraft() : ''
  const undoing = optimized && draftNow === appliedRef.current
  const label = busy ? t('optimizing') : undoing ? t('undo') : t('optimize')
  const tip = busy ? t('optimizingTip') : undoing ? t('undoTip') : t('optimizeTip')
  const Icon = busy ? icons.Loader2 : undoing ? icons.CornerDownLeft : icons.Zap

  return jsx(Tip, {
    label: tip,
    children: jsx(Button, {
      type: 'button',
      variant: 'ghost',
      size: 'icon-xs',
      disabled: busy,
      onClick,
      'aria-label': label,
      className: cn(
        'text-(--ui-text-secondary)',
        optimized && !busy && 'text-(--ui-accent)',
        busy && 'cursor-wait opacity-70'
      ),
      children: jsx(Icon, { className: cn('size-3.5', busy && 'animate-spin') })
    })
  })
}

// ---------------------------------------------------------------------------
// 插件入口
// ---------------------------------------------------------------------------

export default {
  id: ID,
  name: 'Prompt Optimizer',
  register(ctx) {
    restRequest = ctx.rest
    ctx.i18n.register({
      zh: {
        optimize: '优化提示词',
        optimizing: '优化中…',
        undo: '撤回',
        optimizeTip: '使用当前会话模型优化输入框中的提示词',
        optimizingTip: '正在优化，请稍候…',
        undoTip: '撤回本次优化，恢复原始提示词',
        empty: '请先在输入框中输入要优化的提示词',
        done: '提示词已优化并填入输入框（可点击「撤回」恢复原文）',
        doneElsewhere: '提示词已优化并填入原会话输入框（切回原会话可查看并「撤回」）',
        reverted: '已恢复原始提示词',
        timedOut: '优化超时：任务超过 5 分钟未完成，请检查模型服务状态后重试',
        failed: '优化失败：模型返回异常，请重试'
      },
      en: {
        optimize: 'Optimize prompt',
        optimizing: 'Optimizing…',
        undo: 'Undo',
        optimizeTip: 'Optimize the draft with the current session model',
        optimizingTip: 'Optimizing, please wait…',
        undoTip: 'Undo this optimization and restore the original prompt',
        empty: 'Type a prompt in the composer first',
        done: 'Prompt optimized and filled in (click Undo to restore)',
        doneElsewhere: 'Prompt optimized and filled in the original session composer (switch back to view and Undo)',
        reverted: 'Restored the original prompt',
        timedOut: 'Optimization timed out: the task exceeded 5 minutes, check the model service and retry',
        failed: 'Optimization failed: model error, please retry'
      }
    })

    ctx.register({
      id: 'optimize',
      area: COMPOSER_AREAS.actions,
      order: 10,
      render: () => jsx(OptimizeButton, {})
    })
  }
}
