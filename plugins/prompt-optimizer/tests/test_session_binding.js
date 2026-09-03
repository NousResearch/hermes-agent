/**
 * 会话绑定回归测试 —— 验证「优化结果只写入发起操作时的会话输入框」。
 *
 * 被测逻辑（与 plugin.js 修复后保持一致的行为级复刻）：
 *   1. 点击时同步捕获编辑器引用 editor + 会话 ID（不再在异步返回后 findEditor）
 *   2. 异步返回后写入捕获的编辑器（isConnected 检查，断开则放弃）
 *   3. 隐藏 tab 不抢焦点（writeEditorText 可见性检查）
 *   4. 撤回写入当前可见编辑器
 *
 * 运行：node test_session_binding.js
 */

// ── 最小 DOM stub：模拟桌面 App keep-alive 多会话 tab ──────────────────────

class FakeEditor {
  constructor(id, { visible = true, connected = true } = {}) {
    this.id = id
    this.visible = visible
    this.isConnected = connected
    this.innerHTML = ''
    this.dataset = { slot: 'composer-rich-input' }
    this.childNodes = []
    this.firstChild = null
    this.tagName = 'DIV'
    this.events = []
  }
  getClientRects() {
    return this.visible ? [{ width: 200, height: 24 }] : []
  }
  closest(sel) {
    // 模拟 [hidden] 祖先：不可见的 tab 容器带 hidden 属性
    return this.visible ? null : { hidden: true }
  }
  dispatchEvent(ev) {
    this.events.push(ev)
  }
  focus() {
    this.focused = true
  }
}

const editors = []
function registerEditor(editor) {
  editors.push(editor)
  return editor
}

const documentStub = {
  querySelectorAll: sel => {
    if (sel === '[data-slot="composer-rich-input"]') return editors
    return []
  },
  createRange: () => ({
    selectNodeContents() {},
    collapse() {}
  })
}

const windowStub = {
  getComputedStyle: el => ({
    display: el.visible ? 'block' : 'none',
    visibility: el.visible ? 'visible' : 'hidden'
  }),
  getSelection: () => null
}

// ── 被测逻辑（plugin.js 修复后逻辑的忠实复刻）──────────────────────────────

function findEditor() {
  for (const el of editors) {
    if (!el.isConnected || el.closest('[hidden]')) continue
    const style = windowStub.getComputedStyle(el)
    if (style.display === 'none' || style.visibility === 'hidden') continue
    if (el.getClientRects().length === 0) continue
    return el
  }
  return editors[0] ?? null
}

function writeEditorText(editor, text) {
  editor.innerHTML = text
  editor.dispatchEvent({ type: 'input', bubbles: true, inputType: 'insertText', data: text })
  // 可见性检查：隐藏 tab 不抢焦点
  const style = windowStub.getComputedStyle(editor)
  const visible =
    editor.isConnected &&
    style.display !== 'none' &&
    style.visibility !== 'hidden' &&
    editor.getClientRects().length > 0
  if (visible) editor.focus()
  return visible
}

function writeDraft(editor, text) {
  if (!editor || !editor.isConnected) return false
  writeEditorText(editor, text)
  return true
}

// 模拟异步优化（resolve 时已发生会话切换）
function asyncOptimize(raw, onResolve) {
  return new Promise(resolve => setTimeout(() => resolve(`[优化] ${raw}`), 10))
}

async function runOptimize(draft, editor, sessionId) {
  const result = await asyncOptimize(draft)
  return writeDraft(editor, result)
}

// ── 测试框架 ────────────────────────────────────────────────────────────────

let passed = 0
let failed = 0
const results = []

function check(name, cond, detail = '') {
  if (cond) {
    passed++
    results.push(`  ✅ ${name}`)
  } else {
    failed++
    results.push(`  ❌ ${name} ${detail}`)
  }
}

// ── 场景 1：优化返回前切换会话 → 结果写入原会话，新会话不受影响 ────────────

function resetAll() {
  for (const e of editors) {
    e.visible = false
    e.isConnected = true
  }
}

async function scenario1() {
  resetAll()
  const A = registerEditor(new FakeEditor('A', { visible: true }))
  const B = registerEditor(new FakeEditor('B', { visible: false }))
  B.innerHTML = 'B 的原有草稿'

  // 点击时（A 可见）
  const editorA = findEditor()
  const draftA = 'A 的草稿'
  const promise = runOptimize(draftA, editorA, 'session-A')

  // 等待期间切换会话：B 变为可见，A 隐藏
  A.visible = false
  B.visible = true

  const ok = await promise
  check('场景1: 写入返回 true', ok === true)
  check('场景1: 优化结果写入原会话 A', A.innerHTML === '[优化] A 的草稿', `实际: ${A.innerHTML}`)
  check('场景1: 新会话 B 内容未被覆盖', B.innerHTML === 'B 的原有草稿', `实际: ${B.innerHTML}`)
  check('场景1: 隐藏的 A 未被聚焦', A.focused === undefined)
  check('场景1: 可见的 B 未被聚焦(写的是 A)', B.focused === undefined)
}

// ── 场景 2：连续快速切换 A→B→C→A，结果仍写回 A ─────────────────────────────

async function scenario2() {
  resetAll()
  const A = editors.find(e => e.id === 'A')
  const B = editors.find(e => e.id === 'B')
  const C = registerEditor(new FakeEditor('C', { visible: false }))

  // 当前可见 A
  A.visible = true
  B.visible = false
  C.visible = false
  const editorA = findEditor()
  const promise = runOptimize('草稿2', editorA, 'session-A')

  // 快速切换
  A.visible = false
  B.visible = true
  B.visible = false
  C.visible = true
  C.visible = false
  A.visible = true

  const ok = await promise
  check('场景2: 写入返回 true', ok === true)
  check('场景2: 结果写入 A', A.innerHTML === '[优化] 草稿2', `实际: ${A.innerHTML}`)
  check('场景2: B 内容保持不变', B.innerHTML === 'B 的原有草稿', `实际: ${B.innerHTML}`)
  check('场景2: C 保持空', C.innerHTML === '', `实际: ${C.innerHTML}`)
}

// ── 场景 3：优化期间原会话被关闭 → 放弃写入，绝不串写 ───────────────────────

async function scenario3() {
  resetAll()
  const D = registerEditor(new FakeEditor('D', { visible: true }))
  const E = registerEditor(new FakeEditor('E', { visible: false }))
  E.innerHTML = 'E 的内容'

  const editorD = findEditor()
  const promise = runOptimize('草稿3', editorD, 'session-D')

  // 会话 D 被关闭（DOM 断开），切到 E
  D.isConnected = false
  D.visible = false
  E.visible = true

  const ok = await promise
  check('场景3: 会话关闭时写入返回 false(放弃)', ok === false)
  check('场景3: 断开的 D 未被写入', D.innerHTML === '', `实际: ${D.innerHTML}`)
  check('场景3: E 内容未被串写', E.innerHTML === 'E 的内容', `实际: ${E.innerHTML}`)
}

// ── 场景 4：切回原会话时优化结果仍在（keep-alive DOM 保持）──────────────────

async function scenario4() {
  resetAll()
  const A = editors.find(e => e.id === 'A')
  const B = editors.find(e => e.id === 'B')
  A.visible = true
  B.visible = false
  const editorA = findEditor()
  const promise = runOptimize('草稿4', editorA, 'session-A')
  A.visible = false
  B.visible = true
  await promise
  // 切回 A
  A.visible = true
  B.visible = false
  check('场景4: 切回 A 时优化结果仍在', findEditor().innerHTML === '[优化] 草稿4', `实际: ${findEditor().innerHTML}`)
}

// ── 场景 5：撤回写入当前可见编辑器（原会话内操作）──────────────────────────

async function scenario5() {
  resetAll()
  const A = editors.find(e => e.id === 'A')
  const B = editors.find(e => e.id === 'B')
  // 用户在 A 发起优化并完成（A 可见）
  A.visible = true
  B.visible = false
  const editorA = findEditor()
  await runOptimize('原始草稿', editorA, 'session-A')
  const optimizedText = A.innerHTML
  check('场景5: 优化已写入 A', optimizedText === '[优化] 原始草稿')
  // 撤回：写入当前可见编辑器（即 A）
  const ok = writeDraft(findEditor(), '原始草稿')
  check('场景5: 撤回写入成功', ok === true)
  check('场景5: A 恢复原始草稿', A.innerHTML === '原始草稿', `实际: ${A.innerHTML}`)
}

// ── 场景 6：两个会话各自独立优化，互不干扰 ──────────────────────────────────

async function scenario6() {
  resetAll()
  const F = registerEditor(new FakeEditor('F', { visible: true }))
  const G = registerEditor(new FakeEditor('G', { visible: false }))

  // F 发起优化
  F.visible = true
  G.visible = false
  const editorF = findEditor()
  const p1 = runOptimize('F 的草稿', editorF, 'session-F')

  // 切到 G，G 发起优化
  F.visible = false
  G.visible = true
  const editorG = findEditor()
  const p2 = runOptimize('G 的草稿', editorG, 'session-G')

  await Promise.all([p1, p2])
  check('场景6: F 得到自己的结果', F.innerHTML === '[优化] F 的草稿', `实际: ${F.innerHTML}`)
  check('场景6: G 得到自己的结果', G.innerHTML === '[优化] G 的草稿', `实际: ${G.innerHTML}`)
}

// ── 执行 ────────────────────────────────────────────────────────────────────

;(async () => {
  console.log('会话绑定回归测试\n')
  await scenario1()
  await scenario2()
  await scenario3()
  await scenario4()
  await scenario5()
  await scenario6()
  console.log(results.join('\n'))
  console.log(`\n结果: ${passed} 通过, ${failed} 失败`)
  process.exit(failed > 0 ? 1 : 0)
})()
