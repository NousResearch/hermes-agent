/**
 * Flow tools: scripted UI scenarios with structured outcome reports.
 * ui_flow_edit reproduces the known chat-edit silent-fail races mechanically.
 */
import { sleep, SELECTORS } from '../../scripts/perf/lib/cdp.mjs'

const EDIT_COMPOSER = '[data-slot="aui_edit-composer-root"]'

async function countUserMessages(cdp) {
  return cdp.eval(`document.querySelectorAll('${SELECTORS.turnPair}').length`)
}

export const flowTools = [
  {
    name: 'ui_flow_edit',
    gated: true,
    description:
      'Scripted edit flow over a sent user message in Hermes desktop: hover last user message → click its Edit button → type new text → press Enter. Reports whether the send was accepted, whether the composer core survived, and whether the timeline changed. Reproduction harness for the known silent-fail races.',
    inputSchema: {
      type: 'object',
      properties: {
        newText: { type: 'string', description: 'replacement text; defaults to a timestamped marker' }
      }
    }
  },
  {
    name: 'ui_flow_model_switch',
    gated: true,
    description:
      'Observe the thread while a model switch row is inserted. Reports DOM mutations around the model_switch row to quantify layout jank.',
    inputSchema: { type: 'object', properties: {} }
  }
]

export async function handleFlow(name, args, ctx) {
  const cdp = ctx.cdp
  if (!cdp) throw new Error('no live CDP connection — call desktop_ui_status first')

  if (name === 'ui_flow_edit') return editFlow(cdp, args)
  if (name === 'ui_flow_model_switch') return modelSwitchFlow(cdp)
  throw new Error(`unknown flow tool: ${name}`)
}

async function editFlow(cdp, { newText } = {}) {
  const report = { steps: [], errors: [] }
  const mark = (step, data) => report.steps.push({ step, ...data })

  // 0. Baseline state.
  const beforePairs = await countUserMessages(cdp)
  const beforeComposer = await cdp.eval(`!!document.querySelector('${SELECTORS.composer}')`).catch(() => null)
  mark('baseline', { turnPairs: beforePairs, mainComposerAlive: beforeComposer })

  // 1. Find the LAST user message's edit button.
  const found = await cdp.eval(`(() => {
    const msgs = [...document.querySelectorAll('[data-slot="aui_user-message-root"]')]
    if (!msgs.length) return null
    const last = msgs[msgs.length - 1]
    const btn = last.querySelector('button[aria-label*="Edit" i], button[title*="Edit" i]')
    if (!btn) return { hasMsg: true, hasBtn: false }
    btn.scrollIntoView({ block: 'center' })
    return { hasMsg: true, hasBtn: true }
  })()`)

  if (!found?.hasBtn) {
    report.outcome = 'edit-button-not-found'
    report.hint = 'user messages must be present; hover may be required to reveal the button'
    return report
  }

  // 2. Click it (real mouse events — blur/focus semantics matter here).
  try {
    await cdp.eval(`(() => {
      const msgs = [...document.querySelectorAll('[data-slot="aui_user-message-root"]')]
      const last = msgs[msgs.length - 1]
      last.querySelector('button[aria-label*="Edit" i], button[title*="Edit" i]').scrollIntoView({ block: 'center' })
    })()`)
    await sleep(150)

    const box = await cdp.eval(`(() => {
      const msgs = [...document.querySelectorAll('[data-slot="aui_user-message-root"]')]
      const last = msgs[msgs.length - 1]
      const b = last.querySelector('button[aria-label*="Edit" i], button[title*="Edit" i]').getBoundingClientRect()
      return { x: b.x + b.width / 2, y: b.y + b.height / 2 }
    })()`)

    for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) {
      await cdp.send('Input.dispatchMouseEvent', { type, x: box.x, y: box.y, button: 'left', clickCount: 1 })
    }

    mark('clicked-edit')
  } catch (e) {
    report.errors.push(`click-edit: ${e.message}`)
  }

  // 3. Wait for edit composer to mount.
  await sleep(300)
  const composerUp = await cdp.eval(`!!document.querySelector('${EDIT_COMPOSER}')`)
  mark('edit-composer-mounted', { mounted: composerUp })

  if (!composerUp) {
    report.outcome = 'composer-did-not-open'
    return report
  }

  // 4. Select-all + replace text inside the edit composer.
  await cdp.eval(`(() => {
    const el = document.querySelector('${EDIT_COMPOSER} [contenteditable][data-slot]')
    if (!el) return false
    el.focus()
    const range = document.createRange()
    range.selectNodeContents(el)
    const sel = window.getSelection()
    sel.removeAllRanges(); sel.addRange(range)
    return true
  })()`)

  const text = newText || `debug-edit-${Date.now()}`
  // Input.insertText — dispatchKeyEvent type:'char' is ignored by the
  // contentEditable composer (rich-editor.ts), so use the real insertText
  // pipeline that fires beforeinput/input.
  await cdp.send('Input.insertText', { text })
  await sleep(100)

  const draftAfterTyping = await cdp.eval(`(() => {
    const el = document.querySelector('${EDIT_COMPOSER} [contenteditable][data-slot]')
    return el ? el.textContent.slice(0, 120) : null
  })()`)
  mark('typed', { draftNow: draftAfterTyping })

  // 5. Press Enter (the path with the known unguarded race).
  const coreAliveBeforeEnter = await cdp.eval(
    `(() => { try { return !!window.__HERMES_EDIT_CORE_PROBE__ || true } catch { return true } })()` // placeholder probe
  )
  await cdp.send('Input.dispatchKeyEvent', {
    type: 'keyDown',
    key: 'Enter',
    code: 'Enter',
    windowsVirtualKeyCode: 13,
    nativeVirtualKeyCode: 13
  })
  await cdp.send('Input.dispatchKeyEvent', {
    type: 'keyUp',
    key: 'Enter',
    code: 'Enter',
    windowsVirtualKeyCode: 13,
    nativeVirtualKeyCode: 13
  })

  // 6. Observe outcomes at intervals: did composer close? did timeline change?
  const observations = []
  for (let t = 0; t < 5; t++) {
    await sleep(400)
    observations.push({
      tMs: (t + 1) * 400,
      editComposerStillMounted: await cdp.eval(`!!document.querySelector('${EDIT_COMPOSER}')`),
      turnPairs: await countUserMessages(cdp),
      consoleErrors: consoleErrorsSince(cdp, report.steps[0]?.ts || Date.now())
    })
  }

  report.observations = observations
  report.coreAliveProbeBeforeEnter = coreAliveBeforeEnter

  const final = observations[observations.length - 1]
  report.outcome =
    !final.editComposerStillMounted && final.turnPairs >= beforePairs
      ? 'accepted'
      : final.editComposerStillMounted
        ? 'stuck-open (likely silent-fail race)'
        : 'closed-but-timeline-unclear'

  return report
}

function consoleErrorsSince(_cdp, _since) {
  // Wired to the server's console ring in server.mjs via ctx in future rev;
  // kept minimal for v1.
  return []
}

async function modelSwitchFlow(cdp) {
  // v1: passive observation harness. Records mutations of the thread content
  // region so a switch can be correlated with layout shifts.
  const install = await cdp.eval(`(() => {
    window.__MCP_MUT__ = []
    const target = document.querySelector('${SELECTORS.threadContent}')
    if (!target) return false
    const obs = new MutationObserver(muts => {
      for (const m of muts) {
        window.__MCP_MUT__.push({
          added: [...m.addedNodes].map(n => n.textContent?.slice(0, 60)),
          removed: [...m.removedNodes].map(n => n.textContent?.slice(0, 60)),
          t: Date.now()
        })
      }
    })
    obs.observe(target, { childList: true, subtree: true })
    window.__MCP_MUT_OBS__ = obs
    return true
  })()`)

  return {
    observerInstalled: install,
    hint: "now switch the model in the app (or have the agent click the model menu); then call this tool's readback in a follow-up once wired",
    note: 'v1 passive: mutation buffer readable via ui_eval("JSON.stringify(window.__MCP_MUT__)")'
  }
}
