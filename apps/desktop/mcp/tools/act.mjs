/**
 * Mutating tools: real CDP Input events (not synthetic DOM events), so handlers
 * like blur→cancel races behave exactly as with a human. Gated upstream by
 * DESKTOP_DEBUG_MCP_ALLOW_ACT=1.
 */

import { requireSelector } from './read.mjs'

async function centerOf(cdp, sel) {
  const box = await cdp.eval(`(() => {
    const el = document.querySelector(${JSON.stringify(sel)})
    if (!el) return null
    const b = el.getBoundingClientRect()
    return { x: b.x + b.width / 2, y: b.y + b.height / 2 }
  })()`)

  if (!box) throw new Error(`element not found: ${sel}`)
  return box
}

async function click(cdp, sel) {
  const { x, y } = await centerOf(cdp, sel)
  for (const type of ['mousePressed', 'mouseReleased']) {
    await cdp.send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1 })
  }
  return { clicked: true, at: { x: Math.round(x), y: Math.round(y) } }
}

async function type(cdp, sel, text) {
  const focused = await cdp.eval(`(() => {
    const el = document.querySelector(${JSON.stringify(sel)})
    if (!el) return false
    el.focus()
    return true
  })()`)

  if (!focused) throw new Error(`element not found: ${sel}`)

  // Use Input.insertText (not dispatchKeyEvent type:'char'): the rich-text
  // composer (rich-editor.ts / contentEditable) ignores synthetic char events,
  // so dispatchKeyEvent silently drops the text. insertText fires the real
  // beforeinput/input pipeline the editor listens to.
  await cdp.send('Input.insertText', { text })
  return { typed: text.length }
}

async function press(cdp, key) {
  const map = {
    Enter: { key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13, commands: ['Enter'] },
    Escape: { key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27 },
    Backspace: { key: 'Backspace', code: 'Backspace', windowsVirtualKeyCode: 8, nativeVirtualKeyCode: 8 }
  }
  const def = map[key]
  if (!def) throw new Error(`unsupported key: ${key} (try Enter/Escape/Backspace)`)

  // Ensure the composer (or last-focused editable) keeps focus so the key lands
  // where a human expects. Without this, a click elsewhere between type and press
  // blurs the composer and Enter is swallowed — exactly the silent-fail class
  // this server exists to reproduce.
  await cdp.eval(`(() => {
    const el = document.querySelector('[data-slot="composer-rich-input"]') || document.activeElement
    if (el && typeof el.focus === 'function') el.focus()
    return true
  })()`).catch(() => {})

  await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', ...def })
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', ...def })
  return { pressed: key }
}

export const actTools = [
  {
    name: 'ui_click',
    gated: true,
    description:
      'Click an element in the Hermes desktop renderer via REAL mouse events (fires blur/focus exactly like a human). Selector may be a SELECTORS key or CSS selector.',
    inputSchema: {
      type: 'object',
      properties: { selector: { type: 'string' } },
      required: ['selector']
    }
  },
  {
    name: 'ui_type',
    gated: true,
    description: 'Focus an element and type text as real char events.',
    inputSchema: {
      type: 'object',
      properties: { selector: { type: 'string' }, text: { type: 'string' } },
      required: ['selector', 'text']
    }
  },
  {
    name: 'ui_press',
    gated: true,
    description: 'Press a key (Enter, Escape, Backspace) via real key events.',
    inputSchema: {
      type: 'object',
      properties: { key: { type: 'string' } },
      required: ['key']
    }
  },
  {
    name: 'ui_eval',
    gated: true,
    description:
      'Evaluate JS in the renderer page and get a bounded result. Escape hatch — prefer dedicated tools when they exist.',
    inputSchema: {
      type: 'object',
      properties: { expression: { type: 'string' } },
      required: ['expression']
    }
  }
]

export async function handleAct(name, args, ctx) {
  // Dispatch connects AND attests first, then hands us the live handle —
  // same contract as handleFlow (ctx.cdp). Bugbot P1: this used to call
  // ctx.ensureCdp(), which dispatch never provided, so every act tool
  // threw before any CDP input was sent.
  const cdp = ctx.cdp

  switch (name) {
    case 'ui_click':
      // A3: same friendly empty-selector guard the read tools use — a raw
      // renderer SyntaxError is useless to the agent.
      return click(cdp, requireSelector(ctx.resolveSelector(args.selector)))
    case 'ui_type':
      return type(cdp, requireSelector(ctx.resolveSelector(args.selector)), args.text || '')
    case 'ui_press':
      return press(cdp, args.key)
    case 'ui_eval':
      return ctx.evalBounded(args.expression)
    default:
      throw new Error(`unknown act tool: ${name}`)
  }
}
