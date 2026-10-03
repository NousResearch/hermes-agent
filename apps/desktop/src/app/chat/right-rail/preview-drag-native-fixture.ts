import { $rightRailActiveTabId } from '@/store/layout'
import { openPreview } from '@/store/preview'

import { actOnActivePreview } from './preview-act'
import { registerPreviewInput } from './preview-input'
import { capturePreviewGuest, type PreviewGuest } from './preview-input-guest'
import { registerPreviewScriptRunner } from './preview-script-runner'

type Guest = HTMLElement & PreviewGuest & {
  setZoomFactor: (zoom: number) => void
}

const fixture = `<!doctype html><style>
body { margin:0; user-select:none } #box { position:absolute; left:60px; top:60px; width:200px; height:80px; background:lightblue }
#handle, #adjacent-handle { position:absolute; right:-10px; bottom:-10px; width:20px; height:20px; touch-action:none }
#adjacent-handle { display:none; bottom:10px }
.small { transform:scale(.25); transform-origin:top left }
.small #handle, .small #adjacent-handle { display:block; transform:scale(4) }
svg { position:absolute; left:360px; top:120px; width:100px; height:100px } circle { touch-action:none }
#cover { display:none; position:absolute; left:240px; top:120px; width:60px; height:60px; background:gray; z-index:3 }
</style><div id="box"><button id="handle" aria-label="Resize"></button><button id="adjacent-handle" aria-label="Other handle"></button></div>
<svg><circle id="svg-handle" cx="20" cy="20" r="10" fill="blue" /></svg><div id="cover"></div><pre id="ledger"></pre>
<script>
window.events=[]; let start;
for (const name of ['pointerdown','pointermove','pointerup','mousedown','mousemove','mouseup']) {
 document.addEventListener(name, e => {
  events.push({type:e.type, trusted:e.isTrusted, buttons:e.buttons, x:e.clientX, y:e.clientY, target:e.target.id});
  ledger.textContent=JSON.stringify(events.slice(-8));
 });
}
document.addEventListener('pointerdown', e => {
 if (!['handle','svg-handle'].includes(e.target.id)) return;
 start={x:e.clientX,y:e.clientY,w:box.offsetWidth,h:box.offsetHeight};
 e.target.setPointerCapture(e.pointerId);
});
document.addEventListener('pointermove', e => {
 if (start && e.buttons===1) {box.style.width=start.w+e.clientX-start.x+'px';box.style.height=start.h+e.clientY-start.y+'px'}
});
document.addEventListener('pointerup', () => {start=null});
window.state=()=>({width:box.offsetWidth,height:box.offsetHeight,events});
window.coverHandle=on=>cover.style.display=on?'block':'none';
window.smallScale=on=>box.classList.toggle('small',on);
</script>`

function check(condition: unknown, message: string): asserts condition {
  if (!condition) {throw new Error(message)}
}

async function createGuest(): Promise<Guest> {
  const guest = document.createElement('webview') as Guest
  guest.setAttribute('style', 'width:600px;height:650px;display:inline-flex')
  guest.setAttribute('partition', 'native-drag-fixture')

  const ready = new Promise<void>((resolve, reject) => {
    guest.addEventListener('dom-ready', () => resolve(), { once: true })
    guest.addEventListener('did-fail-load', () => reject(new Error('Guest load failed')), { once: true })
  })

  guest.setAttribute('src', 'data:text/html;charset=utf-8,' + encodeURIComponent(fixture))
  document.body.append(guest)
  await ready

  return guest
}

interface State {
  width: number
  height: number
  events: Array<{ type: string; trusted: boolean; buttons: number; x: number; y: number; target: string }>
}

async function runNativeDragFixture() {
  const guest = await createGuest()
  const other = await createGuest()
  const read = () => guest.executeJavaScript('state()') as Promise<State>
  openPreview({ kind: 'url', url: 'https://example.test', source: 'https://example.test', label: 'Native fixture' })
  const id = $rightRailActiveTabId.get()!
  let abort: AbortController | undefined
  let held = 0

  const unregister = registerPreviewInput(id, () => {
    const captured = capturePreviewGuest(() => guest, () => false)!

    return { ...captured, send: async event => {
      await captured.send(event)

      if (event.type === 'mouseMove' && event.modifiers?.includes('leftbuttondown') && ++held === 3) {abort?.abort()}
    } }
  })

  const unregisterScript = registerPreviewScriptRunner(id, code => guest.executeJavaScript(code))
  const observations = []

  try {
    for (const zoom of [1, 1.25]) {
      guest.setZoomFactor(zoom)
      await guest.executeJavaScript('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))')
      const inventory = await actOnActivePreview({ kind: 'elements', full: true })
      const ref = inventory.elements?.find(element => element.label === 'Resize')?.ref
      check(ref, 'Inventory must find accessible HTML handle: ' + JSON.stringify(inventory))

      for (const target of [{ ref }, { selector: '#svg-handle' }]) {
        for (const sign of [1, -1]) {
          await guest.executeJavaScript('events.length=0')
          const result = await actOnActivePreview({ kind: 'drag', ...target, dx: sign * 40, dy: sign * 20 })
          check(result.success, JSON.stringify(result))
          const state = await read()
          check(state.width === (sign === 1 ? 240 : 200) && state.height === (sign === 1 ? 100 : 80), `Wrong geometry: ${JSON.stringify(state)}`)
          const down = state.events.filter(event => event.type === 'pointerdown')
          const up = state.events.filter(event => event.type === 'pointerup')
          check(down.length === 1 && down[0].trusted && down[0].buttons === 1, 'Trusted pointerdown required')
          check(up.length === 1 && up[0].trusted && up[0].buttons === 0, 'Trusted release required')
          const moves = state.events.filter(event => event.type === 'pointermove' && event.buttons === 1)
          check(moves.length > 1 && moves.every(event => event.trusted), 'Trusted held movement required')
          observations.push({ zoom, target, delta: [sign * 40, sign * 20], ...state })
        }
      }
    }

    // Delivery is not a business-outcome claim: occlusion can intercept it.
    for (const mode of ['coverHandle', 'smallScale']) {
      await guest.executeJavaScript(`events.length=0; ${mode}(true)`)
      const result = await actOnActivePreview({ kind: 'drag', selector: '#handle', dx: 40, dy: 20 })
      const state = await read()
      check(state.width === 200 && state.height === 80, 'Occluded target must not resize')
      const hit = state.events.find(event => event.type === 'pointerdown')
      check(hit?.trusted && hit.target !== 'handle', 'Fixture must demonstrate actual interception')
      observations.push({ mode, result: { success: result.success, note: result.note }, ...state })
      await guest.executeJavaScript(`${mode}(false)`)
    }

    abort = new AbortController()
    held = 0
    const interrupted = await actOnActivePreview({ kind: 'drag', selector: '#handle', dx: 40, dy: 20 }, abort.signal)
    check(!interrupted.success, 'Interrupted gesture must fail')
    abort = undefined
    const interruptedState = await read()
    check(interruptedState.events.at(-1)?.buttons === 0, 'Interrupted gesture must release')
    const otherState = await other.executeJavaScript('state()') as State
    check(!otherState.events.some(event => event.type === 'pointerdown'), 'No input may reach unrelated guest')

    return { success: true, observations, interrupted: { result: interrupted, width: interruptedState.width, height: interruptedState.height }, otherGuestDowns: 0 }
  } finally {
    unregister()
    unregisterScript()
    // Leave fixture pixels available for the host screenshot; its window owns teardown.
  }
}

Object.assign(window, { runNativeDragFixture })
