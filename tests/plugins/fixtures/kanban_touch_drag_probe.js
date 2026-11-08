// Behavioral probe for attachTouchDrag() (#115568): extracts the function from the shipped
// dashboard bundle (no build step — the bundle IS the source) and drives it through real
// pointerdown/pointermove/pointerup sequences with a minimal DOM stub. Exits 0 and prints "PASS"
// when a stationary tap never claims the gesture (leaves preventDefault/dispatchEvent untouched
// so the synthesized click still opens the card) and a real drag still claims it past the
// movement threshold. Run via: node kanban_touch_drag_probe.js <path-to-bundle>
const fs = require("fs");

const bundlePath = process.argv[2];
const src = fs.readFileSync(bundlePath, "utf8");
const start = src.indexOf("function attachTouchDrag");
if (start === -1) { console.error("attachTouchDrag not found in bundle"); process.exit(1); }
const bodyStart = src.indexOf("{", start);
let depth = 0, end = bodyStart;
for (; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}") { depth--; if (depth === 0) break; }
}
const fnSrc = src.slice(start, end + 1);

class FakeEl {
  constructor() {
    this.listeners = {};
    this.classList = { add() {}, remove() {}, contains() { return false; } };
    this.style = {};
    this.offsetWidth = 100;
  }
  addEventListener(t, f) { this.listeners[t] = f; }
  removeEventListener(t) { delete this.listeners[t]; }
  cloneNode() { return new FakeEl(); }
  closest() { return null; }
  getAttribute() { return null; }
  hasAttribute() { return false; }
  dispatchEvent(ev) { this.dispatched = (this.dispatched || []).concat([ev.type]); }
  remove() {}
}
const docListeners = {};
global.document = {
  body: { appendChild() {} },
  addEventListener(t, f) { docListeners[t] = f; },
  removeEventListener(t) { delete docListeners[t]; },
  elementFromPoint() { return null; },
};
global.CustomEvent = function (type, opts) { this.type = type; this.detail = opts && opts.detail; };

eval(fnSrc);

// A real tap: pointerdown then pointerup with sub-threshold jitter must NOT claim the gesture.
const tapEl = new FakeEl();
attachTouchDrag(tapEl, "task-tap");
const tapDown = { pointerType: "touch", clientX: 100, clientY: 100, preventDefault() { this._pd = true; } };
tapEl.listeners["pointerdown"](tapDown);
const tapMove = { clientX: 102, clientY: 101, preventDefault() { this._pd = true; } };
docListeners["pointermove"](tapMove);
docListeners["pointerup"]({});

if (tapDown._pd || tapMove._pd) {
  console.error("FAIL: a stationary tap called preventDefault (suppresses the click)");
  process.exit(1);
}
if (tapEl.dispatched) {
  console.error("FAIL: a stationary tap dispatched a drag/drop event");
  process.exit(1);
}

// A real drag: movement past the threshold must still claim the gesture.
const dragEl = new FakeEl();
attachTouchDrag(dragEl, "task-drag");
dragEl.listeners["pointerdown"]({ pointerType: "touch", clientX: 100, clientY: 100, preventDefault() {} });
let dragClaimed = false;
docListeners["pointermove"]({ clientX: 140, clientY: 140, preventDefault() { dragClaimed = true; } });
if (!dragClaimed) {
  console.error("FAIL: a real drag past the threshold never claimed the gesture");
  process.exit(1);
}

console.log("PASS");
