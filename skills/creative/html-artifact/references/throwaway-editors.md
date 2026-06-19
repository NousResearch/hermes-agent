# Throwaway Editors

A throwaway editor is a single-file HTML UI purpose-built for one task, ending in an
**export button** that serializes its state to the clipboard so you can paste the
result back into the next prompt. Triage a backlog, tune a prompt, flip feature
flags, adjust animation params — then copy the result out as markdown / JSON / diff /
plain text.

The defining rule: **the artifact must hand its result back.** A pretty editor with
no export is useless to the workflow. (The exception: a *feel-test* prototype — a
drag-to-reorder or animation bench you only need to *experience* — can skip export.
And if the deliverable is one snippet, a static hand-selectable `<pre>` is a valid
"export"; don't add clipboard JS where selection suffices.)

## The skeleton

State → render → controls → export → feedback. `templates/editor.html` is this,
filled in:

```html
<button id="copyBtn" class="btn-primary">Copy as markdown</button>
<button id="resetBtn" class="btn-ghost">Reset</button>
<script>
  const INITIAL = /* the real starting data */;
  let state = structuredClone(INITIAL);        // or read live from the DOM controls

  function render() { /* pure function of state -> DOM; idempotent; call after every change */ }

  function serialize(s) { /* return the pasteable string */ }

  let timer = null;
  function flash(btn, label, orig) {
    btn.textContent = label; btn.classList.add("copied");
    clearTimeout(timer);
    timer = setTimeout(() => { btn.textContent = orig; btn.classList.remove("copied"); }, 1200);
  }

  copyBtn.addEventListener("click", () => {
    writeClipboard(serialize(state)).then(
      () => flash(copyBtn, "Copied \u2713", "Copy as markdown"),
      () => flash(copyBtn, "Copied \u2713", "Copy as markdown")   // flash even on reject; fallback already ran
    );
  });
  resetBtn.addEventListener("click", () => { state = structuredClone(INITIAL); render(); });
  render();  // boot
</script>
```

Conventions: a two-button toolbar (primary Copy + ghost Reset); feedback = swap text
to "Copied ✓" + `.copied` class for 1200ms, guarded by `clearTimeout`; a frozen
`INITIAL` so Reset is trivial and diffs have a baseline; serialize at click time from
current state (don't keep a parallel export buffer); recompute derived values
(counts, totals, diffs) at export time, never trust a stale summary.

## State, three ways

- **Cloned object/array** — `let state = structuredClone(INITIAL)`; mutate fields,
  call `render()`. Best for drag-between-columns boards.
- **Read live from controls** — no JS state object; `currentState()` reads the
  checkboxes/inputs on demand. Best for form/flag editors.
- **The editor text itself** — for a prompt/template editor, the `contenteditable`'s
  text *is* the state; read it with a TreeWalker that mirrors how you insert newlines.

## The clipboard pattern that survives `file://`

`file://` pages often have `navigator.clipboard` undefined or rejected (insecure
context). This helper feature-detects, falls back to an off-screen textarea +
`execCommand`, and **always returns a Promise** so callers uniformly `.then(flash)`:

```js
function writeClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);            // async API when available
  }
  const ta = document.createElement("textarea");           // fallback for file://
  ta.value = text;
  ta.style.position = "fixed";                             // fixed + off-screen = no scroll jump
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); } catch (e) { /* ignore */ }
  document.body.removeChild(ta);
  return Promise.resolve();                                // uniform return so .then() always works
}
```

Rules, in order: feature-detect; fall back to textarea + `execCommand('copy')` inside
the user-gesture handler (works synchronously on `file://`); position the textarea
off-screen; wrap `execCommand` in try/catch; always remove the textarea; normalize to
a Promise; flash on both success and reject (the fallback usually succeeded anyway).

## Export formats — pick by intent

| Format | Build with | Use when you need to… | 
|---|---|---|
| **Markdown** | `lines.push(...)` → `join("\n")`; `#`/`##` headers, `- **id**` bullets | drop the result into a doc / PR / issue for humans |
| **Diff** (`-`/`+`) | compare `state` vs `INITIAL`; emit `'- "k": '+from` / `'+ "k": '+to` | apply only the changes / review intent |
| **JSON** | hand-build to preserve key order, or `JSON.stringify(state, null, 2)` | machine-parseable config to paste into a file |
| **Prompt / plain text** | read the editor text directly | feed a prompt/template/snippet back to the model |

Offer two when both reviewing and applying matter (a Copy-diff *and* a Copy-JSON
button). Hand-roll the serializer when fidelity to a target file's shape matters —
`JSON.stringify` reorders and reformats; build the string yourself to preserve grouped
key order.

## Controls

Native HTML wherever possible — `<input type=range>` (style the thumb clay),
`<input type=checkbox>` toggles, HTML5 drag-and-drop (`draggable="true"` +
`dragstart`/`dragover`/`drop`, snap the drop indicator to element midpoints),
`contenteditable` for text. Live token feedback without a tokenizer:
`Math.round(chars / 4.2)`. For sliders that retune CSS, write a custom property:
`root.style.setProperty('--ease', btn.dataset.ease)` and let the CSS reference
`var(--ease)`.
