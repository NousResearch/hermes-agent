Compacted conversation## Bug Debrief: The "0-0, 1-1" Segment Display Bug

---

### Background: What the build script does

Your caption editor UI is written in modern React (`.tsx` files). Browsers can't run that directly — they need plain JavaScript. `build.mjs` is the **compiler/bundler**: a small script you run once that takes all your source files and squashes them into a single `dist/index.js` that the browser can load.

Think of it like a **printing press**: your source files are the manuscript, `build.mjs` is the press operator, and `dist/index.js` is the finished pamphlet handed to the reader.

---

### Background: What JSX is

React lets you write HTML-looking code inside JavaScript:

```tsx
<div className="card">Hello</div>
```

Browsers don't understand this syntax. So before bundling, the compiler **translates** it into plain function calls:

```js
React.createElement("div", { className: "card" }, "Hello")
```

That translation step is called the **JSX transform**. There are two modes for doing it.

---

### The two transform modes (this is where the bug lives)

**Mode 1: `transform` (classic)**
Translates `<div key="abc">Hello</div>` into:
```js
React.createElement("div", { key: "abc" }, "Hello")
```
The `key` goes **inside the props object** (the `{}` bag of attributes). Children come after, as extra arguments.

**Mode 2: `automatic`**
Uses a different helper function with a different calling convention:
```js
jsx("div", { children: "Hello" }, "abc")  // key is the 3rd positional argument
```

Key is passed as a **separate positional argument**, not inside the props bag.

---

### The shim: the middleman that caused the crash

Because the plugin runs inside the Hermes dashboard (which already has React loaded), the plugin can't bundle its own copy of React — that would create two conflicting Reacts. Instead, `build.mjs` writes a tiny "shim" — a fake `react` module that just points to the React instance the dashboard already loaded:

```js
// shim (simplified)
const R = window.__HERMES_PLUGIN_SDK__.React;
export const jsx = R.createElement;  // ← THE BUG
```

This shim said: "when someone calls `jsx(...)`, just call `React.createElement(...)`".

---

### Why this exploded

With `automatic` mode active, the compiled code for a segment card looked like:

```js
jsx(SegmentCard, { text: "Xin chào", lang: "vi" }, "0-0")
//                                                    ↑
//                                              key: "0-0"
```

The shim faithfully forwarded this to `React.createElement`:

```js
React.createElement(SegmentCard, { text: "Xin chào", lang: "vi" }, "0-0")
//                                                                    ↑
//                                                         treated as a CHILD
```

`React.createElement`'s 3rd argument and beyond are **children** — content to render inside the element. So React dutifully rendered the string `"0-0"` as the visible content of every card, erasing everything else.

**Analogy:** Imagine a phone operator who's supposed to route calls. You dial `1-800-FLOWERS` and ask them to "connect me to extension 42". They mishear "42" as the message to deliver, and instead of connecting you, they just say "42" to the florist and hang up.

---

### The fix

Switch `build.mjs` from `automatic` to `transform` mode:

```js
// Before
jsx: "automatic"

// After
jsx: "transform",
jsxFactory: "React.createElement",
jsxFragment: "React.Fragment",
```

Now the compiler translates JSX into direct `React.createElement(type, { key: "0-0", ...props }, children)` calls — key goes **in the props bag**, not as a positional argument. The shim is bypassed entirely for key handling, and the cards render their actual content.

Also added `import React from "react"` to the source file, because `transform` mode requires React to be in scope (unlike `automatic` which imports the helpers itself).

---

### Why it was hard to spot

The symptom ("0-0, 1-1" text visible in cards) looked like a **data problem** — wrong content being fetched or passed. It was actually a **compilation problem** — the right content was there, but the build toolchain's calling convention mismatch caused it to be overwritten by a metadata string that was never supposed to be visible at all.