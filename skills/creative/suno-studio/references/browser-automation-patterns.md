# Suno Browser Automation Patterns

Reusable JS snippets for driving suno.com/create via `browser_console`.
Run these as a single IIFE to keep variable scopes clean and avoid the
"Identifier already declared" SyntaxError on repeat calls.

## Batch-fill all three fields (title + lyrics + style)

```javascript
(() => {
  const inputSetter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  const taSetter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, 'value').set;

  // Title (input element)
  const titleEl = document.querySelector('input[placeholder*="Song Title"]');
  if (titleEl) {
    inputSetter.call(titleEl, 'YOUR TITLE HERE');
    titleEl.dispatchEvent(new Event('input', { bubbles: true }));
    titleEl.dispatchEvent(new Event('change', { bubbles: true }));
  }

  // Find lyrics + style textareas by inspecting placeholder text
  const allTAs = document.querySelectorAll('textarea');
  let lyricsEl = null, styleEl = null;
  for (const ta of allTAs) {
    if (ta.placeholder && ta.placeholder.includes('lyrics')) lyricsEl = ta;
    if (ta.placeholder && ta.placeholder.includes('classic jazz')) styleEl = ta;
  }

  if (lyricsEl) {
    taSetter.call(lyricsEl, `YOUR LYRICS HERE WITH [metatags]`);
    lyricsEl.dispatchEvent(new Event('input', { bubbles: true }));
    lyricsEl.dispatchEvent(new Event('change', { bubbles: true }));
  }

  if (styleEl) {
    taSetter.call(styleEl, 'YOUR STYLE PROMPT HERE');
    styleEl.dispatchEvent(new Event('input', { bubbles: true }));
    styleEl.dispatchEvent(new Event('change', { bubbles: true }));
  }

  return 'ok';
})();
```

### Why IIFEs

Calling `browser_console` multiple times with top-level `let/const`
declarations raises `SyntaxError: Identifier 'X' has already been
declared`. Wrapping each call in `(() => { ... })()` isolates scope.

## Selectors cheatsheet

| Element | Selector | Notes |
|---------|----------|-------|
| Title input | `input[placeholder*="Song Title"]` | HTMLInputElement |
| Lyrics textarea | `textarea` with placeholder containing "lyrics" | Iterate to find it |
| Style textarea | `textarea` with placeholder containing "classic jazz" | **Not obvious** — this is a sample-style hint, not a label |
| Create button | Loop `button` elements matching `textContent.trim() === 'Create song'` or `'Create'` | Below the fold; use JS click not browser_click |
| Generated song links | `a[href*="/song/"]` | Returns pairs of (title text, full URL) — best way to harvest all tracks after generation |

## Click the Create button via JS

```javascript
(() => {
  for (const btn of document.querySelectorAll('button')) {
    const t = btn.textContent.trim();
    if (t === 'Create song' || t === 'Create') {
      btn.click();
      return 'clicked, disabled=' + btn.disabled;
    }
  }
  return 'not found';
})();
```

Returns `'clicked, disabled=false'` on success. If `disabled=true` the
form has a validation error — check that all required fields are filled.

## Harvest generated track links

```javascript
(() => {
  const results = [];
  for (const link of document.querySelectorAll('a[href*="/song/"]')) {
    const t = link.textContent.trim();
    if (t) results.push(t + ' | ' + link.href);
  }
  return results.join('\n');
})();
```

Each generation produces TWO entries with the same title (two variations).
Match them up manually by order or by clicking through to see durations.

## Polling generation status

`browser_snapshot` does NOT reliably tell you if a track is done.
Use `browser_vision` with a question like _"Are all generated tracks
showing timestamps and cover art, or are any still showing loading
spinners?"_ — the vision model reads the spinner icons and timestamps
accurately.

Typical wait per generation on v5.5: **60-120 seconds**.
The two variations finish at different times (often 30-90s apart).
