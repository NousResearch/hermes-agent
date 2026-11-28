# Suno Browser Automation Patterns

Reusable JavaScript snippets and `browser_exec` recipes for driving `suno.com/create`.
Every snippet is wrapped in an IIFE (`(() => { ... })()`) so variables stay scoped to the call and repeated evaluations never raise `SyntaxError: Identifier 'X' has already been declared`.

## 1. Verify Login and Switch to the Advanced Tab (`browser_exec`)

```python
# Open suno.com/create and switch to the Advanced tab
new_tab("https://suno.com/create")
wait_for_load()
print(page_info())
tab_status = js("""(() => {
  for (const el of document.querySelectorAll('[role="tab"], button')) {
    if ((el.textContent || '').trim() === 'Advanced') {
      el.click();
      return 'switched-to-advanced';
    }
  }
  return 'advanced-tab-not-found';
})()""")
print(tab_status)
```

If `page_info()` or the DOM shows a `Sign In` button, stop and instruct the user to complete the one-time login in the persistent CDP browser window.

## 2. Batch-Fill All Fields via Native Setters

Suno's create form uses controlled React inputs. Assigning `el.value = ...` directly does not update React state; always call the native prototype value setter (`HTMLInputElement.prototype` for `<input>`, `HTMLTextAreaElement.prototype` for `<textarea>`) and dispatch bubbling `input` and `change` events.

You can generate a properly escaped IIFE with `python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py emit-fill-js ...` or use this pattern directly inside `browser_exec` (`js(...)`) / `browser_console`:

```javascript
(() => {
  const inputSetter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  const taSetter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, 'value').set;

  const setField = (el, setter, val) => {
    if (!el) return false;
    setter.call(el, val);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  };

  // Title (<input>)
  const titleEl = document.querySelector('input[placeholder*="Song Title"], input[placeholder*="Title"]');

  // Lyrics & Style (<textarea>)
  let lyricsEl = null, styleEl = null;
  for (const ta of document.querySelectorAll('textarea')) {
    const ph = (ta.placeholder || '').toLowerCase();
    const aria = (ta.getAttribute('aria-label') || '').toLowerCase();
    if (!lyricsEl && (ph.includes('lyrics') || aria.includes('lyrics'))) lyricsEl = ta;
    else if (!styleEl && (ph.includes('classic jazz') || ph.includes('style') || aria.includes('style'))) styleEl = ta;
  }

  return JSON.stringify({
    title: setField(titleEl, inputSetter, 'YOUR TITLE HERE'),
    lyrics: setField(lyricsEl, taSetter, 'YOUR LYRICS HERE WITH [metatags]'),
    style: setField(styleEl, taSetter, 'YOUR STYLE PROMPT HERE'),
  });
})()
```

## 3. Selectors Cheatsheet

| Element | Selector Strategy | Notes |
|---|---|---|
| Advanced tab | `[role="tab"], button` where `textContent.trim() === 'Advanced'` | Replaces the legacy "Custom Mode" toggle on v5.5 |
| Title input | `input[placeholder*="Song Title"], input[placeholder*="Title"]` | Use `HTMLInputElement.prototype` setter |
| Lyrics textarea | `textarea` with placeholder or `aria-label` containing `"lyrics"` | Up to ~3,000 chars; include structural `[metatags]` |
| Style textarea | `textarea` with placeholder containing `"classic jazz"` or `"style"` | Sample-style placeholder hint; up to 1,000 chars |
| Exclude Styles | `input, textarea` with placeholder containing `"exclude"` | Inside "More Options" when expanded |
| Range sliders | `input[type="range"]` | Weirdness / Style Influence; set via `HTMLInputElement.prototype` setter |
| Create button | Loop `button` elements matching `'Create song'`, `'Create songs'`, or `'Create'` | Often below the fold; click via JS `.click()` |
| Generated song links | `a[href*="/song/"]` | Returns 2 variations per generation click |

## 4. Setting Range Inputs (Sliders) via JavaScript

```javascript
(() => {
  const slider = document.querySelector('input[type="range"]');
  if (!slider) return 'slider-not-found';
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  setter.call(slider, '50');
  slider.dispatchEvent(new Event('input', { bubbles: true }));
  slider.dispatchEvent(new Event('change', { bubbles: true }));
  return 'ok';
})()
```

## 5. Click the Create Button via JS

```javascript
(() => {
  for (const btn of document.querySelectorAll('button')) {
    const t = (btn.textContent || '').trim();
    if (t === 'Create song' || t === 'Create' || t === 'Create songs') {
      btn.click();
      return JSON.stringify({ clicked: true, label: t, disabled: Boolean(btn.disabled) });
    }
  }
  return JSON.stringify({ clicked: false, error: 'Create button not found' });
})()
```

If `disabled: true` is returned, one of the required fields did not register with React state—re-run the batch-fill IIFE and check that `title`, `lyrics`, and `style` all returned `true`.

## 6. Poll Generation Status and Harvest Track Links

Generation on Suno v5.5 takes **60–120+ seconds**, and the two variations per click often complete 30–90 seconds apart. Inside `browser_exec`, call `capture_screenshot()` and inspect with `vision_analyze` (checking for cover art + duration timestamps vs. active loading spinners), then harvest all `/song/` URLs:

```javascript
(() => {
  const seen = new Set();
  const results = [];
  for (const link of document.querySelectorAll('a[href*="/song/"]')) {
    const title = (link.textContent || '').trim();
    const href = link.href || link.getAttribute('href') || '';
    if (!title || !href) continue;
    const key = title + '|' + href;
    if (seen.has(key)) continue;
    seen.add(key);
    results.push(title + ' | ' + href);
  }
  return results.join('\n');
})()
```

Pipe the harvested lines into `python3 ${HERMES_SKILL_DIR}/scripts/suno_helper.py parse-harvest` to group variations by song title and extract `song_id`s.
