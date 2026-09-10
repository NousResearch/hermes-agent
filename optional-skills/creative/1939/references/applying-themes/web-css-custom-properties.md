# Web CSS Custom Properties — Theme Application

## The Pattern

Every 1939 theme maps to 10+ CSS custom properties. The key insight: **you cannot just set `--bg` and `--accent` and call it done.** You need the full hierarchy or 90% of the UI stays hardcoded.

## Complete loadTheme() Implementation

```javascript
function loadTheme(themeJson) {
  const r = themeJson.roles;
  const root = document.documentElement;
  const isDark = themeJson.dark !== false;

  // Swap Background/Canvas based on polarity
  const bg    = isDark ? r.Background.hex : r.Canvas.hex;
  const fg    = isDark ? r.Canvas.hex : r.Background.hex;
  const muted = r.Muted.hex;

  // Primary accents (polarity-agnostic)
  root.style.setProperty('--accent',     r.Highlight.hex);
  root.style.setProperty('--accent-dim', r.Support.hex);
  root.style.setProperty('--chart1',     r.Chart1.hex);
  root.style.setProperty('--chart2',     r.Chart2.hex);

  // Background hierarchy via OKLCH color-mix
  root.style.setProperty('--bg',        bg);
  root.style.setProperty('--surface',   `color-mix(in oklch, ${bg} ${isDark ? 88 : 96}%, ${r.Text.hex})`);
  root.style.setProperty('--surface-2', `color-mix(in oklch, ${bg} ${isDark ? 78 : 92}%, ${muted})`);

  // Text with readability safeguard for dark themes
  root.style.setProperty('--text',      isDark ? `color-mix(in oklch, ${r.Text.hex} 75%, ${fg})` : r.Text.hex);
  root.style.setProperty('--text-dim',   muted);
  root.style.setProperty('--text-muted', `color-mix(in oklch, ${muted} 65%, ${bg})`);

  // Border blends muted toward background
  root.style.setProperty('--border',    `color-mix(in oklch, ${muted} 80%, ${bg})`);

  // Light mode class toggle
  document.documentElement.classList.toggle('light-mode', !isDark);
}
```

## CSS Custom Property Contract

```css
:root {
  /* Core 8 — must be in every theme */
  --bg:        #000001;    /* Background or Canvas (swapped for light mode) */
  --surface:   #0a0a0c;    /* color-mix(bg 88%, text) — dark mode */
  --surface-2: #141416;    /* color-mix(bg 78%, muted) — dark mode */
  --border:    #1f1f22;    /* color-mix(muted 80%, bg) — dark mode */
  --text:      #e8e6e1;    /* Text or mixed — dark mode */
  --text-dim:  #888480;    /* Muted — always */
  --text-muted:#5a5650;    /* color-mix(muted 65%, bg) */
  --accent:    #FF9E93;    /* Highlight — always */
  --accent-dim:#C18DD3;    /* Support — always */
  --chart1:    #AC8FB5;    /* Chart1 — always */
  --chart2:    #BC584B;    /* Chart2 — always */
}
```

## Key Rules

1. **Never hardcode theme colors in CSS rules.** Always use `var(--accent)`, `var(--text)`, etc.
2. **`color: #fff` is the #1 violation.** It works in dark mode but vanishes in light mode. Use `var(--canvas-50)` for text on colored backgrounds.
3. **Chart colors are for DATA only.** Chart1 and Chart2 are for chart series, bar graphs, palette swatches. Never for body text or headings.
4. **Do NOT create per-role tint variables** like `--chart1-tint`. Use `color-mix()` inline or assign the correct semantic role.
5. **`color-mix(in oklch)` is perceptually uniform.** It handles hue and chroma correctly. `color-mix(in srgb)` does not.
6. **The `dark` boolean in theme.json controls polarity.** When `true`, Background is dark. When `false`, Canvas is the background.

## Using Tint Levels

The brand JSON provides 10 tints per role. Access them by index:

```javascript
// In brand.json, each role has:
// "tints": ["#4E4E53", "#39393D", "#252529", "#17171B", "#0A0A0D", ...]
//  Index:     0           1           2           3           4

// To use a specific tint in CSS:
const tint = themeJson.roles.Background.tints[6]; // 700-level
root.style.setProperty('--surface-strong', tint);
```

## Common Mistakes

| Mistake | Result | Fix |
|---------|--------|-----|
| Only setting `--bg` and `--accent` | 90% of UI stays old colors | Set all 10+ variables |
| Using `color: #fff` everywhere | Invisible text in light mode | Use `var(--canvas-50)` or `var(--text)` |
| Hardcoding tint variables | Variable bloat, semantic drift | Use `color-mix()` at the rule level |
| Using Chart1 for headings | Blue/purple text everywhere | Use Highlight for headings, Text for body |
| Forgetting `classList.toggle('light-mode')` | Light mode CSS overrides don't activate | Toggle the class based on `theme.json.dark` |