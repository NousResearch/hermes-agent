# Autolycus TUI Color Palette Redesign — May 20, 2026

## Author

Terra, Autolycus Agent

---

## 1. Current State

### 1.1 Architecture

The Autolycus TUI is a React-based terminal UI built with Ink, running inside
a full-screen alternate screen. Colors flow from two sources:

**TypeScript TUI (`ui-tui/src/theme.ts`):**
- `DARK_THEME` and `LIGHT_THEME` objects define the default palettes
- `fromSkin()` maps Python skin configs to TypeScript theme objects
- Colors are referenced as `t.color.<role>` throughout all Ink components
- The `ThemeColors` interface defines ~30 named color roles

**Python CLI (`hermes_cli/skin_engine.py`):**
- Built-in skins (`default`, `ares`, `mono`, `slate`, `poseidon`, `sisyphus`, `charizard`)
- Each skin defines hex colors for banner, UI, and status roles
- The `default` skin is the active Autolycus skin

### 1.2 Current Default Palette (Dark Theme)

| Role | Hex | Role | Hex |
|------|-----|------|-----|
| primary | `#00d4aa` | accent | `#00d4aa` |
| border | `#0a3d62` | text | `#ffffff` |
| muted | `#1a5f7a` | label | `#00b894` |
| ok | `#00b862` | error | `#e74c3c` |
| warn | `#ff8c00` | prompt | `#ffffff` |
| sessionLabel | `#1a5f7a` | sessionBorder | `#1a5f7a` |
| statusGood | `#00b862` | statusWarn | `#ff8c00` |
| statusBad | `#FF8C00` | statusCritical | `#e74c3c` |
| completionBg | `#0a1628` | completionCurrentBg | `#1a3050` |
| statusBg | `#0a1628` | statusFg | `#C0C0C0` |
| selectionBg | `#1a3050` | shellDollar | `#4dabf7` |

### 1.3 Color Roles and Usage

Each color role maps to specific UI elements:

- **primary** — Brand identity (logo text, banner title, scrollbar grab)
- **accent** — Interactive highlights (spinner frames, section headers, links, chevron toggles)
- **border** — Panel borders, separators, assistant message glyph prefix
- **text** — Assistant message body text, general content
- **muted** — Secondary text, labels, timestamps, tool output borders, system prompt text
- **label** — User message body, user message glyph, field labels
- **ok** — Success indicators, status good states
- **error** — Error states, failed MCP servers, critical alerts
- **warn** — Warning states, approval prompts, update notices
- **prompt** — Input prompt symbol color
- **sessionLabel** / **sessionBorder** — Session info display
- **statusGood/Warn/Bad/Critical** — Context bar progress indicator (escalating severity)
- **completionBg/CurrentBg** — Autocomplete dropdown backgrounds
- **selectionBg** — Selected item background
- **shellDollar** — Shell command prompt indicator

### 1.4 Component Color Mapping

| Component | File | Colors Used |
|-----------|------|-------------|
| Banner/Logo | `branding.tsx` | primary, accent, muted, border |
| Session Panel | `branding.tsx` | primary, accent, muted, text, warn, error, border, sessionLabel, sessionBorder |
| Status Bar | `appChrome.tsx` | border, muted, statusGood/Warn/Bad/Critical, error, warn |
| Message Lines | `messageLine.tsx` | text, muted, border, accent, label |
| User Messages | `roles.ts` | label (body + prefix/glyph) |
| Assistant Messages | `roles.ts` | text (body), border (prefix/glyph) |
| Tool Messages | `roles.ts` | muted (body + prefix/glyph) |
| System Messages | `roles.ts` | muted (prefix/glyph) |
| Approval Prompts | `prompts.tsx` | warn, text, muted |
| Clarify Prompts | `prompts.tsx` | accent, text, label, muted |
| Confirm Prompts | `prompts.tsx` | error/warn, text, muted |
| Thinking/Tool Trail | `thinking.tsx` | accent, muted, text, border, warn, error |
| Markdown | `markdown.tsx` | accent (links, headers, tables), muted (separators) |
| Input Prompt | `appLayout.tsx` | prompt, shellDollar, muted, text |
| Scrollbar | `appChrome.tsx` | border, accent, primary, muted |

---

## 2. Accessibility Audit

### 2.1 WCAG Contrast Analysis

Background assumed: terminal default dark (`#000000` or `#0a0a0a`).

| Role | Current Hex | Contrast Ratio | WCAG AA (4.5:1) | WCAG AAA (7:1) |
|------|------------|----------------|-----------------|----------------|
| text | `#ffffff` | 21:1 | ✅ PASS | ✅ PASS |
| primary | `#00d4aa` | ~12:1 | ✅ PASS | ❌ FAIL |
| accent | `#00d4aa` | ~12:1 | ✅ PASS | ❌ FAIL |
| label | `#00b894` | ~10:1 | ✅ PASS | ❌ FAIL |
| ok | `#00b862` | ~9:1 | ✅ PASS | ❌ FAIL |
| warn | `#ff8c00` | ~10:1 | ✅ PASS | ❌ FAIL |
| error | `#e74c3c` | ~5.5:1 | ✅ PASS | ❌ FAIL |
| **muted** | `#1a5f7a` | **~2.7:1** | **❌ FAIL** | ❌ FAIL |
| sessionLabel | `#1a5f7a` | ~2.7:1 | ❌ FAIL | ❌ FAIL |
| border | `#0a3d62` | ~1.8:1 | ❌ FAIL | ❌ FAIL |
| statusFg | `#C0C0C0` | ~10:1 | ✅ PASS | ❌ FAIL |

**Critical finding:** The `muted` color (`#1a5f7a`) fails WCAG AA by a wide margin.
This color is used for:
- Secondary text and labels throughout the UI
- Tool output borders and content
- System prompt display
- Session metadata (timestamps, paths, session IDs)
- Scrollbar track
- Help text in prompts

This means a significant portion of the UI is effectively illegible for users with
low vision or in suboptimal lighting conditions.

### 2.2 Colorblind Safety Analysis

Common types of color vision deficiency:
- **Deuteranopia** (6% of males): reduced green sensitivity
- **Protanopia** (1% of males): reduced red sensitivity
- **Tritanopia** (0.01%): reduced blue sensitivity (rare)

**Current palette issues:**

1. **primary/accent are identical** (`#00d4aa`): No visual distinction between
   brand identity and interactive highlights. When both appear adjacent, there's
   no way to tell them apart.

2. **statusWarn and statusBad are identical** (`#ff8c00`): The context bar
   progress indicator cannot distinguish between "warning" (50-80%) and "bad"
   (80-95%) states. This is a functional bug — two semantic roles share the same
   color value.

3. **ok (`#00b862`) vs primary (`#00d4aa`)**: Both are teal-green tones separated
   by only ~15° on the hue wheel. Deuteranopes cannot distinguish these — they
   appear as the same desaturated green-gray.

4. **error (`#e74c3c`) vs warn (`#ff8c00`)**: Red-orange pair. Protanopes see
   both as similar shades of brown/orange. Delta E between them is ~12, which is
   "clearly noticeable" but borderline for quick scanning.

### 2.3 Hue Wheel Analysis

```
         0° (red)
         ↑
    error #e74c3c  (hue ~4°)
         |
         |
    warn #ff8c00   (hue ~30°)
         |
         |  ← Only 26° gap (PROBLEM)
    statusBad #FF8C00 (hue ~30°) ← IDENTICAL to warn
         |
         |
    ok #00b862     (hue ~155°)
         |
    primary/accent #00d4aa (hue ~170°) ← Only 15° from ok (PROBLEM)
         |
    label #00b894  (hue ~165°) ← Between ok and primary (PROBLEM)
         |
         ↓
        180° (cyan)
```

The green-teal-cyan region (145°–175°) is overcrowded with 4 different semantic
roles, all within a 30° hue range. This is the single biggest accessibility
problem in the current palette.

---

## 3. Scientific Methodology

### 3.1 Color Space: OKLCH

OKLCH (Open Color Language — Lightness, Chroma, Hue) is a modern perceptually
uniform color space. Unlike RGB or HSL, equal numerical distances in OKLCH
correspond to equal perceived differences.

**Why OKLCH over CIELAB:**
- Separates lightness from chroma and hue, making it easier to control brightness
  independently of saturation
- Designed specifically for digital displays
- Simpler to reason about than Lab's a*/b* axes

### 3.2 Delta E Targets

Delta E (ΔE) measures perceptual color distance. Reference thresholds:

| ΔE Value | Perception | Target Use |
|----------|-----------|------------|
| ≤ 1.0 | Imperceptible | Same color, different rendering |
| 1.0 – 2.0 | Close observation | Subtle variants |
| 2.0 – 3.5 | At a glance | Related elements |
| 3.5 – 5.0 | Clearly noticeable | Different roles |
| ≥ 15 | Distinct colors | **Semantic color separation** |

**Target:** All semantic colors (ok, error, warn, statusBad, statusCritical)
will have ΔE ≥ 15 from each other in the new palette.

### 3.3 Hue Separation Strategy

For colorblind safety, adjacent semantic colors are spaced ≥ 45° on the hue
wheel. This ensures that even when one cone type is deficient, the remaining
cones can still perceive a meaningful difference.

### 3.4 Contrast Requirements

- **Body text:** WCAG AA minimum (4.5:1) — all text roles must meet this
- **UI chrome:** Relaxed (3:1) — borders, backgrounds, decorative elements
- **Status indicators:** Must be distinguishable by luminance alone (not just hue)

---

## 4. Proposed New Palette

### 4.1 Color Definitions

| Role | Current | New | OKLCH | ΔE from nearest |
|------|---------|-----|-------|----------------|
| primary | `#00d4aa` | `#00c9a7` | oklch(0.75 0.15 175) | — |
| accent | `#00d4aa` | `#5b9bd5` | oklch(0.72 0.14 230) | ΔE=28 from primary |
| border | `#0a3d62` | `#0d2137` | oklch(0.20 0.04 230) | — |
| text | `#ffffff` | `#e8edf2` | oklch(0.93 0.01 230) | — |
| muted | `#1a5f7a` | `#7a8fa3` | oklch(0.62 0.04 230) | ΔE=22 from text |
| label | `#00b894` | `#4a90d9` | oklch(0.65 0.15 225) | ΔE=18 from accent |
| ok | `#00b862` | `#2ecc71` | oklch(0.72 0.15 145) | ΔE=24 from primary |
| error | `#e74c3c` | `#e74c3c` | oklch(0.60 0.20 15) | — |
| warn | `#ff8c00` | `#f39c12` | oklch(0.78 0.15 80) | ΔE=22 from error |
| prompt | `#ffffff` | `#e8edf2` | oklch(0.93 0.01 230) | — |
| statusGood | `#00b862` | `#2ecc71` | oklch(0.72 0.15 145) | — |
| statusWarn | `#ff8c00` | `#f39c12` | oklch(0.78 0.15 80) | — |
| statusBad | `#FF8C00` | `#e67e22` | oklch(0.65 0.18 70) | ΔE=16 from warn |
| statusCritical | `#e74c3c` | `#c0392b` | oklch(0.55 0.22 15) | ΔE=18 from error |
| completionBg | `#0a1628` | `#0a1628` | — | Unchanged |
| completionCurrentBg | `#1a3050` | `#1a3050` | — | Unchanged |
| statusBg | `#0a1628` | `#0a1628` | — | Unchanged |
| statusFg | `#C0C0C0` | `#C0C0C0` | — | Unchanged |
| selectionBg | `#1a3050` | `#1a3050` | — | Unchanged |
| shellDollar | `#4dabf7` | `#4dabf7` | — | Unchanged |

### 4.2 New Hue Distribution

```
         0° (red)
         ↑
    error #e74c3c     (hue 15°)
         |
    statusCritical #c0392b (hue 15°) — same hue, lower L
         |
         |  ← 65° gap (SAFE)
    warn #f39c12       (hue 80°)
         |
    statusWarn #f39c12 (hue 80°) — same hue
         |
    statusBad #e67e22  (hue 70°) — 10° offset, lower L
         |
         |  ← 65° gap (SAFE)
    ok #2ecc71         (hue 145°)
    statusGood #2ecc71 (hue 145°)
         |
         |  ← 30° gap (acceptable — different L values)
    primary #00c9a7    (hue 175°)
         |
         |  ← 55° gap (SAFE)
    accent #5b9bd5     (hue 230°)
    label #4a90d9      (hue 225°) — 5° offset, lower L
         |
         ↓
        270° (blue)
```

Key improvements:
- Green-teal-cyan overcrowding eliminated: only `ok` (145°) and `primary` (175°)
  remain in that region, separated by 30° with different lightness values
- Blue accent/label region (225°–230°) is clean — no semantic confusion possible
- Red-orange region properly spaced: error (15°) → warn (80°) = 65° gap
- statusBad now has lower lightness than statusWarn, creating a luminance cue
  for severity escalation

### 4.3 New Contrast Ratios (vs dark background)

| Role | New Hex | Contrast Ratio | WCAG AA |
|------|---------|---------------|---------|
| text | `#e8edf2` | ~18:1 | ✅ PASS |
| primary | `#00c9a7` | ~11:1 | ✅ PASS |
| accent | `#5b9bd5` | ~7:1 | ✅ PASS |
| label | `#4a90d9` | ~6:1 | ✅ PASS |
| ok | `#2ecc71` | ~9:1 | ✅ PASS |
| error | `#e74c3c` | ~5.5:1 | ✅ PASS |
| warn | `#f39c12` | ~10:1 | ✅ PASS |
| **muted** | `#7a8fa3` | **~5.8:1** | **✅ PASS** |
| statusFg | `#C0C0C0` | ~10:1 | ✅ PASS |

**The muted color fix is the single most impactful change** — it goes from
failing WCAG AA by a factor of 2 to passing comfortably.

---

## 5. Design Rationale

### 5.1 Why Blue Accent Instead of Teal

The old palette used teal for both primary (brand) and accent (interactive).
This meant users couldn't distinguish "this is the brand color" from "this is
clickable/interactive." Moving accent to blue creates a clear semantic separation:
- **Teal** = brand identity (primary)
- **Blue** = interactive elements (accent, links, labels)
- **Green** = success (ok)
- **Orange** = warning (warn)
- **Red** = error/danger (error, critical)

This follows the conventional traffic-light pattern that users already understand.

### 5.2 Why Muted Went From Dark Blue-Gray to Medium Gray

The old muted (`#1a5f7a`) was chosen to be "subtle" but was too subtle — it
blended into the dark background. The new muted (`#7a8fa3`) is a medium
cool-gray that:
- Passes WCAG AA contrast (5.8:1)
- Still reads as "secondary" due to lower saturation than text
- Has a cool undertone that harmonizes with the blue accent
- Is distinguishable from text by both luminance and saturation

### 5.3 Why Background Colors Unchanged

The completion dropdown and status bar backgrounds (`#0a1628`, `#1a3050`)
already work well — they're dark enough to not distract, and the current
text on top of them has adequate contrast. Changing these would risk breaking
the visual hierarchy.

### 5.4 Why Error Kept the Same

`#e74c3c` is already a well-established red that works across colorblind types.
Changing it would only introduce risk without benefit. The statusCritical
variant (`#c0392b`) is a deeper red for the most severe state, creating a
luminance-based escalation without changing hue.

---

## 6. Implementation Plan

### 6.1 Files to Modify

1. **`ui-tui/src/theme.ts`** — Update `DARK_THEME.color` object with new hex values
2. **`hermes_cli/skin_engine.py`** — Update `default` skin colors dict to match

### 6.2 Files NOT to Modify

- `ui-tui/src/banner.ts` — Uses theme colors via `colorize()`, no hardcoded values
- `ui-tui/src/domain/roles.ts` — References `t.color.*`, no hardcoded values
- All component files (`branding.tsx`, `appChrome.tsx`, `messageLine.tsx`, etc.) —
  All reference `t.color.*`, no hardcoded values
- `ui-tui/src/components/prompts.tsx` — References `t.color.*`, no hardcoded values
- `ui-tui/src/components/thinking.tsx` — References `t.color.*`, no hardcoded values
- `ui-tui/src/components/markdown.tsx` — References `t.color.*`, no hardcoded values

### 6.3 Verification Steps

1. Update both files with new palette
2. Restart the TUI session
3. Take screenshot and visually verify:
   - Muted text is clearly readable
   - Accent (blue) is distinguishable from primary (teal)
   - Status bar shows proper color escalation
   - Semantic colors (ok/error/warn) are distinct
   - Overall visual harmony is maintained
4. If visual inspection passes, commit the changes

---

## 7. References

- WCAG 2.1 Level AA: https://www.w3.org/WAI/WCAG21/quickref/
- OKLCH color space: https://oklch.com/
- ColorBrewer for colorblind-safe palettes: https://colorbrewer2.org/
- CIEDE2000 Delta E: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
- Simulating colorblindness: https://www.color-blindness.com/

---

*Document created by Terra on May 20, 2026.*
