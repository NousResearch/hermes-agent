# Hermes Caption Agent — Plan v6 (Caption Positioning)

**Hackathon**: Hermes Agent Creative Hackathon  
**Prize pool**: $25k (Main $15k + Kimi Track $5k)  
**Deadline**: EOD Sunday May 3rd, 2026  
**Previous plans**:
- `PLAN_v1.md` — Telegram-native pipeline (executed)
- `PLAN_v2.md` — Dashboard core-edit approach (archived)
- `PLAN_v3.md` — Plugin architecture, editing + burn (executed)
- `PLAN_v4.md` — Hermes-integrated features: upload, NL edits, QA, style memory (executed)
- `PLAN_v5.md` — 3-Column Editor + Named Preset Library (executed)

---

## What Changed from v5

v5 shipped the 3-column editor with the Hermes panel (preset gallery, AI style creation, QA).
Caption positioning was limited to a `margin_bottom` number field — the `alignment` field existed
in the data model and burn logic but was never exposed in the UI.

Plan v6 adds **two complementary positional controls**:
1. **Alignment picker** — a 3×3 grid that sets the caption anchor zone (coarse)
2. **Drag-to-position overlay** — drag captions anywhere on the video player (fine)

---

## New Capabilities

### 1. Alignment Picker (3×3 Grid)

Replace the hidden `alignment` int field with a visual 9-button grid in the Caption Style section.
Each button represents one of the 9 ASS numpad positions (↖ ↑ ↗ / ← · → / ↙ ↓ ↘).

- Clicking a button sets `style.alignment` (1–9)
- The active button is highlighted
- Default: button 2 (bottom-center, same as current default)
- When a drag position is active, the alignment picker dims slightly (still usable — sets anchor)

### 2. Global Drag-to-Position Overlay

An invisible drag target overlaid on the Col 1 video player. Dragging stores a normalised
`position: { x: number, y: number }` (0.0–1.0) in the style object. At burn time the plugin
converts the normalised coords to pixel coords (video resolution) and injects `{\an<N>\pos(x,y)}`
into each dialogue line.

- A small draggable handle (the caption text rendered as a ghost) sits on the video at the
  stored position
- Dragging it moves the handle in real time
- A "reset" (×) button removes the override and falls back to alignment + margin_v
- If `position` is null/absent the burn behaves exactly as before (no regression)

### 3. Rename "Bottom margin" → "Margin from edge"

The label `margin_bottom` is misleading when alignment is top or middle. Rename to
"Margin from edge" in the UI (backend field name stays `margin_bottom` for compatibility).

---

## Data Model Changes

### `CaptionStyle` — TypeScript

```typescript
interface CaptionStyle {
  font: string;
  font_size: number;
  primary_color: string;
  outline_color: string;
  outline_width: number;
  alignment: number;          // 1–9, ASS numpad — now exposed in UI
  margin_bottom: number;
  max_line_length: number;
  position?: { x: number; y: number } | null;   // NEW: normalised 0.0–1.0, null = use alignment+margin
}
```

### `_DEFAULT_STYLE` — Python (tools/video_caption.py)

No change — `position` defaults to absent/null and the burn code handles it gracefully.

---

## Burn Logic Change (tools/video_caption.py)

In `_build_ass_content()`, after building dialogue text:

```python
pos = style.get("position")
if pos:
    vid_w = video_width   # passed in or read from probe
    vid_h = video_height
    px = int(pos["x"] * vid_w)
    py = int(pos["y"] * vid_h)
    an = int(style.get("alignment", 2))
    position_tag = f"{{\\an{an}\\pos({px},{py})}}"
else:
    position_tag = ""

# prepend to each Dialogue text field:
f"Dialogue: 0,{start},{end},MAIN,,0,0,0,,{position_tag}{text}"
f"Dialogue: 0,{start},{end},PHONETIC,,0,0,0,,{position_tag}{phonetic}"
```

The video dimensions are already accessible because `_build_ass_content` receives the
`PlayResX` / `PlayResY` values — or we add a probe step if not already present.

---

## UI Changes (index.tsx)

### AlignmentPicker component

```tsx
const ALIGNMENT_GRID = [7, 8, 9, 4, 5, 6, 1, 2, 3];  // row-major, top→bottom

function AlignmentPicker({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="grid grid-cols-3 gap-0.5 w-24">
      {ALIGNMENT_GRID.map(n => (
        <button
          key={n}
          onClick={() => onChange(n)}
          className={`h-7 w-7 rounded text-xs border ${
            value === n
              ? 'bg-amber-500 text-white border-amber-500'
              : 'bg-muted text-muted-foreground border-border hover:bg-accent'
          }`}
        >
          {ALIGNMENT_ICONS[n]}
        </button>
      ))}
    </div>
  );
}
```

Arrow icons: `['↙','↓','↘','←','·','→','↖','↑','↗']` mapped by ASS numpad number.

### DragPositionOverlay component

Overlaid on the `<video>` element in Col 1 (absolute-positioned sibling inside a `relative` wrapper):

- If `style.position` is set: renders a small pill showing the caption text at
  `left: x*100%`, `top: y*100%` — draggable via `onMouseDown` / `onMouseMove` on the
  overlay div
- If not set: shows a subtle dashed border + "Drag to pin position" hint text when
  hovered
- A "× Reset position" link appears below the video when position is set

### Caption Style section layout (Col 2)

```
Font               [input]
Font size          [number]
Text color         [swatch]
Outline color      [swatch]
Outline width      [number]
Alignment          [3×3 grid]
Margin from edge   [number]
```

`position` is not in this form — it's controlled purely by the drag overlay.

---

## Architecture (v6)

No new routes. The `position` field travels through the existing `PUT /jobs/{id}/style`
endpoint as part of the style dict. The burn endpoint already passes the full style to
`_build_ass_content()`.

```
Col 1 video player
  └─ DragPositionOverlay (absolute, z-10)
       ├─ ghostCaption handle (draggable)
       └─ "× Reset position" link (below video)

Col 2 Caption Style section
  └─ AlignmentPicker (new, replaces hidden field)
  └─ StyleNumberField "Margin from edge"  (renamed)

PUT /jobs/{id}/style  ←→  style dict (now may include `position`)
                                ↓
            _build_ass_content()  →  {\anN\pos(x,y)} prefix on Dialogue lines
```

---

## Files Changed (v6)

| File | Change |
|---|---|
| `tools/video_caption.py` | `_build_ass_content()`: inject `\anN\pos(x,y)` when `style["position"]` is set; resolve video dims via `PlayResX/Y` |
| `plugins/phonetic-captions/dashboard/src/index.tsx` | Add `AlignmentPicker` component; add `DragPositionOverlay` component; add `position` to `CaptionStyle` type; rename "Bottom margin" label; wire drag state into `localStyle` |
| `plugins/phonetic-captions/dashboard/dist/index.js` | Rebuilt bundle |

No backend route changes, no Pydantic model changes, no core files touched.

---

## Testing Flows

### Test I — Alignment picker

1. Open any job in the editor
2. Caption Style → Alignment grid → click top-center button (8)
3. Re-burn → captions appear at top-center of video

### Test J — Drag to position

1. Open any job → hover Col 1 video → dashed border + hint text appears
2. Drag caption handle to upper-right area
3. "× Reset position" link appears below video
4. Re-burn → captions appear at dragged position
5. Click "× Reset position" → handle disappears; next burn uses alignment+margin only

### Test K — Alignment + drag combined

1. Set alignment to top-left (7) via the picker
2. Drag to a custom position
3. Burn: captions use `\an7\pos(x,y)` — anchor is top-left of the text block at (x,y)

### Test L — No regression

1. Open a job that was burned before v6 (no `position` in style)
2. Re-burn without touching positioning controls → output identical to v5

---

## Submission Checklist (EOD Sunday)

- [ ] Alignment picker renders and sets `style.alignment` correctly
- [ ] Drag overlay appears on the video in Col 1
- [ ] Dragging moves the ghost caption handle in real time
- [ ] "× Reset position" removes the override
- [ ] Re-burn with position set injects `\pos` tags in ASS output
- [ ] Re-burn without position set is identical to v5 (no regression)
- [ ] Alignment picker + drag work together (anchor point respected)
- [ ] "Margin from edge" label is correct
- [ ] All v5 checklist items still pass
