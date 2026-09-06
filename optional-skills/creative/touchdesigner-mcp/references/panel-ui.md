# Panel & UI Reference

Interactive control surfaces inside TouchDesigner — buttons, sliders, fields, custom parameter pages, panel callbacks. For HUD overlays (rendered text on visuals) see `layout-compositor.md`.

Use cases:
- VJ control rack (master fader, scene buttons, FX toggles)
- Installation operator console
- Self-contained TOX components with their own parameter UIs
- Phone-style touch interfaces displayed on a tablet

---

## Two Layers of UI

| Layer | What it is | Use for |
|---|---|---|
| **Custom Parameters** | Params on any COMP, edited like built-in TD params | Configurable components, presets, "settings" panels |
| **Panel COMPs** | Visible widgets (button, slider, field) inside a containerCOMP | Interactive control surfaces, real-time UIs |

Combine both: build a containerCOMP with panel widgets that read/write custom parameters on a parent component.

---

## Custom Parameters

Add user-editable params to any COMP. Params persist with the COMP, drive expressions, and survive save/reload.

```python
# Add a custom page to a baseCOMP
comp = op('/project1/my_component')
page = comp.appendCustomPage('Controls')

# Add typed params
page.appendFloat('Intensity', label='Intensity')[0]   # returns a Par
page.appendInt('Count', label='Count')[0]
page.appendToggle('Enabled', label='Enabled')[0]
page.appendMenu('Mode', menuNames=['off', 'soft', 'hard'], menuLabels=['Off', 'Soft', 'Hard'])[0]
page.appendStr('Title', label='Title')[0]
page.appendRGB('Color', label='Color')                # returns 3 pars
page.appendXY('Offset', label='Offset')               # returns 2 pars
page.appendPulse('Reset', label='Reset')[0]
page.appendFile('TextureFile', label='Texture')[0]
```

**Read/write from anywhere:**

```python
val = op('/project1/my_component').par.Intensity.eval()
op('/project1/my_component').par.Intensity = 0.7
```

**Drive other params via expression:**

```python
op('bloom1').par.threshold.mode = ParMode.EXPRESSION
op('bloom1').par.threshold.expr = "op('/project1/my_component').par.Intensity"
```

**Pulse handler (Reset button):**

Use a `parameterExecuteDAT` watching the COMP's pulse params. See `dat-scripting.md`.

---

## Panel COMPs — The Widgets

Each is a COMP that renders as a clickable/draggable widget inside a `containerCOMP`.

| Type | Type Name | Use |
|---|---|---|
| Button | `buttonCOMP` | Click action — momentary or toggle |
| Slider | `sliderCOMP` | Drag to set 0-1 value (1D or 2D) |
| Field | `fieldCOMP` | Text input |
| Container | `containerCOMP` | Layout + visual styling, holds children |
| Select | `selectCOMP` | Reference and display content from another COMP |
| List | `listCOMP` | Scrollable list with row callbacks |

### Button

```python
btn = root.create(buttonCOMP, 'play_btn')
btn.par.w = 120; btn.par.h = 40
btn.par.buttontype = 'momentary'    # 'momentary' | 'toggleup' | 'togglepress' | 'radio'
btn.par.bgcolorr = 0.1; btn.par.bgcolorg = 0.1; btn.par.bgcolorb = 0.1
btn.par.text = 'Play'

# Read state
state = btn.panel.state          # 1 when active
```

### Slider

```python
sld = root.create(sliderCOMP, 'master_fader')
sld.par.w = 60; sld.par.h = 300
sld.par.style = 'vertical'        # 'vertical' | 'horizontal' | 'xy'
sld.par.value0min = 0.0
sld.par.value0max = 1.0

# Drive a parameter via expression (always-on, no callback needed)
op('/project1/master_level').par.opacity.mode = ParMode.EXPRESSION
op('/project1/master_level').par.opacity.expr = "op('master_fader').panel.u"
```

`panel.u` and `panel.v` give the 0-1 normalized values. For 2D sliders both are populated.

### Field (Text Input)

```python
fld = root.create(fieldCOMP, 'scene_name')
fld.par.w = 200; fld.par.h = 30
fld.par.fieldtype = 'string'      # 'string' | 'integer' | 'float'

# Read current text
text = fld.panel.field            # the text content
```

### List

For scrollable lists with selectable rows, use the docked `list1_callbacks` DAT to handle row interactions. Set up cells via the `list_definition` table DAT.

---

## Container COMP — Layout & Styling

`containerCOMP` is the primary parent for grouping widgets and arranging layouts.

```python
panel = root.create(containerCOMP, 'control_panel')
panel.par.w = 400; panel.par.h = 600
panel.par.bgcolorr = 0.05
panel.par.bgcolorg = 0.05
panel.par.bgcolorb = 0.05
panel.par.bgalpha = 1.0

# Layout child panels in vertical stack
panel.par.align = 'lefttoright'   # 'lefttoright' | 'toptobottom' | etc.
```

Children are positioned automatically based on `par.align`. For absolute positioning use `par.align = 'fillresize'` and set each child's `par.x` / `par.y`.

### Layout Strategies

| `par.align` | Behavior |
|---|---|
| `lefttoright` | Children stacked horizontally |
| `toptobottom` | Children stacked vertically |
| `righttoleft` / `bottomtotop` | Reversed stacks |
| `fillresize` | Children sized to fill, manual positioning |
| `top` / `bottom` / `left` / `right` | Fixed positioning |

For complex grids: nest containers — vertical container holding horizontal containers.

---

## Panel Callbacks — Reacting to Events

`panelExecuteDAT` watches a panel and fires Python callbacks on user interaction.

```python
pe = root.create(panelExecuteDAT, 'btn_handler')
pe.par.panel = '/project1/play_btn'
pe.par.click = True              # respond to clicks
pe.par.value = True              # respond to value changes
```

In its docked DAT:

```python
def onOffToOn(panelValue):
    # Click pressed
    op('/project1/scene_timer').par.start.pulse()
    return

def onOnToOff(panelValue):
    # Click released
    return

def onValueChange(panelValue):
    # Slider drag, field change, etc.
    new_val = panelValue.eval()
    op('/project1/master').par.opacity = new_val
    return
```

For pulse params on custom-parameter pages, use a `parameterExecuteDAT` instead.

---

## Building a Complete VJ Control Panel

End-to-end pattern:

```python
# 1. Top-level container
panel = root.create(containerCOMP, 'vj_control')
panel.par.w = 800; panel.par.h = 200
panel.par.align = 'lefttoright'

# 2. Master fader column
master_col = panel.create(containerCOMP, 'master')
master_col.par.w = 120; master_col.par.h = 200
master_col.par.align = 'toptobottom'

master_label = master_col.create(textTOP, 'lbl')
master_label.par.text = 'MASTER'

master_sld = master_col.create(sliderCOMP, 'fader')
master_sld.par.w = 60; master_sld.par.h = 150
master_sld.par.style = 'vertical'

# 3. Scene buttons row
scene_col = panel.create(containerCOMP, 'scenes')
scene_col.par.w = 400; scene_col.par.h = 200
scene_col.par.align = 'lefttoright'
for i in range(8):
    b = scene_col.create(buttonCOMP, f'scene_{i+1}')
    b.par.w = 50; b.par.h = 50
    b.par.text = str(i+1)
    b.par.buttontype = 'radio'      # only one active at a time

# 4. FX toggle column
fx_col = panel.create(containerCOMP, 'fx')
fx_col.par.w = 280; fx_col.par.h = 200
fx_col.par.align = 'toptobottom'
for fx in ['Bloom', 'CRT', 'Glitch', 'Strobe']:
    t = fx_col.create(buttonCOMP, fx.lower())
    t.par.w = 220; t.par.h = 35
    t.par.text = fx
    t.par.buttontype = 'toggleup'

# 5. Display in a window
win = root.create(windowCOMP, 'control_win')
win.par.winop = panel.path
win.par.winw = 800; win.par.winh = 200
win.par.borders = True
win.par.winopen.pulse()
```

Then wire panel values to ops via expressions or panelExecuteDATs.

---

## Showing the Panel — Window or Embedded

| Approach | When |
|---|---|
| `windowCOMP` pointing at panel | Standalone control surface, separate display |
| Render the containerCOMP via `renderTOP` | Composite UI over visuals (HUD-style) |
| Use a `panelCOMP` directly inside a network editor pane | Designer/dev preview only — panel is fully interactive |

For a touch-screen tablet, use a `windowCOMP` on a second display routed to the tablet's HDMI input.

---

## Pitfalls

1. **Panel won't respond to clicks** — likely `par.disabled = True` or the parent container has `par.disableinputs = True`. Check the panel hierarchy.
2. **Slider value not updating** — `panel.u/v` reads the visual position. If you set `par.value0` directly, the visual lags. Use `par.value0` AS the source of truth and let the slider follow.
3. **Custom param won't appear** — must call `appendCustomPage` first, then append params. Pages with no params don't show.
4. **Custom param disappears on reload** — params added via Python at runtime persist only if the COMP is saved AFTER. Use a `tox` save (`comp.save('mycomp.tox')`) or commit via `td_execute_python` then save the project.
5. **Event callback fires twice** — both `onOffToOn` and `onValueChange` may fire on a single button press. Pick one to handle the action; don't double-trigger.
6. **Pulse params need `.pulse()`** — setting `par.X = True` on a pulse param does nothing. Always use `.pulse()`.
7. **Field text doesn't commit until Tab/Enter** — fields don't fire callbacks while typing. Use `par.committemode = 'all'` to fire on every keystroke (heavy).
8. **`par.text` vs panel content** — `buttonCOMP.par.text` is the LABEL on the button. The button's STATE is `panel.state` (0/1). Don't confuse them.
9. **Touch input on macOS** — multi-touch via direct touch panels works but TD's gesture handling is rudimentary. For complex multi-touch (pinch/rotate), use TouchOSC on a tablet instead.
10. **Layout doesn't update** — changing `par.align` requires the container to re-cook. Touch a child or pulse the container to trigger.

---

## Quick Recipes

| Goal | Setup |
|---|---|
| Master fader | `sliderCOMP` (vertical) → expression on `level.par.opacity` |
| Scene picker | 8 `buttonCOMP` (radio) → `selectCHOP` on their state → drive `switchTOP.par.index` |
| FX toggle | `buttonCOMP` (toggleup) → expression on `bypass` of an FX op |
| Numeric input | `fieldCOMP` (float) → expression on target par |
| Component settings | Custom params on the component COMP, panel widgets inside drive them |
| Touch tablet UI | `containerCOMP` with widgets → `windowCOMP` to second display |
| Status display | `textTOP` rendered into the panel via `selectCOMP` |
