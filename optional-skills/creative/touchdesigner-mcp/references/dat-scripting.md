# DAT-Based Scripting Reference

TD's event/callback model — Python that runs in response to network events. The full set of "Execute DATs" plus their idiomatic patterns.

For arbitrary Python execution (not callback-based), see `python-api.md`. For the MCP's `td_execute_python` tool, see `mcp-tools.md`.

---

## The Execute DAT Family

Every type watches one kind of event source and fires Python on changes.

| DAT | Watches | Use for |
|---|---|---|
| `chopExecuteDAT` | A CHOP's channel values | Audio triggers, threshold callbacks, state machines on numeric input |
| `datExecuteDAT` | A DAT's content (table cells, text) | Reacting to data updates from APIs, parsing webDAT responses |
| `parameterExecuteDAT` | A parameter's value or pulse | Reacting to user-changed params, custom pulse buttons |
| `panelExecuteDAT` | A panel COMP's interaction | Button clicks, slider drags, field commits |
| `opExecuteDAT` | Operator lifecycle | New operator created, deleted, name changed |
| `executeDAT` | Project lifecycle, frame events | Run-once setup, per-frame logic, save/load hooks |

All have a docked DAT with predefined callback functions. You only fill in the bodies of the ones you care about.

---

## chopExecuteDAT — Numeric Triggers

```python
ce = root.create(chopExecuteDAT, 'kick_handler')
ce.par.chop = '/project1/audio/out_kick'      # source CHOP
ce.par.offtoon = True                          # fire when channel rises above 0
ce.par.ontooff = False
ce.par.whileon = False
ce.par.valuechange = False
```

In the docked callback DAT:

```python
def offToOn(channel, sampleIndex, val, prev):
    """Channel went from 0 to non-zero. Classic beat trigger."""
    op('/project1/strobe').par.flash.pulse()
    op('/project1/scene').par.index = (op('/project1/scene').par.index + 1) % 8
    return

def onToOff(channel, sampleIndex, val, prev):
    """Channel went from non-zero to 0."""
    return

def whileOn(channel, sampleIndex, val, prev):
    """Fires every frame while channel is non-zero. Use sparingly."""
    return

def valueChange(channel, sampleIndex, val, prev):
    """Fires every frame the value changes (continuous). Heavy."""
    return
```

`channel` is a `Channel` object — `.name`, `.owner`, `.vals[]`. Use `channel.name == 'chan1'` to filter.

**Threshold-based custom triggers:** wire the source CHOP through a `triggerCHOP` first to get clean 0/1 pulses, then watch with `offtoon`.

---

## datExecuteDAT — Table/Text Changes

```python
de = root.create(datExecuteDAT, 'api_response')
de.par.dat = '/project1/api/web1'              # source DAT
de.par.tablechange = True                      # any cell change
de.par.cellchange = False
de.par.rowchange = False
de.par.colchange = False
```

```python
def onTableChange(dat):
    """Whole table changed (including text DAT content updates)."""
    if dat.numRows == 0:
        return
    # If it's a webDAT response, parse JSON
    import json
    try:
        data = json.loads(dat.text)
    except json.JSONDecodeError:
        debug(f'Bad JSON: {dat.text[:100]}')
        return
    # Write to a CHOP
    op('/project1/api_value').par.value0 = float(data.get('count', 0))
    return

def onCellChange(dat, cells, prev):
    """Specific cells changed."""
    for cell in cells:
        # cell.row, cell.col, cell.val
        pass
    return
```

`debug()` prints to the textport — readable via `td_read_textport`.

---

## parameterExecuteDAT — Param Changes & Pulse

```python
pe = root.create(parameterExecuteDAT, 'comp_params')
pe.par.op = '/project1/my_component'           # COMP whose params to watch
pe.par.parameters = '*'                         # or specific names like 'Intensity Reset'
pe.par.valuechange = True
pe.par.pulse = True
```

```python
def onValueChange(par, prev):
    """par is a Par object. par.name, par.eval(), par.owner."""
    if par.name == 'Intensity':
        op('/project1/bloom').par.threshold = par.eval()
    return

def onPulse(par):
    """Pulse param was triggered."""
    if par.name == 'Reset':
        op('/project1/scene').par.index = 0
        op('/project1/audio_player').par.cuepoint = 0
        op('/project1/audio_player').par.cuepulse.pulse()
    return

def onExpressionChange(par, val, prev):
    """User changed the expression on a param."""
    return

def onExportChange(par, val, prev):
    """Export source changed."""
    return

def onModeChange(par, val, prev):
    """Param mode changed (CONSTANT / EXPRESSION / EXPORT / etc)."""
    return
```

---

## panelExecuteDAT — UI Events

For interactive control surfaces. See `panel-ui.md` for the full panel COMP context.

```python
pe = root.create(panelExecuteDAT, 'btn_handler')
pe.par.panel = '/project1/play_btn'
pe.par.click = True              # mouse click events
pe.par.value = True              # state changes (toggle)
pe.par.lockedchange = False
```

```python
def onOffToOn(panelValue):
    """Panel value rose to 1 (button pressed, slider crossed threshold)."""
    op('/project1/scene_timer').par.start.pulse()
    return

def onOnToOff(panelValue):
    """Panel value dropped to 0."""
    return

def onValueChange(panelValue):
    """Continuous: every frame the value changes."""
    val = panelValue.eval()
    op('/project1/master').par.opacity = val
    return

def onClick(panelValue):
    """Discrete click event, fires once per click."""
    return
```

`panelValue` is a `Par` object on the panel COMP.

---

## opExecuteDAT — Operator Lifecycle

Watches creation/deletion/renaming of operators in a parent COMP.

```python
oe = root.create(opExecuteDAT, 'lifecycle')
oe.par.op = '/project1'
oe.par.create = True
oe.par.destroy = True
oe.par.namechange = True
oe.par.flagchange = False
```

```python
def onCreate(opCreated):
    """A new operator was created. Useful for auto-applying conventions."""
    if opCreated.OPType == 'glslTOP':
        # Always wrap with a null
        n = opCreated.parent().create(nullTOP, opCreated.name + '_out')
        n.inputConnectors[0].connect(opCreated)
    return

def onDestroy(opDestroyed):
    """Operator was deleted. opDestroyed.path is still valid for one frame."""
    return

def onNameChange(opChanged):
    """Operator was renamed."""
    return
```

Useful for dev-time scaffolding (auto-create downstream nullTOPs, auto-name conventions). Disable in production projects to avoid surprise side effects.

---

## executeDAT — Project Lifecycle & Per-Frame

The catch-all. Gets you hooks into project start, save, load, frame-start, frame-end.

```python
exec_dat = root.create(executeDAT, 'lifecycle')
exec_dat.par.start = True
exec_dat.par.create = True
exec_dat.par.framestart = True
exec_dat.par.frameend = False
```

```python
def onStart():
    """Project just started cooking. Run once."""
    op('/project1/scene').par.index = 0
    debug('Project started')
    return

def onCreate():
    """Component was just created (only fires for component executeDATs, not project root)."""
    return

def onFrameStart(frame):
    """Per-frame, BEFORE network cooks. Heavy logic here = bottleneck."""
    return

def onFrameEnd(frame):
    """Per-frame, AFTER network cooks. Use for capture, recording, post-network logic."""
    return

def onPlayStateChange(playing):
    """Project play/pause toggled."""
    return

def onProjectPreSave():
    """Right before saving the .toe file."""
    return

def onProjectPostSave():
    return
```

Heavy per-frame logic in `onFrameStart` is one of the top performance regressions in TD projects. Use CHOPs for per-frame computation, scripts for events.

---

## Pattern: Triggering an Animation Sequence on Beat

```python
# Source: a kick trigger CHOP
# Goal: on each kick, run a 1.5s scale pulse + color flash

# Setup (create once)
animator = root.create(timerCHOP, 'pulse_anim')
animator.par.length = 1.5
animator.par.cycle = False

# Param expressions on visual targets:
op('logo').par.sx.expr = "1.0 + (1 - op('pulse_anim')['timer_fraction']) * 0.3"
op('logo').par.sx.mode = ParMode.EXPRESSION
op('logo').par.sy.expr = "1.0 + (1 - op('pulse_anim')['timer_fraction']) * 0.3"
op('logo').par.sy.mode = ParMode.EXPRESSION

# In a chopExecuteDAT watching the kick CHOP:
def offToOn(channel, sampleIndex, val, prev):
    op('pulse_anim').par.start.pulse()
    return
```

---

## Pattern: Live Editing a CHOP from API Data

```python
# webDAT polls an API every 5 seconds
# datExecuteDAT parses the response and writes to a constantCHOP

def onTableChange(dat):
    import json
    try:
        data = json.loads(dat.text)
    except:
        return
    target = op('/project1/external_state')
    target.par.name0 = 'temperature'
    target.par.value0 = float(data['temp_c'])
    target.par.name1 = 'humidity'
    target.par.value1 = float(data['humidity'])
    return
```

Visuals just reference `op('external_state')['temperature']` — they update live.

---

## Pattern: Self-Cleaning Network

```python
# An opExecuteDAT watching for orphaned helper ops, deleting them after their parent disappears

def onDestroy(opDestroyed):
    parent_name = opDestroyed.name
    helper = op(f'/project1/{parent_name}_helper')
    if helper:
        helper.destroy()
    return
```

---

## Pitfalls

1. **Callbacks crash silently** — exceptions print to the textport but don't show up in the UI. Always `td_clear_textport` before debugging, then `td_read_textport` after.
2. **`debug()` vs `print()`** — both write to textport, but `debug()` includes the file/line of the calling DAT. Prefer `debug()` for scripts.
3. **`val` is the new value, `prev` is old** — easy to swap. Always: `def offToOn(channel, sampleIndex, val, prev)`. Check parameter order in TD docs if confused.
4. **`whileOn` and `valueChange` are per-frame** — heavy. Avoid unless absolutely needed. Drive via expressions instead.
5. **Callbacks don't run during cooking-paused state** — if the parent COMP has `allowCooking=False`, callbacks freeze. Useful for "disable me" toggles.
6. **`par` vs `panelValue`** — parameterExecuteDAT gives `par` (a Par object), panelExecuteDAT gives `panelValue` (also a Par-like object). Both have `.name` and `.eval()` but their context differs.
7. **`opExecuteDAT` fires for itself** — when you create an opExecuteDAT, it can fire `onCreate` for itself if `par.create=True` and parent matches. Filter by `if opCreated == me: return`.
8. **Reload behavior** — when reloading an extension (`td_reinit_extension`), all callback DATs reset their internal state. Module-level vars are lost. Persist state in tableDATs or the docked DAT itself, not in module globals.
9. **Cooking dependencies** — if a callback writes to an op that's upstream of the callback's source, you get a cooking loop. TD warns about it but doesn't always block. Keep dataflow one-directional.
10. **Active flag** — every Execute DAT has `par.active`. False = silent. Easy to toggle for testing without deleting wiring.

---

## Quick Recipes

| Goal | Setup |
|---|---|
| Beat trigger | `chopExecuteDAT.par.offtoon=True` watching a `triggerCHOP` |
| API response handler | `datExecuteDAT.par.tablechange=True` watching a `webDAT` |
| Custom button → action | `parameterExecuteDAT.par.pulse=True` watching a custom pulse param |
| Slider → continuous param | `panelExecuteDAT.par.value=True` watching a `sliderCOMP` |
| Run-once setup | `executeDAT.par.start=True` with logic in `onStart()` |
| Per-frame metrics | `executeDAT.par.frameend=True` recording values to a CHOP |
| Auto-name new ops | `opExecuteDAT.par.create=True` enforcing naming conventions |
