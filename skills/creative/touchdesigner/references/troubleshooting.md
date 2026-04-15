# TouchDesigner Troubleshooting

> See `references/pitfalls.md` for the comprehensive lessons-learned list.

## Quick Connection Diagnostic

```bash
lsof -i :9981 -P -n | grep LISTEN    # Step 1: Is TD listening?
curl -s http://127.0.0.1:9981/api/td/server/td   # Step 2: API working?
```

| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused | No WebServer DAT | Deploy `scripts/custom_api_handler.py` in TD Textport |
| HTTP 404 on all routes | .tox module import failed | Deploy custom handler (pitfalls #1-2) |
| HTTP 200, empty body | Response in wrong key | Handler uses `response['data']` not `response['body']` (pitfalls #6) |
| HTTP 200, JSON body | Working | Proceed to discovery |
| MCP tools not callable | Normal — use curl instead | `td_exec()` pattern in SKILL.md works without MCP |

## Node Creation Issues

### "Node type not found" error

**Cause:** Wrong `nodeType` string in `create_td_node`.

**Fix:** Use camelCase with family suffix. Common mistakes:
- Wrong: `NoiseTop`, `noise_top`, `NOISE TOP`, `Noise`
- Right: `noiseTop`
- Wrong: `AudioSpectrum`, `audio_spectrum_chop`
- Right: `audiospectrumChop`

**Discovery method:** Use `get_td_classes` to see available types, or `execute_python_script` with `dir(td)` filtered for operator classes.

### Node created but not visible in TD

**Cause:** Node was created in a different container than expected, or TD viewport is looking at a different network.

**Fix:** Check `parentPath` — use absolute paths like `/project1`. Verify with `get_td_nodes(parentPath="/project1")`.

### Cannot create node inside a non-COMP

**Cause:** Only COMP operators (Container, Base, Geometry, etc.) can contain child operators. You cannot create nodes inside a TOP, CHOP, SOP, DAT, or MAT.

**Fix:** Create a Container COMP or Base COMP first, then create nodes inside it.

## Parameter Issues

### Parameter not updating

**Causes:**
1. **Wrong parameter name.** TD parameter names change across versions. Run the discovery script (SKILL.md Step 0) or use `get_td_node_parameters` to discover exact names for your TD version. Never trust online docs or this skill's tables — always verify.
2. **Parameter is read-only.** Some parameters are computed/locked.
3. **Wrong value type.** Menu parameters need integer index or exact string label.
4. **Parameter has an expression.** If `node.par.X.expr` is set, `.val` is ignored. Clear the expression first.

**Discovery-based approach (preferred):**
```python
execute_python_script(script="""
n = op('/project1/mynode')
pars = [(p.name, type(p.val).__name__, p.val) for p in n.pars()
        if any(k in p.name.lower() for k in ['color', 'size', 'dat', 'font', 'alpha'])]
result = pars
""")
```

**Safe parameter setter pattern:**
```python
def safe_par(node, name, value):
    p = getattr(node.par, name, None)
    if p is not None:
        p.val = value
        return True
    return False  # param doesn't exist in this TD version
```

### Common parameter name gotchas

| What you expect | Actual name | Notes |
|----------------|-------------|-------|
| `width` | `resolutionw` | TOP resolution width |
| `height` | `resolutionh` | TOP resolution height |
| `filepath` | `file` | File path parameter |
| `color` | `colorr`, `colorg`, `colorb`, `colora` | Separate RGBA components |
| `position_x` | `tx` | Translate X |
| `rotation` | `rz` | Rotate Z (2D rotation) |
| `scale` | `sx`, `sy` | Separate X/Y scale |
| `blend_mode` | `operand` | Composite TOP blend mode (integer) |
| `opacity` | `opacity` | On Level TOP (this one is correct!) |

### Composite TOP operand values

| Mode | Index |
|------|-------|
| Over | 0 |
| Under | 1 |
| Inside | 2 |
| Add | 3 |
| Subtract | 4 |
| Difference | 5 |
| Multiply | 18 |
| Screen | 27 |
| Maximum | 13 |
| Minimum | 14 |
| Average | 28 |

## Connection/Wiring Issues

### Connections not working

**Causes:**
1. **Cross-family wiring.** TOPs can only connect to TOPs, CHOPs to CHOPs, etc. Use converter operators to bridge families.
2. **Wrong connector index.** Most operators have one output connector (index 0). Multi-output operators may need index 1, 2, etc.
3. **Node path wrong.** Verify paths are absolute and correctly spelled.

**Verify connections:**
```python
execute_python_script(script="""
node = op('/project1/level1')
result = {
    'inputs': [i.path if i else None for i in node.inputs],
    'outputs': [o.path if o else None for o in node.outputs]
}
""")
```

### Feedback loops causing errors

**Symptom:** "Circular dependency" or infinite cook loop.

**Fix:** Always use a Feedback TOP (or a Null TOP with a one-frame delay) to break the loop:
```
A -> B -> Feedback(references B) -> A
```
Never create A -> B -> A directly.

## Performance Issues

### Low FPS / choppy output

**Common causes and fixes:**

1. **Resolution too high.** Start at 1920x1080, only go higher if GPU handles it.
2. **Too many operators.** Each operator has GPU/CPU overhead. Consolidate where possible.
3. **Expensive shader.** GLSL TOPs with complex math per-pixel drain GPU. Profile with TD's Performance Monitor (F2).
4. **No GPU instancing.** Rendering 1000 separate geometry objects is much slower than 1 instanced geometry.
5. **Unnecessary cooks.** Operators that don't change frame-to-frame still recook if inputs change. Use Null TOPs to cache stable results.
6. **Large texture transfers.** TOP to CHOP and CHOP to TOP involve GPU-CPU memory transfers. Minimize these.

**Performance Monitor:**
```python
execute_python_script(script="td.performanceMonitor = True")
# After testing:
execute_python_script(script="td.performanceMonitor = False")
```

### Memory growing over time

**Causes:**
- Cache TOPs with high `length` value
- Feedback loops without brightness decay (values accumulate)
- Table DATs growing without clearing
- Movie File In loading many unique frames

**Fix:** Always add slight decay in feedback loops (Level TOP with `opacity=0.98` or multiply blend). Clear tables periodically.

## Export / Recording Issues

### Movie File Out not recording

**Checklist:**
1. Is the `record` parameter toggled on? `update_td_node_parameters(properties={"record": true})`
2. Is an input connected? The Movie File Out needs a TOP input.
3. Is the output path valid and writable? Check `file` parameter.
4. Is the codec available? H.264 (type 4) is most reliable.

### Exported video is black

**Causes:**
1. The TOP chain output is all black (brightness too low).
2. The input TOP has errors (check with `get_td_node_errors`).
3. Resolution mismatch — the output may be wrong resolution.

**Debug:** Check the input TOP's actual pixel values:
```python
execute_python_script(script="""
import numpy as np
top = op('/project1/out')
arr = top.numpyArray(delayed=True)
result = {'mean': float(arr.mean()), 'max': float(arr.max()), 'shape': list(arr.shape)}
""")
```

### .tox export losing connections

**Note:** When saving a component as .tox, only the component and its internal children are saved. External connections (wires to operators outside the component) are lost. Design self-contained components.

## Python Scripting Issues

### execute_python_script returns empty result

**Causes:**
1. The script used `exec()` semantics (multi-line) but didn't set `result`.
2. The last expression has no return value (e.g., `print()` returns None).

**Fix:** Explicitly set `result`:
```python
execute_python_script(script="""
nodes = op('/project1').findChildren(type=TOP)
result = len(nodes)  # explicitly set return value
""")
```

### Script errors not clear

**Check stderr in the response.** The MCP server captures both stdout and stderr from script execution. Error tracebacks appear in stderr.

### Module not found in TD Python

**Cause:** TD's Python environment may not have the module. TD bundles numpy, scipy, opencv, Pillow, and requests. Other packages need manual installation.

**Check available packages:**
```python
execute_python_script(script="""
import sys
result = [p for p in sys.path]
""")
```

## Common Workflow Pitfalls

### Building before verifying connection

Always call `get_td_info` first. If TD isn't running or the WebServer DAT isn't loaded, all subsequent tool calls will fail.

### Not checking errors after building

Always call `get_td_node_errors(nodePath="/project1")` after creating and wiring a network. Broken connections and missing references are silent until you check.

### Creating too many operators in one go

When building complex networks, create in logical groups:
1. Create all operators in a section
2. Wire that section
3. Verify with `get_td_node_errors`
4. Move to the next section

Don't create 50 operators, wire them all, then discover something was wrong 30 operators ago.

### Parameter expressions vs static values

If you set `node.par.X.val = 5` but there's an expression on that parameter (`node.par.X.expr`), the expression wins. To use a static value, clear the expression first:
```python
execute_python_script(script="""
op('/project1/noise1').par.seed.expr = ''  # clear expression
op('/project1/noise1').par.seed.val = 42   # now static value works
""")
```

### Forgetting to start audio playback

Audio File In CHOP won't produce data unless `play` is True and a valid `file` is set:
```
update_td_node_parameters(nodePath="/project1/audio_in",
    properties={"file": "/path/to/music.wav", "play": true})
```

### GLSL shader compilation errors

If a GLSL TOP shows errors after setting shader code:
1. Check the shader code in the Text DAT for syntax errors
2. Ensure the GLSL version is compatible (TD uses GLSL 3.30+)
3. Input sampler name must be `sTD2DInputs[0]` (not custom names)
4. Output must use `layout(location = 0) out vec4 fragColor`
5. UV coordinates come from `vUV.st` (not `gl_FragCoord`)
