# TouchDesigner Troubleshooting (twozero MCP)

> See `references/pitfalls.md` for the comprehensive lessons-learned list.

## 1. Connection Issues

### Port 40404 not responding

Check these in order:

1. Is TouchDesigner running?
   ```bash
   pgrep TouchDesigner
   ```

1b. Quick hub health check (no JSON-RPC needed):
   A plain GET to the MCP URL returns instance info:
   ```
   curl -s http://localhost:40404/mcp
   ```
   Returns: `{"hub": true, "pid": ..., "instances": {"127.0.0.1_PID": {"project": "...", "tdVersion": "...", ...}}}`
   If this returns JSON but `instances` is empty, TD is running but twozero hasn't registered yet.

2. Is twozero installed in TD?
   Open TD Palette Browser > twozero should be listed. If not, install it.

3. Is MCP enabled in twozero settings?
   In TD, open twozero preferences and confirm MCP server is toggled ON.

4. Test the port directly:
   ```bash
   nc -z 127.0.0.1 40404
   ```

5. Test the MCP endpoint:
   ```bash
   curl -s http://localhost:40404/mcp
   ```
   Should return JSON with hub info. If it does, the server is running.

### Hub responds but no TD instances

The twozero MCP hub is running but TD hasn't registered. Causes:
- TD project not loaded yet (still on splash screen)
- twozero COMP not initialized in the current project
- twozero version mismatch

Fix: Open/reload a TD project that contains the twozero COMP. Use td_list_instances
to check which TD instances are registered.

### Multi-instance setup

twozero auto-assigns ports for multiple TD instances:
- First instance: 40404
- Second instance: 40405
- Third instance: 40406
- etc.

Use `td_list_instances` to discover all running instances and their ports.

## 2. MCP Tool Errors

### td_execute_python returns error

The error message from td_execute_python often contains the Python traceback.
If it's unclear, use `td_read_textport` to see the full TD console output —
Python exceptions are always printed there.

Common causes:
- Syntax error in the script
- Referencing a node that doesn't exist (op() returns None, then you call .par on None)
- Using wrong parameter names (see pitfalls.md)

### td_set_operator_pars fails

Parameter name mismatch is the #1 cause. The tool validates param names and
returns clear errors, but you must use exact names.

Fix: ALWAYS call `td_get_par_info` first to discover the real parameter names:
```
td_get_par_info(op_type='glslTOP')
td_get_par_info(op_type='noiseTOP')
```

### td_create_operator type name errors

Operator type names use camelCase with family suffix:
- CORRECT: noiseTOP, glslTOP, levelTOP, compositeTOP, audiospectrumCHOP
- WRONG:   NoiseTOP, noise_top, NOISE TOP, Noise

### td_get_operator_info for deep inspection

If unsure about any aspect of an operator (params, inputs, outputs, state):
```
td_get_operator_info(path='/project1/noise1', detail='full')
```

## 3. Parameter Discovery

CRITICAL: ALWAYS use td_get_par_info to discover parameter names.

The agent's LLM training data contains WRONG parameter names for TouchDesigner.
Do not trust them. Known wrong names include dat vs pixeldat, colora vs alpha,
sizex vs size, and many more. See pitfalls.md for the full list.

Workflow:
1. td_get_par_info(op_type='glslTOP') — get all params for a type
2. td_get_operator_info(path='/project1/mynode', detail='full') — get params for a specific instance
3. Use ONLY the names returned by these tools

## 4. Performance

### Diagnosing slow performance

Use `td_get_perf` to see which operators are slow. Look at cook times —
anything over 1ms per frame is worth investigating.

Common causes:
- Resolution too high (especially on Non-Commercial)
- Complex GLSL shaders
- Too many TOP-to-CHOP or CHOP-to-TOP transfers (GPU-CPU memory copies)
- Feedback loops without decay (values accumulate, memory grows)

### Non-Commercial license restrictions

- Resolution cap: 1280x1280. Setting resolutionw=1920 silently clamps to 1280.
- H.264/H.265/AV1 encoding requires Commercial license. Use ProRes or Hap instead.
- No commercial use of output.

Always check effective resolution after creation:
```python
n.cook(force=True)
actual = str(n.width) + 'x' + str(n.height)
```

## 5. Hermes Configuration

### Config location

`$HERMES_HOME/config.yaml` (defaults to `~/.hermes/config.yaml` when `HERMES_HOME` is unset)

### MCP entry format

The twozero TD entry should look like:
```yaml
mcpServers:
  twozero_td:
    url: http://localhost:40404/mcp
```

### After config changes

Restart the Hermes session for changes to take effect. The MCP connection is
established at session startup.

### Verifying MCP tools are available

After restarting, the session log should show twozero MCP tools registered.
If tools show as registered but aren't callable, check:
- The twozero MCP hub is still running (curl test above)
- TD is still running with a project loaded
- No firewall blocking localhost:40404

## 6. Node Creation Issues

### "Node type not found" error

Wrong type string. Use camelCase with family suffix:
- Wrong: NoiseTop, noise_top, NOISE TOP
- Right: noiseTOP

### Node created but not visible

Check parentPath — use absolute paths like /project1. The default project
root is /project1. System nodes live at /, /ui, /sys, /local, /perform.
Don't create user nodes outside /project1.

### Cannot create node inside a non-COMP

Only COMP operators (Container, Base, Geometry, etc.) can contain children.
You cannot create nodes inside a TOP, CHOP, SOP, DAT, or MAT.

## 7. Wiring Issues

### Cross-family wiring

TOPs connect to TOPs, CHOPs to CHOPs, SOPs to SOPs, DATs to DATs.
Use converter operators to bridge: choptoTOP, topToCHOP, soptoDAT, etc.

Note: choptoTOP has NO input connectors. Use par.chop reference instead:
```python
spec_tex.par.chop = resample_node  # correct
# NOT: resample.outputConnectors[0].connect(spec_tex.inputConnectors[0])
```

### Feedback loops

Never create A -> B -> A directly. Use a Feedback TOP:
```python
fb = root.create(feedbackTOP, 'fb')
fb.par.top = comp.path          # reference only, no wire to fb input
fb.outputConnectors[0].connect(next_node)
```
"Cook dependency loop detected" warning on the chain is expected and correct.

## 8. GLSL Issues

### Shader compilation errors are silent

GLSL TOP shows a yellow warning in the UI but node.errors() may return empty.
Check node.warnings() too. Create an Info DAT pointed at the GLSL TOP for
full compiler output.

### TD GLSL specifics

- Uses GLSL 4.60 (Vulkan backend). GLSL 3.30 and earlier removed.
- UV coordinates: vUV.st (not gl_FragCoord)
- Input textures: sTD2DInputs[0]
- Output: layout(location = 0) out vec4 fragColor
- macOS CRITICAL: Always wrap output with TDOutputSwizzle(color)
- No built-in time uniform. Pass time via GLSL TOP Values page or Constant TOP.

## 9. Recording Issues

### H.264/H.265/AV1 requires Commercial license

Use Apple ProRes on macOS (hardware accelerated, not license-restricted):
```python
rec.par.videocodec = 'prores'  # Preferred on macOS — lossless, Non-Commercial OK
# rec.par.videocodec = 'mjpa'  # Fallback — lossy, works everywhere
```

### MovieFileOut has no .record() method

Use the toggle parameter:
```python
rec.par.record = True   # start
rec.par.record = False  # stop
```

### All exported frames identical

TOP.save() captures same frame when called rapidly. Use MovieFileOut for
real-time recording. Set project.realTime = False for frame-accurate output.
