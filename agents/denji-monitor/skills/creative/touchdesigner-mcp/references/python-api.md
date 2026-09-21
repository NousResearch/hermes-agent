# TouchDesigner Python API Reference

## The td Module

TouchDesigner's Python environment auto-imports the `td` module. All TD-specific classes, functions, and constants live here. Scripts inside TD (Script DATs, CHOP/DAT Execute callbacks, Extensions) have full access.

When using the MCP `execute_python_script` tool, these globals are pre-loaded:
- `op` — shortcut for `td.op()`, finds operators by path
- `ops` — shortcut for `td.ops()`, finds multiple operators by pattern
- `me` — the operator running the script (via MCP this is the twozero internal executor)
- `parent` — shortcut for `me.parent()`
- `project` — the root project component
- `td` — the full td module

## Finding Operators: op() and ops()

### op(path) — Find a single operator

```python
# Absolute path (always works from MCP)
node = op('/project1/noise1')

# Relative path (relative to current operator — only in Script DATs)
node = op('noise1')      # sibling
node = op('../noise1')   # parent's sibling

# Returns None if not found (does NOT raise)
node = op('/project1/nonexistent')  # None
```

### ops(pattern) — Find multiple operators

```python
# Glob patterns
nodes = ops('/project1/noise*')       # all nodes starting with "noise"
nodes = ops('/project1/*')            # all direct children
nodes = ops('/project1/container1/*') # all children of container1

# Returns a tuple of operators (may be empty)
for n in ops('/project1/*'):
    print(n.name, n.OPType)
```

### Navigation from a node

```python
node = op('/project1/noise1')

node.name        # 'noise1'
node.path        # '/project1/noise1'
node.OPType      # 'noiseTop'
node.type         # <class 'noiseTop'>
node.family       # 'TOP'

# Parent / children
node.parent()              # the parent COMP
node.parent().children     # all siblings + self
node.parent().findChildren(name='noise*')  # filtered

# Type checking
node.isTOP   # True
node.isCHOP  # False
node.isSOP   # False
node.isDAT   # False
node.isMAT   # False
node.isCOMP  # False
```

## Parameters

Every operator has parameters accessed via the `.par` attribute.

### Reading parameters

```python
node = op('/project1/noise1')

# Direct access
node.par.seed.val        # current evaluated value (may be an expression result)
node.par.seed.eval()     # same as .val
node.par.seed.default    # default value
node.par.monochrome.val  # boolean parameters: True/False

# List all parameters
for p in node.pars():
    print(f"{p.name}: {p.val} (default: {p.default})")

# Filter by page (parameter group)
for p in node.pars('Noise'):  # page name
    print(f"{p.name}: {p.val}")
```

### Setting parameters

```python
# Direct value setting
node.par.seed.val = 42
node.par.monochrome.val = True
node.par.resolutionw.val = 1920
node.par.resolutionh.val = 1080

# String parameters
op('/project1/text1').par.text.val = 'Hello World'

# File paths
op('/project1/moviefilein1').par.file.val = '/path/to/video.mp4'

# Reference another operator (for "dat", "chop", "top" type parameters)
op('/project1/glsl1').par.dat.val = '/project1/shader_code'
```

### Parameter expressions

```python
# Python expressions that evaluate dynamically
node.par.seed.expr = "me.time.frame"
node.par.tx.expr = "math.sin(me.time.seconds * 2)"

# Reference another parameter
node.par.brightness1.expr = "op('/project1/constant1').par.value0.val"

# Export (one-way binding from CHOP to parameter)
# This makes the parameter follow a CHOP channel value
op('/project1/noise1').par.seed.val  # can also be driven by exports
```

### Parameter types

| Type | Python Type | Example |
|------|------------|---------|
| Float | `float` | `node.par.brightness1.val = 0.5` |
| Int | `int` | `node.par.seed.val = 42` |
| Toggle | `bool` | `node.par.monochrome.val = True` |
| String | `str` | `node.par.text.val = 'hello'` |
| Menu | `int` (index) or `str` (label) | `node.par.type.val = 'sine'` |
| File | `str` (path) | `node.par.file.val = '/path/to/file'` |
| OP reference | `str` (path) | `node.par.dat.val = '/project1/text1'` |
| Color | separate r/g/b/a floats | `node.par.colorr.val = 1.0` |
| XY/XYZ | separate x/y/z floats | `node.par.tx.val = 0.5` |

## Creating and Deleting Operators

```python
# Create via parent component
parent = op('/project1')
new_node = parent.create(noiseTop)         # using class reference
new_node = parent.create(noiseTop, 'my_noise')  # with custom name

# The MCP create_td_node tool handles this automatically:
# create_td_node(parentPath="/project1", nodeType="noiseTop", nodeName="my_noise")

# Delete
node = op('/project1/my_noise')
node.destroy()

# Copy
original = op('/project1/noise1')
copy = parent.copy(original, name='noise1_copy')
```

## Connections (Wiring Operators)

### Output to Input connections

```python
# Connect noise1's output to level1's input
op('/project1/noise1').outputConnectors[0].connect(op('/project1/level1'))

# Connect to specific input index (for multi-input operators like Composite)
op('/project1/noise1').outputConnectors[0].connect(op('/project1/composite1').inputConnectors[0])
op('/project1/text1').outputConnectors[0].connect(op('/project1/composite1').inputConnectors[1])

# Disconnect all outputs
op('/project1/noise1').outputConnectors[0].disconnect()

# Query connections
node = op('/project1/level1')
inputs = node.inputs          # list of connected input operators
outputs = node.outputs        # list of connected output operators
```

### Connection patterns for common setups

```python
# Linear chain: A -> B -> C -> D
ops_list = [op(f'/project1/{name}') for name in ['noise1', 'level1', 'blur1', 'null1']]
for i in range(len(ops_list) - 1):
    ops_list[i].outputConnectors[0].connect(ops_list[i+1])

# Fan-out: A -> B, A -> C, A -> D
source = op('/project1/noise1')
for target_name in ['level1', 'composite1', 'transform1']:
    source.outputConnectors[0].connect(op(f'/project1/{target_name}'))

# Merge: A + B + C -> Composite
comp = op('/project1/composite1')
for i, source_name in enumerate(['noise1', 'text1', 'ramp1']):
    op(f'/project1/{source_name}').outputConnectors[0].connect(comp.inputConnectors[i])
```

## DAT Content Manipulation

### Text DATs

```python
dat = op('/project1/text1')

# Read
content = dat.text          # full text as string

# Write
dat.text = "new content"
dat.text = '''multi
line
content'''

# Append
dat.text += "\nnew line"
```

### Table DATs

```python
dat = op('/project1/table1')

# Read cell
val = dat[0, 0]         # row 0, col 0
val = dat[0, 'name']    # row 0, column named 'name'
val = dat['key', 1]     # row named 'key', col 1

# Write cell
dat[0, 0] = 'value'

# Read row/col
row = dat.row(0)         # list of Cell objects
col = dat.col('name')    # list of Cell objects

# Dimensions
rows = dat.numRows
cols = dat.numCols

# Append row
dat.appendRow(['col1_val', 'col2_val', 'col3_val'])

# Clear
dat.clear()

# Set entire table
dat.clear()
dat.appendRow(['name', 'value', 'type'])
dat.appendRow(['frequency', '440', 'float'])
dat.appendRow(['amplitude', '0.8', 'float'])
```

## Time and Animation

```python
# Global time
td.absTime.frame       # absolute frame number (never resets)
td.absTime.seconds     # absolute seconds

# Timeline time (affected by play/pause/loop)
me.time.frame          # current frame on timeline
me.time.seconds        # current seconds on timeline
me.time.rate           # FPS setting

# Timeline control (via execute_python_script)
project.play = True
project.play = False
project.frameRange = (1, 300)   # set timeline range

# Cook frame (when operator was last computed)
node.cookFrame
node.cookTime
```

## Extensions (Custom Python Classes on Components)

Extensions add custom Python methods and attributes to COMPs.

```python
# Create extension on a Base COMP
base = op('/project1/myBase')

# The extension class is defined in a Text DAT inside the COMP
# Typically named 'ExtClass' with the extension code:

extension_code = '''
class MyExtension:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.counter = 0

    def Reset(self):
        self.counter = 0

    def Increment(self):
        self.counter += 1
        return self.counter

    @property
    def Count(self):
        return self.counter
'''

# Write extension code to DAT inside the COMP
op('/project1/myBase/extClass').text = extension_code

# Configure the extension on the COMP
base.par.extension1 = 'extClass'  # name of the DAT
base.par.promoteextension1 = True  # promote methods to parent

# Call extension methods
base.Increment()       # calls MyExtension.Increment()
count = base.Count     # accesses MyExtension.Count property
base.Reset()
```

## Useful Built-in Modules

### tdu — TouchDesigner Utilities

```python
import tdu

# Dependency tracking (reactive values)
dep = tdu.Dependency(initial_value)
dep.val = new_value   # triggers dependents to recook

# File path utilities
tdu.expandPath('$HOME/Desktop/output.mov')

# Math
tdu.clamp(value, min, max)
tdu.remap(value, from_min, from_max, to_min, to_max)
```

### TDFunctions

```python
from TDFunctions import *

# Commonly used utilities
clamp(value, low, high)
remap(value, inLow, inHigh, outLow, outHigh)
interp(value1, value2, t)  # linear interpolation
```

### TDStoreTools — Persistent Storage

```python
from TDStoreTools import StorageManager

# Store data that survives project reload
me.store('myKey', 'myValue')
val = me.fetch('myKey', default='fallback')

# Storage dict
me.storage['key'] = value
```

## Common Patterns via execute_python_script

### Build a complete chain

```python
# Create a complete audio-reactive noise chain
parent = op('/project1')

# Create operators
audio_in = parent.create(audiofileinChop, 'audio_in')
spectrum = parent.create(audiospectrumChop, 'spectrum')
chop_to_top = parent.create(choptopTop, 'chop_to_top')
noise = parent.create(noiseTop, 'noise1')
level = parent.create(levelTop, 'level1')
null_out = parent.create(nullTop, 'out')

# Wire the chain
audio_in.outputConnectors[0].connect(spectrum)
spectrum.outputConnectors[0].connect(chop_to_top)
noise.outputConnectors[0].connect(level)
level.outputConnectors[0].connect(null_out)

# Set parameters
audio_in.par.file = '/path/to/music.wav'
audio_in.par.play = True
spectrum.par.size = 512
noise.par.type = 1  # Sparse
noise.par.monochrome = False
noise.par.resolutionw = 1920
noise.par.resolutionh = 1080
level.par.opacity = 0.8
level.par.gamma1 = 0.7
```

### Query network state

```python
# Get all TOPs in the project
tops = [c for c in op('/project1').findChildren(type=TOP)]
for t in tops:
    print(f"{t.path}: {t.OPType} {'ERROR' if t.errors() else 'OK'}")

# Find all operators with errors
def find_errors(parent_path='/project1'):
    parent = op(parent_path)
    errors = []
    for child in parent.findChildren(depth=-1):
        if child.errors():
            errors.append((child.path, child.errors()))
    return errors

result = find_errors()
```

### Batch parameter changes

```python
# Set parameters on multiple nodes at once
settings = {
    '/project1/noise1': {'seed': 42, 'monochrome': False, 'resolutionw': 1920},
    '/project1/level1': {'brightness1': 1.2, 'gamma1': 0.8},
    '/project1/blur1': {'sizex': 5, 'sizey': 5},
}

for path, params in settings.items():
    node = op(path)
    if node:
        for key, val in params.items():
            setattr(node.par, key, val)
```

## Python Version and Packages

TouchDesigner bundles Python 3.11+ with these pre-installed:
- **numpy** — array operations, fast math
- **scipy** — signal processing, FFT
- **OpenCV** (cv2) — computer vision
- **PIL/Pillow** — image processing
- **requests** — HTTP client
- **json**, **re**, **os**, **sys** — standard library

**IMPORTANT:** Parameter names in examples below are illustrative. Always run discovery (SKILL.md Step 0) to get actual names for your TD version. Do NOT copy param names from these examples verbatim.

Custom packages can be installed to TD's Python site-packages directory. See TD documentation for the exact path per platform.

## SOP Vertex/Point Access (TD 2025.32)

In TD 2025.32, `td.Vertex` does NOT have `.x`, `.y`, `.z` attributes. Use index access:

```python
# WRONG — crashes in TD 2025.32:
vertex.x, vertex.y, vertex.z

# CORRECT — index/attribute access:
pt = sop.points()[i]
pos = pt.P          # Position object
x, y, z = pos[0], pos[1], pos[2]

# Always introspect first:
dir(sop.points()[0])   # see what attributes actually exist
dir(sop.points()[0].P) # see Position object interface
```
