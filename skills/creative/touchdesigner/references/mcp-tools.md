# TouchDesigner MCP Tools Reference

Complete parameter schemas and usage examples for all 13 MCP tools from the 8beeeaaat/touchdesigner-mcp server.

## Hermes Configuration

Add a `touchdesigner` entry under the `mcp_servers` section of your Hermes config. Example YAML block:

```yaml
# Under mcp_servers: in config.yaml
mcp_servers:
  touchdesigner:
    command: npx
    args: ["-y", "touchdesigner-mcp-server@latest"]
    env:
      TD_API_URL: "http://127.0.0.1:9981"
    timeout: 120
    connect_timeout: 60
```

For a locally built server, point `command` to `node` and `args` to the built server index.js path. Set `TD_API_URL` to the TouchDesigner WebServer DAT address (default port 9981).

For the documentation/knowledge server (no running TD needed), add a `td_docs` entry using `touchdesigner-mcp-server` as the npx package.

Tools are registered as `mcp_touchdesigner_<tool_name>` in Hermes.

**If MCP tools are not available as direct function calls** (common when the MCP server connects but Hermes doesn't expose them as callable tools), use the custom API handler directly via `curl` in `execute_code` or `terminal`:

```python
import json, shlex
from hermes_tools import terminal

def td_exec(script):
    """Execute Python in TouchDesigner via the REST API."""
    escaped = json.dumps({"script": script})
    cmd = f"curl -s --max-time 15 -X POST -H 'Content-Type: application/json' -d {shlex.quote(escaped)} 'http://127.0.0.1:9981/api/td/server/exec'"
    r = terminal(cmd, timeout=20)
    return json.loads(r['output'])

# Example: list all nodes
result = td_exec('result = [c.name for c in op("/project1").children]')
print(result)  # {"result": ["node1", "node2", ...], "stdout": "", "stderr": ""}
```

This `td_exec` helper works with both the official .tox handler and the custom API handler from `scripts/custom_api_handler.py`.

Tools are registered as `mcp_touchdesigner_<tool_name>` in Hermes.

## Common Formatting Parameters

Most tools accept these optional formatting parameters:

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `detailLevel` | string | `"minimal"`, `"summary"`, `"detailed"` | Response verbosity |
| `responseFormat` | string | `"json"`, `"yaml"`, `"markdown"` | Output format |
| `limit` | integer | 1-500 | Max items (on list-type tools only) |

These are client-side formatting — they control how the MCP server formats the response text, not what data TD returns.

---

## Tool 1: describe_td_tools

**Purpose:** Meta-tool — lists all available TouchDesigner MCP tools with descriptions and parameters.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filter` | string | No | Keyword to filter tools by name, description, or parameter |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Example:** Find tools related to node creation
```
describe_td_tools(filter="create")
```

**Note:** This tool runs entirely in the MCP server — it does NOT contact TouchDesigner. Use it to discover what's available.

---

## Tool 2: get_td_info

**Purpose:** Get TouchDesigner server information (version, OS, build).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Example:** Check TD is running and get version
```
get_td_info()
```

**Returns:** TD version, build number, OS name/version, MCP API version.

**Use this first** to verify the connection is working before building networks.

---

## Tool 3: execute_python_script

**Purpose:** Execute arbitrary Python code inside TouchDesigner's Python environment.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `script` | string | **Yes** | Python code to execute |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Available globals in the script:**
- `op` — find operators by path
- `ops` — find multiple operators by pattern
- `me` — the WebServer DAT running the script
- `parent` — me.parent()
- `project` — root project component
- `td` — the full td module
- `result` — set this to explicitly return a value

**Execution behavior:**
- Single-line scripts: tries `eval()` first (returns value), falls back to `exec()`
- Multi-line scripts: uses `exec()` always
- stdout/stderr are captured and returned separately
- If `result` is not set, tries to evaluate the last expression as the return value

**Examples:**

```python
# Simple query
execute_python_script(script="op('/project1/noise1').par.seed.val")
# Returns: {"result": 42, "stdout": "", "stderr": ""}

# Multi-line script
execute_python_script(script="""
nodes = op('/project1').findChildren(type=TOP)
result = [{'name': n.name, 'type': n.OPType} for n in nodes]
""")

# Connect two operators
execute_python_script(script="op('/project1/noise1').outputConnectors[0].connect(op('/project1/level1'))")

# Create and configure in one script
execute_python_script(script="""
parent = op('/project1')
n = parent.create(noiseTop, 'my_noise')
n.par.seed.val = 42
n.par.monochrome.val = True
n.par.resolutionw.val = 1920
n.par.resolutionh.val = 1080
result = {'path': n.path, 'type': n.OPType}
""")

# Batch wire a chain
execute_python_script(script="""
chain = ['noise1', 'level1', 'blur1', 'composite1', 'null_out']
for i in range(len(chain) - 1):
    src = op(f'/project1/{chain[i]}')
    dst = op(f'/project1/{chain[i+1]}')
    if src and dst:
        src.outputConnectors[0].connect(dst)
result = 'Wired chain: ' + ' -> '.join(chain)
""")
```

**When to use:** Wiring connections, complex logic, batch operations, querying state that other tools don't cover. This is the most powerful and flexible tool.

---

## Tool 4: create_td_node

**Purpose:** Create a new operator in TouchDesigner.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `parentPath` | string | **Yes** | Path to parent (e.g., `/project1`) |
| `nodeType` | string | **Yes** | Operator type (e.g., `noiseTop`, `mathChop`) |
| `nodeName` | string | No | Custom name (auto-generated if omitted) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Examples:**

```
create_td_node(parentPath="/project1", nodeType="noiseTop", nodeName="bg_noise")
create_td_node(parentPath="/project1", nodeType="compositeTop")  # auto-named
create_td_node(parentPath="/project1/audio_chain", nodeType="audiospectrumChop", nodeName="spectrum")
```

**Returns:** Node summary with id, name, path, opType, and all default parameter values.

**Node type naming convention:** camelCase family suffix — `noiseTop`, `mathChop`, `gridSop`, `tableDat`, `phongMat`, `geometryComp`. See `references/operators.md` for the full list.

---

## Tool 5: delete_td_node

**Purpose:** Delete an existing operator.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `nodePath` | string | **Yes** | Absolute path to node (e.g., `/project1/noise1`) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Example:**

```
delete_td_node(nodePath="/project1/noise1")
```

**Returns:** Confirmation with the deleted node's summary (captured before deletion).

---

## Tool 6: get_td_nodes

**Purpose:** List operators under a path with optional filtering.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `parentPath` | string | **Yes** | Parent path (e.g., `/project1`) |
| `pattern` | string | No | Glob pattern for name filtering (default: `*`) |
| `includeProperties` | boolean | No | Include full parameter values (default: false) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |
| `limit` | integer | No | Max items (1-500) |

**Examples:**

```
# List all direct children of /project1
get_td_nodes(parentPath="/project1")

# Find all noise operators
get_td_nodes(parentPath="/project1", pattern="noise*")

# Get full parameter details
get_td_nodes(parentPath="/project1", pattern="*", includeProperties=true, limit=20)
```

**Returns:** List of node summaries. With `includeProperties=false` (default): id, name, path, opType only. With `includeProperties=true`: full parameter values included.

---

## Tool 7: get_td_node_parameters

**Purpose:** Get detailed parameters of a specific node.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `nodePath` | string | **Yes** | Node path (e.g., `/project1/noise1`) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |
| `limit` | integer | No | Max parameters (1-500) |

**Example:**

```
get_td_node_parameters(nodePath="/project1/noise1")
```

**Returns:** All parameter name-value pairs for the node. Use this to discover available parameters before calling update_td_node_parameters.

---

## Tool 8: get_td_node_errors

**Purpose:** Check for errors on a node and all its descendants (recursive).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `nodePath` | string | **Yes** | Absolute path to inspect (e.g., `/project1`) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |
| `limit` | integer | No | Max error items (1-500) |

**Examples:**

```
# Check entire project for errors
get_td_node_errors(nodePath="/project1")

# Check a specific chain
get_td_node_errors(nodePath="/project1/audio_chain")
```

**Returns:** Error count, hasErrors boolean, and list of errors each with nodePath, nodeName, opType, and error message.

**Always call this after building a network** to catch wiring mistakes, missing references, and configuration errors.

---

## Tool 9: update_td_node_parameters

**Purpose:** Update parameters on an existing node.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `nodePath` | string | **Yes** | Path to node (e.g., `/project1/noise1`) |
| `properties` | object | **Yes** | Key-value pairs to update (e.g., `{"seed": 42, "monochrome": true}`) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Examples:**

```
# Set noise parameters
update_td_node_parameters(
    nodePath="/project1/noise1",
    properties={"seed": 42, "monochrome": false, "period": 4.0, "harmonics": 3,
                "resolutionw": 1920, "resolutionh": 1080}
)

# Set a file path
update_td_node_parameters(
    nodePath="/project1/moviefilein1",
    properties={"file": "/Users/me/Videos/clip.mp4", "play": true}
)

# Set compositing mode
update_td_node_parameters(
    nodePath="/project1/composite1",
    properties={"operand": 0}  # 0=Over, 1=Under, 3=Add, 18=Multiply, 27=Screen
)
```

**Returns:** List of successfully updated properties and any that failed (with reasons). Raises error if zero properties were updated.

**Parameter value types:** Floats, ints, booleans, and strings are all accepted. For menu parameters, use either the string label or the integer index.

---

## Tool 10: exec_node_method

**Purpose:** Call a Python method directly on a specific node.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `nodePath` | string | **Yes** | Path to node |
| `method` | string | **Yes** | Method name to call |
| `args` | array | No | Positional arguments (strings, numbers, booleans) |
| `kwargs` | object | No | Keyword arguments |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Examples:**

```
# Get all children of a component
exec_node_method(nodePath="/project1", method="findChildren")

# Find specific children
exec_node_method(nodePath="/project1", method="findChildren",
                 kwargs={"name": "noise*", "depth": 1})

# Get node errors
exec_node_method(nodePath="/project1/noise1", method="errors")

# Get node warnings
exec_node_method(nodePath="/project1/noise1", method="warnings")

# Save a component as .tox
exec_node_method(nodePath="/project1/myContainer", method="save",
                 args=["/path/to/component.tox"])
```

**Returns:** Processed return value of the method call. TD operators are serialized to their path strings, iterables to lists, etc.

---

## Tool 11: get_td_classes

**Purpose:** List available TouchDesigner Python classes and modules.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |
| `limit` | integer | No | Max items (default: 50) |

**Example:**

```
get_td_classes(limit=100)
```

**Returns:** List of class/module names and their docstrings from the td module. Useful for discovering what's available in TD's Python environment.

---

## Tool 12: get_td_class_details

**Purpose:** Get methods and properties of a specific TD Python class.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `className` | string | **Yes** | Class name (e.g., `noiseTop`, `OP`, `COMP`) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |
| `limit` | integer | No | Max methods/properties (default: 30) |

**Examples:**

```
# Inspect the noiseTop class
get_td_class_details(className="noiseTop")

# Inspect the base OP class (all operators inherit from this)
get_td_class_details(className="OP", limit=50)

# Inspect COMP (component) class
get_td_class_details(className="COMP")
```

**Returns:** Class name, type, description, methods (name + description + type), and properties (name + description + type).

---

## Tool 13: get_td_module_help

**Purpose:** Retrieve Python help() text for any TD module, class, or function.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `moduleName` | string | **Yes** | Module/class name (e.g., `noiseCHOP`, `tdu`, `td.OP`) |
| `detailLevel` | string | No | Response verbosity |
| `responseFormat` | string | No | Output format |

**Examples:**

```
# Get help for the noise CHOP class
get_td_module_help(moduleName="noiseCHOP")

# Get help for the tdu utilities module
get_td_module_help(moduleName="tdu")

# Dotted name resolution works
get_td_module_help(moduleName="td.OP")
```

**Returns:** Full Python help() text output, cleaned of backspace characters.

---

## Workflow: Building a Complete Network

Typical sequence of tool calls to build a project:

1. `get_td_info` — verify connection
2. `get_td_nodes(parentPath="/project1")` — see what already exists
3. `create_td_node` (multiple) — create all operators
4. `update_td_node_parameters` (multiple) — configure each operator
5. `execute_python_script` — wire all connections in one batch script
6. `get_td_node_errors(nodePath="/project1")` — check for problems
7. `get_td_node_parameters` — verify specific nodes if needed
8. Iterate: adjust parameters, add operators, fix errors

## TD Documentation MCP Server Tools

The bottobot/touchdesigner-mcp-server provides 21 reference/knowledge tools (no running TD needed):

| Tool | Purpose |
|------|---------|
| `get_operator` | Get full documentation for a specific operator |
| `search_operators` | Search operators by keyword |
| `list_operators` | List all operators (filterable by family) |
| `compare_operators` | Compare two operators side by side |
| `get_operator_examples` | Get usage examples for an operator |
| `suggest_workflow` | Get workflow suggestions for a task |
| `get_tutorial` | Get a full TD tutorial |
| `list_tutorials` | List available tutorials |
| `search_tutorials` | Search tutorial content |
| `get_python_api` | Get Python API class documentation |
| `search_python_api` | Search Python API |
| `list_python_classes` | List all documented Python classes |
| `get_version_info` | Get TD version release notes |
| `list_versions` | List all documented TD versions |
| `get_experimental_techniques` | Get advanced technique guides (GLSL, ML, generative, etc.) |
| `search_experimental` | Search experimental techniques |
| `get_glsl_pattern` | Get GLSL code patterns (SDF, color, math utilities) |
| `get_operator_connections` | Get common operator wiring patterns |
| `get_network_template` | Get complete network templates with Python generation scripts |
| `get_experimental_build` | Get experimental build info |
| `list_experimental_builds` | List experimental builds |

This server contains 630 operator docs, 14 tutorials, 69 Python API classes, and 7 experimental technique categories with working code.
