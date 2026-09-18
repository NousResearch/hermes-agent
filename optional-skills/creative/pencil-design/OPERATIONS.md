# Pencil Design Operations Reference

Quick reference for all Pencil MCP operations available in Hermes Agent.

## Document Operations

### `open_document`

Open a .pen file for editing.

```
Arguments:
  filePath (string, required) — absolute path to .pen file

Returns:
  editor_state — current document state, selection, viewport
```

Example workflow:
```
User: Open my design file
Agent: open_document(filePath="/path/to/designs/app.pen")
```

### `get_editor_state`

Query current editor state without opening a file. Returns viewport, selection, zoom level, active tool.

```
Arguments:
  (none)

Returns:
  {
    viewport: {x, y, zoom},
    selection: [node_id, ...],
    activeDocument: "path/to/file.pen",
    ...
  }
```

## Design Editing (Batch Operations)

### `batch_design`

Insert, update, replace, move, or delete multiple design nodes in a single call.

```
Arguments:
  filePath (string) — .pen file path
  operations (array) — list of I/U/R/M/D operations

Supports:
  I (Insert) — add new nodes
  U (Update) — modify properties
  R (Replace) — swap out a node entirely
  M (Move) — change parent/position
  D (Delete) — remove nodes
  G (Generate) — AI-generate or stock images
```

Example: Insert a button frame with a text label

```javascript
container=I("parentId",{type:"frame",layout:"horizontal",gap:8,padding:12})
text=I(container,{type:"text",content:"Click Me",fontSize:14})
```

Example: Update button colors

```
U("buttonId",{fill:"#FF0000"})
```

Example: Move node to different parent

```
M("nodeId", "newParentId")
```

### `batch_get`

Fetch multiple nodes by ID, path, or pattern matching.

```
Arguments:
  filePath (string) — .pen file path
  query (string or array) — node IDs, paths, or glob patterns

Returns:
  { nodeId: {type, name, properties, children, ...}, ... }
```

Example: Get all buttons

```
batch_get(filePath="app.pen", query="Button*")
```

## Export & Visualization

### `get_screenshot`

Capture the current viewport as an image (PNG, JPEG, etc.).

```
Arguments:
  filePath (string) — .pen file path
  format (string, default "png") — output format
  quality (number, 0-100, optional) — compression quality
  width, height (numbers, optional) — output dimensions

Returns:
  binary image data (base64 or file path)
```

Example:
```
get_screenshot(filePath="app.pen", format="png")
→ Returns PNG bytes or file path like "/tmp/screenshot_abc123.png"
```

### `snapshot_layout`

Analyze layout metrics: spacing, alignment, sizing, hierarchy.

```
Arguments:
  filePath (string) — .pen file path
  nodeId (string, optional) — specific node to analyze

Returns:
  {
    nodes: [{ id, name, bounds, spacing, alignment, ... }],
    summary: { totalElements, grouping, spacingPattern, ... }
  }
```

Useful for verifying design consistency without full screenshots.

### `export_nodes`

Export selected nodes as image files (PNG, SVG, etc.).

```
Arguments:
  filePath (string) — .pen file path
  nodeIds (array) — node IDs to export
  format (string, default "png") — "png", "svg", "jpeg", "pdf"
  scale (number, default 1) — export scale (2 = 2x)

Returns:
  { nodeId: "file/path/exported.png", ... }
```

Example: Export all screens as PNG

```
export_nodes(filePath="app.pen", 
             nodeIds=["screen-1", "screen-2", "screen-3"],
             format="png")
→ Returns paths for each exported image
```

## Design Variables & Theming

### `get_variables`

Retrieve all defined design variables (colors, typography, spacing, etc.) and current theme values.

```
Arguments:
  filePath (string, optional) — scope to specific file

Returns:
  {
    variables: {
      color: {primary: "#FF0000", secondary: "#00FF00", ...},
      typography: {heading: {...}, body: {...}, ...},
      spacing: {xs: 4, sm: 8, md: 16, ...}
    },
    currentTheme: "light",
    themes: ["light", "dark", ...]
  }
```

### `set_variables`

Update design variables and apply themes.

```
Arguments:
  filePath (string) — .pen file path
  variables (object) — new variable values
  theme (string, optional) — activate a theme

Returns:
  confirmation + updated variable state
```

Example: Update brand colors

```
set_variables(filePath="app.pen",
              variables={
                color: {
                  primary: "#1E88E5",
                  secondary: "#43A047"
                }
              })
```

Example: Switch to dark theme

```
set_variables(filePath="app.pen", theme="dark")
```

## Design Guidelines & Analysis

### `get_guidelines`

Fetch design guidelines, style archetypes, and design system rules.

```
Arguments:
  filePath (string, optional) — scope to specific file

Returns:
  {
    guidelines: [
      {name: "Typography", rules: ["Use system fonts", ...]},
      {name: "Color", rules: ["WCAG AA contrast minimum", ...]},
      ...
    ],
    archetypes: {Button: {...}, Card: {...}, ...}
  }
```

### `find_empty_space_on_canvas`

Find available space for new elements on the canvas.

```
Arguments:
  filePath (string) — .pen file path
  minWidth, minHeight (numbers, optional) — required dimensions

Returns:
  {x, y, width, height} — coordinates of largest empty area
```

Useful for auto-positioning new elements without overlap.

## Property Search & Bulk Updates

### `search_all_unique_properties`

Find all unique values for a specific property across the design.

```
Arguments:
  filePath (string) — .pen file path
  property (string) — property name (e.g., "fill", "fontSize")

Returns:
  ["#FF0000", "#00FF00", "#0000FF", ...]  — unique values found
```

Example: Find all font sizes in use

```
search_all_unique_properties(filePath="app.pen", property="fontSize")
→ Returns [12, 14, 16, 18, 20, 24, ...]
```

### `replace_all_matching_properties`

Bulk-replace all properties matching a pattern.

```
Arguments:
  filePath (string) — .pen file path
  property (string) — property to match
  oldValue (any) — value to find
  newValue (any) — replacement value
  nodeFilter (optional) — restrict to matching nodes

Returns:
  { replaced: count, affectedNodes: [id, ...] }
```

Example: Update all buttons to new padding

```
replace_all_matching_properties(
  filePath="app.pen",
  property="padding",
  oldValue=8,
  newValue=12,
  nodeFilter="Button*")
→ Returns {replaced: 47, affectedNodes: [...]}
```

## Component Instances

When working with reusable components (components.reusable=true):

- **Create instance**: `{type: "ref", ref: "ComponentId"}`
- **Override properties**: Add properties directly to the instance
- **Customize children**: Use `descendants` map in the I/C operation
- **Update nested elements**: Use `U` with path like `instanceId/childId`

Example: Create button instance with custom label

```javascript
button=I("parentId",{type:"ref",ref:"Button.Primary",x:100,y:200})
U(button+"/label",{content:"Custom Label"})
```

## Patterns & Common Workflows

### Bulk Create Buttons

```javascript
buttons=[]
for i in range(3):
  btn=I("parentId",{type:"ref",ref:"Button",x:100,y:i*50})
  buttons.append(btn)
```

### Find and Update All Text

```javascript
text_nodes = batch_get("app.pen", "**/*.text")
for node_id in text_nodes:
  U(node_id, {fontSize: 16})
```

### Create Design Variant (Light/Dark)

```javascript
get_variables("app.pen")  // current state
set_variables("app.pen", {
  colors: {bg: "#FFFFFF", text: "#000000"}
}, theme="light")
export_nodes("app.pen", nodeIds=["all"], format="png")
```

### Verify Design Spacing

```javascript
layout_data = snapshot_layout("app.pen")
// Analyze layout_data.summary for consistency issues
```

## Error Handling

Common errors and solutions:

| Error | Cause | Fix |
|-------|-------|-----|
| `File not found` | Invalid file path | Use absolute path or check permissions |
| `Node not found` | ID doesn't exist | Use `batch_get` to list available nodes |
| `Connection timeout` | Server slow/unreachable | Increase timeout in config |
| `Invalid operation` | Schema mismatch | Check operation signature in docs |
| `Permission denied` | Can't modify file | Check file is writable, not locked |

## Performance Tips

- **Batch operations**: Use `batch_design` for multiple edits (not serial calls)
- **Screenshot caching**: Reuse screenshot paths until you make changes
- **Lazy loading**: Use `get_editor_state` instead of `get_screenshot` when you just need metadata
- **Scoped updates**: Use node filters in replace operations to avoid processing the whole document

## See Also

- `SKILL.md` — Full skill documentation and workflows
- `SETUP.md` — Configuration guide
- Pencil MCP server docs: https://github.com/pencil-mcp/docs (adjust URL)
