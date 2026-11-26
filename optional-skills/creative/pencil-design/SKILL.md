---
name: pencil-design
description: Design and manipulate .pen files using Pencil MCP. Create/edit web and mobile app designs, manage design systems, export assets, and generate design previews. Requires Pencil MCP server connection.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [pencil, design, MCP, UI/UX, design-systems, web-design, mobile-design]
    homepage: https://pencil.com
    related_skills: [claude-design, popular-web-designs, design-md]
    category: creative
    config:
      mcp.pencil:
        description: "Pencil MCP server configuration (command, url, or transport settings)"
        prompt: "Pencil MCP server details"
prerequisites:
  services: ["Pencil MCP server"]
  notes: "See Setup section for MCP connection configuration"
---

# Pencil Design via MCP

Programmatically edit Pencil design files (.pen format) using the Pencil MCP protocol. Create, modify, export, and inspect designs for web and mobile applications with full system prompt awareness.

## When To Use This Skill

Use this skill when:

- **Creating/editing app designs** — wireframes, prototypes, high-fidelity UI mockups
- **Designing design systems** — component libraries, token definitions, reusable patterns
- **Batch design operations** — programmatically update multiple elements, swap properties
- **Extracting design artifacts** — export nodes as images, generate screenshots, extract variables
- **Inspecting designs** — query design structure, find components, analyze layouts
- **Automating design tasks** — bulk updates, design variable management, theme switching

Do NOT use this skill for:
- One-off HTML/CSS prototypes (use `claude-design` instead)
- Design token specs (use `design-md` for DESIGN.md format)
- Brand design lookups (use `popular-web-designs`)

## Setup: Configure Pencil MCP Connection

### Option 1: Stdio (if Pencil MCP is a command-line server)

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  pencil:
    command: "npx"
    args: ["@pencil-mcp/server"]  # adjust package name as needed
    timeout: 120
    connect_timeout: 60
```

### Option 2: HTTP (if Pencil MCP is running as a service)

```yaml
mcp_servers:
  pencil:
    url: "http://localhost:5000/mcp"  # adjust URL/port as needed
    timeout: 120
    connect_timeout: 60
    headers: {}  # add auth headers if needed
```

### Option 3: SSH (if Pencil MCP is on a remote host)

```yaml
mcp_servers:
  pencil:
    command: "ssh"
    args: ["user@host", "pencil-mcp-server"]
    timeout: 120
    connect_timeout: 60
```

### Verify Connection

After configuring, verify the connection:

```bash
hermes tools                    # open curses UI
                                # filter "pencil" or "design"
                                # should list all Pencil operations
```

If tools don't appear:
- Check `~/.hermes/logs/agent.log` for MCP connection errors
- Verify the Pencil server is running and reachable
- Try `hermes /reload-mcp` to reconnect

## Available Operations

Once connected, all Pencil MCP tools are available. Common operations:

| Operation | Purpose |
|-----------|---------|
| `open_document` | Open a .pen file for editing |
| `get_editor_state` | Query current editor state (selection, zoom, viewport) |
| `batch_design` | Insert/update/delete/move design nodes in bulk |
| `batch_get` | Fetch multiple nodes by ID or pattern |
| `get_screenshot` | Capture current viewport as image |
| `snapshot_layout` | Get layout analysis (spacing, alignment, structure) |
| `export_nodes` | Export selected nodes as images (PNG, SVG, etc.) |
| `get_variables` | Retrieve design variables and theme definitions |
| `set_variables` | Update design variables and apply themes |
| `get_guidelines` | Fetch design guidelines and style archetypes |
| `find_empty_space_on_canvas` | Find available space for new elements |
| `search_all_unique_properties` | Find all unique property values in design |
| `replace_all_matching_properties` | Bulk-replace properties matching patterns |

Full operation docs are available in the Pencil MCP server reference.

## Workflow Examples

### 1. Create a New Design from Scratch

```
Agent: I'll create a mobile app login screen design for you.

Action: open_document → create new design document

Then:
- Use batch_design to insert frame containers, text fields, buttons
- Use set_variables to define colors, typography, spacing
- Use get_screenshot to verify the design looks right
- Use export_nodes to save mockups as PNG
```

### 2. Modify an Existing Design

```
User: Update the button colors to our new brand colors.

Agent: I'll open your design, find all buttons, and update their fill colors.

Action: open_document → batch_get all button components
→ replace_all_matching_properties to update fill colors
→ set_variables to sync the design tokens
→ get_screenshot to verify changes
```

### 3. Bulk Design Operations

```
User: Add a 16px padding to all containers.

Agent: I'll find all container frames and update their padding property.

Action: batch_get containers
→ batch_design with padding updates
→ snapshot_layout to verify spacing
```

### 4. Design System Management

```
User: Create a design system with color tokens and typography.

Agent: I'll set up design variables for colors, fonts, and spacing.

Action: get_guidelines → set_variables with theme definitions
→ Create reusable component instances
→ get_variables to confirm token setup
```

### 5. Export Assets for Development

```
User: Export all component screens as PNG for the dev team.

Agent: I'll extract each component and export as individual images.

Action: batch_get component nodes
→ export_nodes to PNG
→ Return file paths to uploaded assets
```

## Design Patterns & Best Practices

### Naming Conventions

Use clear, semantic names for components and variables:

- **Colors**: `color.primary`, `color.secondary`, `color.success`, `color.error`
- **Typography**: `typography.heading-1`, `typography.body`, `typography.caption`
- **Spacing**: `spacing.xs`, `spacing.sm`, `spacing.md`, `spacing.lg`, `spacing.xl`
- **Components**: `Button.Primary`, `Button.Secondary`, `Card.Default`, `Card.Featured`

### Batch Operations Best Practices

When using `batch_design` for bulk updates:

1. **Plan the operations** — list all nodes to update before calling batch_design
2. **Use bindings** — give temporary names to inserted nodes for follow-up operations
3. **Test incrementally** — use `get_screenshot` after each major batch to verify
4. **Keep descendants organized** — use the `descendants` map when copying components to customize sub-elements

Example batch operation:

```javascript
// Insert frame
container=I("parentId",{type:"frame",layout:"vertical",gap:16,padding:16})

// Insert children into that frame
title=I(container,{type:"text",content:"Title",fontSize:24})
body=I(container,{type:"text",content:"Body",fontSize:14})

// Update a nested element inside a component instance
card=I("parentId",{type:"ref",ref:"CardComponent"})
U(card+"/cardTitle",{content:"New Title"})
```

### Component Instance Customization

When working with component instances (reusable=true):

- Use `ref` nodes to create instances of components
- Override properties with the instance's properties object
- Use `descendants` to customize child elements within instances
- Use subsequent `U` (update) operations for nested property changes

## Troubleshooting

### "Tool not found" / MCP operations not available

1. **Check config.yaml**:
   ```bash
   cat ~/.hermes/config.yaml | grep -A 10 "mcp_servers:"
   ```

2. **Check server is running**:
   - Verify Pencil MCP server is accessible
   - Check logs at `~/.hermes/logs/agent.log` for connection errors

3. **Reconnect**:
   ```bash
   hermes /reload-mcp
   ```

### "Connection timeout" / Slow responses

- Increase `timeout` in config.yaml (default 120s)
- Check network latency if using HTTP/SSH transport
- Verify Pencil server isn't overloaded

### "File not found" when opening .pen files

- Ensure file paths are absolute or relative to current working directory
- Use `terminal(cwd=...)` to change working directory before opening
- Verify you have read/write permissions on the file

### Design edits not persisting

- Confirm that Pencil MCP is saving changes to disk (should be automatic)
- Check file system permissions on the .pen file
- Look for any "read-only" or "locked" status in editor state

## Integration with Other Skills

### With `claude-design`

Combine `pencil-design` (programmatic .pen editing) with `claude-design` (design process/taste) when building complex interfaces:

- Use `pencil-design` for automated batch updates, exports, and system management
- Use `claude-design` for one-off artistry, visual polish, and human-facing mockups

### With `design-md`

Use `design-md` alongside `pencil-design` when exporting design tokens:

- Define variables in Pencil design system
- Export to DESIGN.md format with `design-md` skill
- Author design-token specs for downstream consumption

### With `popular-web-designs`

Reference existing design systems (Stripe, Linear, Vercel, etc.) to inform Pencil design decisions:

- Pull visual vocabulary from `popular-web-designs`
- Implement color, typography, and spacing in Pencil
- Export tokens to DESIGN.md or code

## References

- **Pencil MCP docs**: https://github.com/pencil-mcp/docs (adjust as needed)
- **MCP specification**: https://modelcontextprotocol.io/
- **Related skills**: `design-md`, `claude-design`, `popular-web-designs`
- **Hermes MCP integration**: See `/reload-mcp` command and `mcp_servers:` config section

## Notes

- Pencil MCP is accessed through Hermes' generic MCP client in `tools/mcp_tool.py`
- All .pen file operations are mediated by the Pencil server — no direct file parsing
- Design changes are persisted to disk by the Pencil MCP server automatically
- Screenshots and exports are served by the Pencil server (may require additional file handling)
