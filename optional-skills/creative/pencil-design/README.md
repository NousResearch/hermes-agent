# Pencil Design Skill

Programmatic design editing via Pencil MCP for Hermes Agent.

## Files in This Skill

| File | Purpose |
|------|---------|
| **SKILL.md** | Main skill documentation, setup, workflows, best practices |
| **SETUP.md** | Step-by-step configuration guide for Pencil MCP connection |
| **OPERATIONS.md** | Complete reference for all available Pencil MCP operations |
| **EXAMPLES.md** | Real-world examples with full agent workflows |
| **README.md** | This file — quick overview |

## What This Skill Does

Provides programmatic access to Pencil's design editing capabilities via the Pencil MCP protocol. Agent can:

✅ Create/edit .pen design files  
✅ Batch insert/update/delete design nodes  
✅ Export designs as images (PNG, SVG, JPEG)  
✅ Manage design systems and variables  
✅ Analyze design consistency  
✅ Apply themes and design tokens  
✅ Find and replace design properties  

## Quick Start (5 minutes)

### 1. Configure Pencil MCP

Edit `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  pencil:
    url: "http://localhost:5000/mcp"  # or your Pencil server URL
    timeout: 120
    connect_timeout: 60
```

See **SETUP.md** for alternative transports (stdio, SSH).

### 2. Verify Tools Are Available

```bash
hermes tools
# Filter for "pencil" or "design"
# Should list operations like:
#   - open_document
#   - batch_design
#   - export_nodes
#   - get_screenshot
#   - set_variables
#   etc.
```

### 3. Try It

```bash
hermes
# Type: Create a mobile app login screen in ~/designs/login.pen
# Agent will use Pencil MCP to build the design programmatically
```

## Common Tasks

### Create a design from scratch

```
User: Create a landing page design in ~/designs/landing.pen
Agent: Uses batch_design to insert frames, text, buttons, images
```

### Update design colors

```
User: Change all blue buttons to purple
Agent: Uses replace_all_matching_properties to bulk-update colors
```

### Export assets

```
User: Export all screens as PNG for the website
Agent: Uses export_nodes to generate image files
```

### Manage design system

```
User: Set up our design tokens (colors, fonts, spacing)
Agent: Uses set_variables to define reusable tokens
```

## Documentation Files

- **SKILL.md** — Full reference, workflows, design patterns, integration with other skills
- **SETUP.md** — Configuration options, troubleshooting, advanced setup
- **OPERATIONS.md** — API reference for all Pencil MCP operations
- **EXAMPLES.md** — Seven real-world examples with complete workflows

## When to Use This Skill

✅ **Use this skill when:**
- Agent needs to programmatically edit design files
- Bulk design operations (color updates, spacing fixes)
- Exporting designs as marketing assets
- Building design systems with reusable components
- Analyzing design consistency

❌ **Don't use when:**
- Creating HTML/CSS prototypes (use `claude-design`)
- Authoring design token specs (use `design-md`)
- Looking up brand design systems (use `popular-web-designs`)

## MCP Server Requirements

You must have a Pencil MCP server running and accessible. The server can be:

- **Local command** (stdio): `npx @pencil-mcp/server`
- **HTTP service**: Running on localhost or remote host
- **SSH tunnel**: Remote server over SSH

Configure the connection in `~/.hermes/config.yaml` under `mcp_servers.pencil`.

## Integration with Other Skills

| Skill | How they work together |
|-------|----------------------|
| `claude-design` | pencil-design does programmatic edits; claude-design does design process/taste |
| `design-md` | Export Pencil variables to DESIGN.md token specs |
| `popular-web-designs` | Reference brand design systems; implement in Pencil |

## Troubleshooting

**"Pencil tools not showing in `hermes tools`?"**
- Check `mcp_servers.pencil` is configured in config.yaml
- Verify Pencil server is running and reachable
- Try `hermes /reload-mcp` to reconnect
- Check logs: `hermes logs --level DEBUG | grep pencil`

**"Connection timeout errors?"**
- Increase `timeout` in config.yaml (default 120s)
- Verify network connectivity to Pencil server
- For HTTP: `curl http://localhost:5000/mcp` to test

**"Design edits not saving?"**
- Pencil MCP handles persistence automatically
- Check file permissions on .pen files
- Verify Pencil server has write access

See **SETUP.md** for more detailed troubleshooting.

## References

- **Full Skill Guide**: SKILL.md (150+ lines of documentation)
- **Configuration**: SETUP.md (environment vars, multiple servers, advanced setup)
- **API Reference**: OPERATIONS.md (complete operation signatures)
- **Examples**: EXAMPLES.md (7 real-world workflows)
- **Pencil MCP Docs**: https://github.com/pencil-mcp/docs (adjust URL)
- **MCP Spec**: https://modelcontextprotocol.io/

## Contributing

To improve this skill:

1. Test Pencil MCP operations thoroughly
2. Add new examples to EXAMPLES.md if you discover useful workflows
3. Update troubleshooting section if you hit and resolve new issues
4. Keep OPERATIONS.md in sync with actual Pencil MCP schema

## License

MIT (same as Hermes Agent)
