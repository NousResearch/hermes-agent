# Aviation MCP-UI

Stakeholder-demo MCP server for branded aviation risk dashboards. The server exposes four tools that return MCP-UI `rawHtml` resources backed by local fixtures:

- `get_route_risks` — route map with alert pins and corridor risk summary
- `get_airport` — airport deep-dive card
- `get_fir` — FIR/conflict-zone threat panel
- `get_fleet_status` — fleet status board with issue highlighting

## Run

```bash
npm install
npm run build
node dist/server.js
```

## Claude Desktop Config

```json
{
  "mcpServers": {
    "aviation-mcp-ui": {
      "command": "node",
      "args": ["/absolute/path/to/aviation-mcp-ui/dist/server.js"],
      "cwd": "/absolute/path/to/aviation-mcp-ui"
    }
  }
}
```

## Demo Prompts

- Show me security risks along my flight route from PHX to JFK.
- Tell me about LSZH.
- What is happening in the Baghdad FIR?
- Are any planes in my fleet experiencing issues?
