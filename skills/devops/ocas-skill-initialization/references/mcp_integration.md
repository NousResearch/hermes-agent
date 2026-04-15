# MCP Integration Pattern

How to integrate Model Context Protocol (MCP) servers with OCAS skills.

## Discovery

Find existing MCP servers:
```bash
# Check installed MCP servers
find ~/.hermes/node/lib/node_modules -name "*mcp*" -type d

# Check MCP config directory
ls -la ~/.hermes/mcp/
```

## Configuration

Create MCP config at `~/.hermes/mcp/{service}-mcp.json`:
```json
{
  "{service}": {
    "command": "node",
    "args": ["/path/to/mcp/server/build/bin.js"],
    "env": {
      "CLIENT_ID": "${CLIENT_ID}",
      "CLIENT_SECRET": "${CLIENT_SECRET}"
    }
  }
}
```

## Calling MCP Tools from Python

```python
import subprocess

def call_mcp_tool(service, tool_name, args=None):
    """Call MCP tool via hermes mcp."""
    cmd = ["hermes", "mcp", "call", service, tool_name]
    if args:
        for key, value in args.items():
            cmd.extend([f"--{key}", str(value)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout
```

## Common MCP Servers

### Spotify MCP (`@darrenjaws/spotify-mcp`)

**Installation**: Already installed at `~/.hermes/node/lib/node_modules/@darrenjaws/spotify-mcp/`

**Setup**:
```bash
npx @darrenjaws/spotify-mcp setup
```

**Tools**:
- `get_user_profile` — User profile info
- `get_recently_played` — Recently played tracks
- `get_top_items` — Top tracks/artists

**Example**:
```python
# Get recently played
output = call_mcp_tool("spotify", "get_recently_played", {"limit": 50})

# Get top tracks
output = call_mcp_tool("spotify", "get_top_items", {
    "type": "tracks",
    "limit": 20,
    "time_range": "short_term"
})
```

### Tavily MCP

**Tools**:
- `tavily-search` — Web search

**Example**:
```python
output = call_mcp_tool("tavily", "tavily-search", {
    "query": "search terms"
})
```

## Parsing MCP Output

MCP tools return text output that needs parsing:

```python
def parse_recently_played(output):
    """Parse recently played output from MCP."""
    tracks = []
    lines = output.strip().split('\n')
    
    for line in lines:
        # Format: "1. Song Name - Artist (played at time)"
        if line.strip() and line[0].isdigit():
            parts = line.split('. ', 1)
            if len(parts) < 2:
                continue
            
            track_info = parts[1]
            if ' - ' in track_info:
                name, artist = track_info.split(' - ', 1)
                tracks.append({
                    "name": name.strip(),
                    "artist": artist.strip()
                })
    
    return tracks
```

## Environment Variables

Set required environment variables:
```bash
export SPOTIFY_CLIENT_ID='your_client_id'
export SPOTIFY_CLIENT_SECRET='your_client_secret'
```

Or add to `~/.bashrc` for persistence.

## Testing

Test MCP integration:
```bash
# Test MCP server
hermes mcp call spotify get_user_profile

# Test from Python script
python3 -c "
import subprocess
result = subprocess.run(['hermes', 'mcp', 'call', 'spotify', 'get_user_profile'], 
                       capture_output=True, text=True)
print(result.stdout)
"
```

## Troubleshooting

**MCP not found**:
- Check MCP config: `cat ~/.hermes/mcp/{service}-mcp.json`
- Verify server path exists
- Test with `hermes mcp call {service} {tool_name}`

**Environment variables missing**:
- Check: `echo $SPOTIFY_CLIENT_ID`
- Set: `export SPOTIFY_CLIENT_ID='your_value'`

**Timeout errors**:
- Increase timeout in subprocess call
- Check MCP server is running

**Output parsing errors**:
- Print raw output for debugging
- Adjust parsing logic based on actual format

## Best Practices

1. **Error handling**: Always check return code and stderr
2. **Timeouts**: Set reasonable timeouts (30-60 seconds)
3. **Parsing**: Be defensive with output parsing
4. **Deduplication**: Check existing data before creating new records
5. **Logging**: Log MCP calls and results for debugging
6. **Environment**: Document required environment variables in README

## Example: Complete Spotify Integration

```python
import subprocess
import json
from datetime import datetime
from pathlib import Path

def call_mcp_tool(service, tool_name, args=None):
    """Call MCP tool via hermes mcp."""
    cmd = ["hermes", "mcp", "call", service, tool_name]
    if args:
        for key, value in args.items():
            cmd.extend([f"--{key}", str(value)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Error calling MCP tool {tool_name}: {result.stderr}")
        return None
    return result.stdout

def sync_spotify():
    """Sync Spotify listening history."""
    # Check environment variables
    if not os.getenv("SPOTIFY_CLIENT_ID"):
        print("Error: SPOTIFY_CLIENT_ID not set")
        return
    
    # Get recently played
    output = call_mcp_tool("spotify", "get_recently_played", {"limit": 50})
    if not output:
        return
    
    # Parse output
    tracks = parse_recently_played(output)
    
    # Create signals
    for track in tracks:
        signal = {
            "signal_id": f"spotify-{track['name']}-{datetime.utcnow().isoformat()}",
            "item_id": f"track-{track['name'].lower().replace(' ', '-')}",
            "domain": "music",
            "source": "play",
            "strength": 0.60,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "artist": track['artist'],
                "mcp_source": "spotify"
            }
        }
        # Write to signals.jsonl
        with open("signals.jsonl", 'a') as f:
            f.write(json.dumps(signal) + '\n')
```

## References

- MCP specification: https://modelcontextprotocol.io/
- Spotify MCP: https://github.com/darrenjaworski/spotify-mcp
- Hermes MCP integration: `hermes mcp --help`