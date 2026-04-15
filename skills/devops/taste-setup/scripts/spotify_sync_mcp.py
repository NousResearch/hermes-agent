#!/usr/bin/env python3
"""
Spotify listening history sync for Taste skill using MCP.
Pulls recent plays and top tracks via Spotify MCP server.
"""
import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent
SIGNALS_FILE = DATA_DIR / "signals.jsonl"
ITEMS_FILE = DATA_DIR / "items.jsonl"
CHECKPOINT_FILE = DATA_DIR / "music" / "spotify_sync_checkpoint.json"

def load_checkpoint():
    """Load last sync timestamp."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_sync": None}

def save_checkpoint(checkpoint):
    """Save sync timestamp."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_existing_items():
    """Load existing items to avoid duplicates."""
    items = {}
    if ITEMS_FILE.exists():
        with open(ITEMS_FILE, 'r') as f:
            for line in f:
                item = json.loads(line)
                items[item.get("item_id", "")] = item
    return items

def load_existing_signals():
    """Load existing signals to avoid duplicates."""
    signals = {}
    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                signal = json.loads(line)
                signals[signal.get("signal_id", "")] = signal
    return signals

def call_mcp_tool(tool_name, args=None):
    """Call Spotify MCP tool via hermes CLI."""
    cmd = ["hermes", "mcp", "call", "spotify", tool_name]
    if args:
        for key, value in args.items():
            cmd.extend([f"--{key}", str(value)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Error calling MCP tool {tool_name}: {result.stderr}")
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"Timeout calling MCP tool {tool_name}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing MCP response: {e}")
        return None

def sync_recent_plays(existing_items, existing_signals):
    """Sync recently played tracks via MCP."""
    new_signals = []
    new_items = {}
    
    # Call spotify_get_recently_played
    result = call_mcp_tool("spotify_get_recently_played", {"limit": 50})
    
    if not result or "items" not in result:
        print("No recent plays data returned from MCP")
        return new_signals, new_items
    
    now = datetime.utcnow().isoformat()
    
    for item in result["items"]:
        track = item.get("track", {})
        played_at = item.get("played_at")
        track_id = track.get("id")
        
        if not track_id or not played_at:
            continue
        
        # Skip if already have this signal
        signal_id = f"spotify-play-{track_id}-{played_at}"
        if signal_id in existing_signals:
            continue
        
        # Create or update item
        item_id = f"spotify-track-{track_id}"
        if item_id not in existing_items:
            artists = track.get("artists", [])
            artist_name = artists[0].get("name") if artists else None
            album = track.get("album", {})
            
            item_data = {
                "item_id": item_id,
                "name": track.get("name"),
                "domain": "music",
                "source": "spotify",
                "metadata": {
                    "artist": artist_name,
                    "album": album.get("name") if album else None,
                    "duration_ms": track.get("duration_ms"),
                    "spotify_url": track.get("external_urls", {}).get("spotify"),
                    "track_id": track_id
                },
                "enriched": True,
                "enriched_at": now,
                "first_seen": played_at,
                "last_seen": played_at,
                "signal_count": 0,
                "visit_dates": []
            }
            new_items[item_id] = item_data
        else:
            item_data = existing_items[item_id]
            item_data['last_seen'] = played_at
            item_data['signal_count'] += 1
            item_data['visit_dates'].append(played_at)
        
        # Create signal
        signal_data = {
            "signal_id": signal_id,
            "item_id": item_id,
            "domain": "music",
            "source": "play",
            "strength": 0.60,
            "timestamp": played_at,
            "metadata": {
                "played_at": played_at,
                "context": item.get("context", {}).get("type") if item.get("context") else None
            }
        }
        new_signals.append(signal_data)
    
    return new_signals, new_items

def sync_top_tracks(existing_items, existing_signals):
    """Sync top tracks via MCP."""
    new_signals = []
    new_items = {}
    
    # Call spotify_get_top_items for tracks
    result = call_mcp_tool("spotify_get_top_items", {"type": "tracks", "time_range": "short_term", "limit": 20})
    
    if not result or "items" not in result:
        print("No top tracks data returned from MCP")
        return new_signals, new_items
    
    now = datetime.utcnow().isoformat()
    
    for item in result["items"]:
        track_id = item.get("id")
        
        if not track_id:
            continue
        
        # Create or update item
        item_id = f"spotify-track-{track_id}"
        if item_id not in existing_items:
            artists = item.get("artists", [])
            artist_name = artists[0].get("name") if artists else None
            album = item.get("album", {})
            
            item_data = {
                "item_id": item_id,
                "name": item.get("name"),
                "domain": "music",
                "source": "spotify",
                "metadata": {
                    "artist": artist_name,
                    "album": album.get("name") if album else None,
                    "duration_ms": item.get("duration_ms"),
                    "spotify_url": item.get("external_urls", {}).get("spotify"),
                    "track_id": track_id
                },
                "enriched": True,
                "enriched_at": now,
                "first_seen": now,
                "last_seen": now,
                "signal_count": 0,
                "visit_dates": []
            }
            new_items[item_id] = item_data
        else:
            item_data = existing_items[item_id]
            item_data['last_seen'] = now
            item_data['signal_count'] += 1
            item_data['visit_dates'].append(now)
        
        # Create signal
        signal_id = f"spotify-top-{track_id}-{now}"
        signal_data = {
            "signal_id": signal_id,
            "item_id": item_id,
            "domain": "music",
            "source": "play",
            "strength": 0.60,
            "timestamp": now,
            "metadata": {
                "top_track": True,
                "time_range": "short_term"
            }
        }
        new_signals.append(signal_data)
    
    return new_signals, new_items

def main():
    """Main sync function."""
    print("Starting Spotify sync via MCP...")
    
    # Load existing data
    existing_items = load_existing_items()
    existing_signals = load_existing_signals()
    checkpoint = load_checkpoint()
    
    # Check if Spotify MCP is configured
    if not os.getenv("SPOTIFY_CLIENT_ID") or not os.getenv("SPOTIFY_CLIENT_SECRET"):
        print("Warning: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables not set")
        print("Spotify sync will be skipped")
        print("Set them with:")
        print("  export SPOTIFY_CLIENT_ID='your_client_id'")
        print("  export SPOTIFY_CLIENT_SECRET='your_client_secret'")
        return
    
    # Sync recent plays
    print("Syncing recent plays...")
    recent_signals, recent_items = sync_recent_plays(existing_items, existing_signals)
    
    # Sync top tracks
    print("Syncing top tracks...")
    top_signals, top_items = sync_top_tracks(existing_items, existing_signals)
    
    # Merge new items
    all_new_items = {**recent_items, **top_items}
    
    # Write new signals
    if recent_signals or top_signals:
        with open(SIGNALS_FILE, 'a') as f:
            for signal in recent_signals + top_signals:
                f.write(json.dumps(signal) + '\n')
        print(f"Created {len(recent_signals + top_signals)} new signals")
    
    # Write updated items
    if all_new_items:
        with open(ITEMS_FILE, 'a') as f:
            for item in all_new_items.values():
                f.write(json.dumps(item) + '\n')
        print(f"Updated {len(all_new_items)} items")
    
    # Save checkpoint
    checkpoint["last_sync"] = datetime.utcnow().isoformat()
    save_checkpoint(checkpoint)
    
    print(f"Sync complete. Last sync: {checkpoint['last_sync']}")

if __name__ == "__main__":
    main()