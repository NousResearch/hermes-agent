#!/usr/bin/env python3
"""
Spotify sync template for OCAS skills
Template for syncing data from external services via MCP
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

class ExternalServiceSync:
    """Template for syncing data from external services"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".hermes" / "commons" / "data" / "ocas-{skill}"
        self.config_file = self.data_dir / "config.json"
        self.signals_file = self.data_dir / "signals.jsonl"
        self.items_file = self.data_dir / "items.jsonl"
        self.checkpoint_file = self.data_dir / "checkpoints" / "sync_checkpoint.json"

        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from config.json"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}

    def _append_jsonl(self, file_path: Path, data: Dict):
        """Append a record to a JSONL file"""
        with open(file_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _read_jsonl(self, file_path: Path) -> List[Dict]:
        """Read all records from a JSONL file"""
        records = []
        if file_path.exists():
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        return records

    def _load_checkpoint(self) -> Dict:
        """Load sync checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {"last_sync": None, "processed_ids": []}

    def _save_checkpoint(self, checkpoint: Dict):
        """Save sync checkpoint"""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def sync(self, limit: int = 50) -> Dict:
        """Main sync method - customize for your service"""
        print("Syncing data from external service...")
        
        # Load checkpoint
        checkpoint = self._load_checkpoint()
        processed_ids = set(checkpoint.get("processed_ids", []))
        
        # Fetch data from external service
        # This is where you'd call the MCP tools or external API
        data = self._fetch_external_data(limit)
        
        if not data:
            print("No data found")
            return {"error": "No data found"}
        
        print(f"Found {len(data)} items")
        
        signals_created = 0
        items_updated = 0
        
        for item in data:
            item_id = item.get("id")
            
            if not item_id:
                continue
            
            # Skip if already processed
            if item_id in processed_ids:
                continue
            
            # Create signal
            signal = {
                "signal_id": f"sig-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_created}",
                "domain": "{domain}",  # Customize for your skill
                "source_type": "{source_type}",  # Customize for your skill
                "venue_name": item.get("name"),
                "event_date": item.get("date", datetime.now().isoformat()),
                "strength": 0.60,
                "created_at": datetime.now().isoformat(),
                "extraction_source": "{service_name}",  # Customize for your service
                "item_id": item_id
            }
            
            self._append_jsonl(self.signals_file, signal)
            signals_created += 1
            
            # Update item record
            self._update_item_record(item)
            items_updated += 1
            
            # Mark as processed
            processed_ids.add(item_id)
        
        # Save checkpoint
        checkpoint["last_sync"] = datetime.now().isoformat()
        checkpoint["processed_ids"] = list(processed_ids)
        self._save_checkpoint(checkpoint)
        
        return {
            "signals_created": signals_created,
            "items_updated": items_updated,
            "last_sync": checkpoint["last_sync"]
        }

    def _fetch_external_data(self, limit: int) -> List[Dict]:
        """Fetch data from external service - customize for your service"""
        # This is where you'd call MCP tools or external APIs
        # For example, using the agent's tool system to call MCP tools
        # Or using subprocess to call external APIs
        
        # Placeholder implementation
        return []

    def _update_item_record(self, item: Dict):
        """Create or update item record"""
        item_name = item.get("name")
        item_id = item.get("id")
        
        if not item_name:
            return
        
        # Read existing items
        items = self._read_jsonl(self.items_file)
        
        # Find existing item by item_id
        existing_item = None
        for i in items:
            if i.get("item_id") == item_id:
                existing_item = i
                break
        
        if existing_item:
            # Update existing
            existing_item['signal_count'] = existing_item.get('signal_count', 0) + 1
            existing_item['last_seen'] = datetime.now().isoformat()
            if 'visit_dates' not in existing_item:
                existing_item['visit_dates'] = []
            existing_item['visit_dates'].append(datetime.now().isoformat())
        else:
            # Create new
            new_item = {
                "item_id": f"item-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "venue_name": item_name,
                "domain": "{domain}",  # Customize for your skill
                "signal_count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "visit_dates": [datetime.now().isoformat()],
                "enriched": False,
                "external_id": item_id
            }
            items.append(new_item)
        
        # Write back
        with open(self.items_file, 'w') as f:
            for i in items:
                f.write(json.dumps(i) + '\n')

    def get_status(self) -> Dict:
        """Get sync status"""
        checkpoint = self._load_checkpoint()
        
        signals = self._read_jsonl(self.signals_file)
        domain_signals = [s for s in signals if s.get("domain") == "{domain}"]  # Customize
        
        items = self._read_jsonl(self.items_file)
        domain_items = [i for i in items if i.get("domain") == "{domain}"]  # Customize
        
        return {
            "last_sync": checkpoint.get("last_sync"),
            "processed_count": len(checkpoint.get("processed_ids", [])),
            "total_signals": len(domain_signals),
            "total_items": len(domain_items)
        }


def main():
    """CLI entry point"""
    import sys

    sync = ExternalServiceSync()

    if len(sys.argv) < 2:
        print("Usage: {service}_sync.py <command>")
        print("Commands: sync, status")
        sys.exit(1)

    command = sys.argv[1]

    if command == "sync":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        result = sync.sync(limit)
        print(json.dumps(result, indent=2))

    elif command == "status":
        status = sync.get_status()
        print(json.dumps(status, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()