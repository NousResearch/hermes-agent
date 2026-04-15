#!/usr/bin/env python3
"""
Email and calendar scan for Taste skill.
Extracts consumption signals from Gmail and Google Calendar.
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    print("Error: google-api-python-client not installed")
    sys.exit(1)

# Paths
DATA_DIR = Path(__file__).parent.parent
EXTRACTIONS_FILE = DATA_DIR / "extractions.jsonl"
SIGNALS_FILE = DATA_DIR / "signals.jsonl"
ITEMS_FILE = DATA_DIR / "items.jsonl"
CONFIG_FILE = DATA_DIR / "config.json"

# Google OAuth token path
TOKEN_PATH = Path.home() / ".hermes" / "google_token.json"

def load_config():
    """Load Taste config."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    """Save Taste config."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_gmail_service():
    """Get authenticated Gmail service."""
    if not TOKEN_PATH.exists():
        print("Error: Google OAuth token not found at ~/.hermes/google_token.json")
        print("Run Google Workspace setup first")
        sys.exit(1)
    
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    
    return build('gmail', 'v1', credentials=creds)

def get_calendar_service():
    """Get authenticated Calendar service."""
    if not TOKEN_PATH.exists():
        print("Error: Google OAuth token not found")
        sys.exit(1)
    
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    
    return build('calendar', 'v3', credentials=creds)

def normalize_venue_name(name: str) -> str:
    """Normalize venue name for deduplication."""
    name = name.lower()
    name = re.sub(r'\bthe\b', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[^\w\s]', '', name)
    return name.strip()

def compute_dedup_key(service: str, order_id: str, event_date: str, venue_name: str) -> str:
    """Compute deduplication key."""
    normalized = normalize_venue_name(venue_name)
    return f"{service}:{order_id}:{event_date}:{normalized}"

def extract_from_email(service: str, message: Dict, config: Dict) -> Dict:
    """Extract consumption data from email message."""
    subject = message.get('payload', {}).get('headers', [])
    subject_text = next((h['value'] for h in subject if h['name'] == 'Subject'), '')
    
    # Simple extraction based on service patterns
    extraction = {
        "extraction_id": f"email-{message['id']}",
        "source_service": service,
        "email_type": "confirmation",
        "cancelled": False,
        "extracted_at": datetime.utcnow().isoformat(),
        "raw_subject": subject_text
    }
    
    # Service-specific extraction hints would go here
    # For now, this is a placeholder that would need LLM-based extraction
    
    return extraction

def scan_gmail(config: Dict, last_scan: str = None) -> List[Dict]:
    """Scan Gmail for consumption signals."""
    service = get_gmail_service()
    extractions = []
    
    # Build query for configured email sources
    email_sources = config.get('email_sources', {})
    queries = []
    
    for service_name, service_config in email_sources.items():
        for pattern in service_config.get('sender_patterns', []):
            queries.append(f"from:{pattern}")
    
    if not queries:
        print("No email sources configured")
        return extractions
    
    query = " OR ".join(queries)
    
    # Add date filter if last_scan exists
    if last_scan:
        dt = datetime.fromisoformat(last_scan)
        date_str = dt.strftime('%Y/%m/%d')
        query = f"{query} after:{date_str}"
    
    try:
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=100
        ).execute()
        
        messages = results.get('messages', [])
        
        for msg in messages:
            msg_data = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()
            
            # Determine which service this is from
            from_header = next((h['value'] for h in msg_data.get('payload', {}).get('headers', []) 
                              if h['name'] == 'From'), '')
            
            service_name = None
            for svc, svc_config in email_sources.items():
                for pattern in svc_config.get('sender_patterns', []):
                    if pattern in from_header:
                        service_name = svc
                        break
                if service_name:
                    break
            
            if service_name:
                extraction = extract_from_email(service_name, msg_data, config)
                extractions.append(extraction)
        
        print(f"Found {len(extractions)} emails from configured sources")
        
    except Exception as e:
        print(f"Error scanning Gmail: {e}")
    
    return extractions

def scan_calendar(config: Dict, last_scan: str = None) -> List[Dict]:
    """Scan Google Calendar for restaurant reservations and hotel bookings."""
    service = get_calendar_service()
    extractions = []
    
    try:
        now = datetime.utcnow()
        time_min = now - timedelta(days=30)  # Look back 30 days
        time_max = now + timedelta(days=30)   # Look ahead 30 days
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min.isoformat() + 'Z',
            timeMax=time_max.isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        for event in events:
            summary = event.get('summary', '')
            location = event.get('location', '')
            
            # Simple pattern matching for restaurants and hotels
            # In production, this would use LLM-based extraction
            if any(keyword in summary.lower() for keyword in ['restaurant', 'dinner', 'lunch', 'brunch']):
                extraction = {
                    "extraction_id": f"calendar-{event['id']}",
                    "source_service": "calendar",
                    "email_type": "visit",
                    "cancelled": False,
                    "extracted_at": datetime.utcnow().isoformat(),
                    "raw_subject": summary,
                    "venue_name": summary,
                    "event_date": event.get('start', {}).get('dateTime') or event.get('start', {}).get('date')
                }
                extractions.append(extraction)
        
        print(f"Found {len(extractions)} calendar events")
        
    except Exception as e:
        print(f"Error scanning Calendar: {e}")
    
    return extractions

def deduplicate_extractions(extractions: List[Dict]) -> Dict:
    """Deduplicate extractions and handle cancellations."""
    groups = {}
    
    for extraction in extractions:
        # Compute dedup key
        service = extraction.get('source_service', 'unknown')
        order_id = extraction.get('order_id', extraction.get('extraction_id'))
        event_date = extraction.get('event_date', extraction.get('extracted_at'))
        venue_name = extraction.get('venue_name', extraction.get('raw_subject', ''))
        
        dedup_key = compute_dedup_key(service, order_id, event_date, venue_name)
        
        if dedup_key not in groups:
            groups[dedup_key] = []
        groups[dedup_key].append(extraction)
    
    # Process groups
    results = {
        "distinct": [],
        "cancelled": [],
        "possible_matches": []
    }
    
    for key, group in groups.items():
        # Check for cancellations
        if any(e.get('cancelled', False) for e in group):
            results["cancelled"].extend(group)
            continue
        
        # Select richest extraction
        richest = max(group, key=lambda e: len(e))
        results["distinct"].append(richest)
    
    return results

def promote_to_signals(extractions: List[Dict], config: Dict) -> List[Dict]:
    """Promote extractions to consumption signals."""
    signals = []
    
    for extraction in extractions:
        # Create signal
        signal = {
            "signal_id": f"signal-{extraction['extraction_id']}",
            "item_id": f"item-{extraction['extraction_id']}",
            "domain": config.get('email_sources', {}).get(
                extraction.get('source_service', ''), {}
            ).get('domain', 'unknown'),
            "source": config.get('email_sources', {}).get(
                extraction.get('source_service', ''), {}
            ).get('source_type', 'unknown'),
            "strength": config.get('strength', {}).get(
                f"base_{config.get('email_sources', {}).get(
                    extraction.get('source_service', ''), {}
                ).get('source_type', 'manual')}", 0.60
            ),
            "timestamp": extraction.get('event_date') or extraction.get('extracted_at'),
            "metadata": {
                "extraction_id": extraction['extraction_id'],
                "source_service": extraction['source_service']
            }
        }
        signals.append(signal)
    
    return signals

def main():
    """Main scan function."""
    print("Starting email and calendar scan...")
    
    # Load config
    config = load_config()
    last_scan = config.get('email_scan', {}).get('last_scan_timestamp')
    
    # Scan Gmail
    print("Scanning Gmail...")
    email_extractions = scan_gmail(config, last_scan)
    
    # Scan Calendar
    print("Scanning Google Calendar...")
    calendar_extractions = scan_calendar(config, last_scan)
    
    # Combine extractions
    all_extractions = email_extractions + calendar_extractions
    
    if not all_extractions:
        print("No new extractions found")
        return
    
    # Deduplicate
    print("Deduplicating extractions...")
    dedup_results = deduplicate_extractions(all_extractions)
    
    # Promote to signals
    print("Promoting to signals...")
    signals = promote_to_signals(dedup_results['distinct'], config)
    
    # Write extractions
    with open(EXTRACTIONS_FILE, 'a') as f:
        for extraction in all_extractions:
            f.write(json.dumps(extraction) + '\n')
    
    # Write signals
    if signals:
        with open(SIGNALS_FILE, 'a') as f:
            for signal in signals:
                f.write(json.dumps(signal) + '\n')
        print(f"Created {len(signals)} signals")
    
    # Update last scan timestamp
    config['email_scan']['last_scan_timestamp'] = datetime.utcnow().isoformat()
    save_config(config)
    
    print(f"Scan complete. Last scan: {config['email_scan']['last_scan_timestamp']}")
    print(f"  Extractions: {len(all_extractions)}")
    print(f"  Signals: {len(signals)}")
    print(f"  Cancelled: {len(dedup_results['cancelled'])}")

if __name__ == "__main__":
    main()