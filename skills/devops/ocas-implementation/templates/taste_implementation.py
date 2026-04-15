#!/usr/bin/env python3
"""
Taste skill implementation - email/calendar scanning and enrichment
Template for OCAS skills that need to scan Gmail/Calendar
"""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import base64

# Google API imports
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Warning: Google API libraries not available. Install with: pip install google-api-python-client google-auth-oauthlib")


class TasteSkill:
    """Template for OCAS skills that scan Gmail/Calendar"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".hermes" / "commons" / "data" / "ocas-taste"
        self.config_file = self.data_dir / "config.json"
        self.signals_file = self.data_dir / "signals.jsonl"
        self.items_file = self.data_dir / "items.jsonl"
        self.extractions_file = self.data_dir / "extractions.jsonl"
        self.decisions_file = self.data_dir / "decisions.jsonl"

        self.config = self._load_config()
        self.gmail_service = None
        self.calendar_service = None

    def _load_config(self) -> Dict:
        """Load configuration from config.json"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}

    def _save_config(self):
        """Save configuration to config.json"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

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

    def _init_google_services(self):
        """Initialize Gmail and Calendar services"""
        if not GOOGLE_API_AVAILABLE:
            print("Google API libraries not available")
            return False

        token_path = Path.home() / ".hermes" / "google_token.json"

        if not token_path.exists():
            print(f"Google token not found at {token_path}")
            return False

        try:
            # IMPORTANT: Use scopes that match the existing token
            # Check ~/.hermes/google_token.json for the actual scopes
            creds = Credentials.from_authorized_user_file(
                str(token_path),
                ['https://www.googleapis.com/auth/gmail.modify', 
                 'https://www.googleapis.com/auth/calendar']
            )

            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    with open(token_path, 'w') as f:
                        f.write(creds.to_json())
                else:
                    print("Credentials invalid and cannot be refreshed")
                    return False

            self.gmail_service = build('gmail', 'v1', credentials=creds)
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            return True

        except Exception as e:
            print(f"Error initializing Google services: {e}")
            return False

    def scan_email(self, days_back: int = 30) -> Dict:
        """Scan Gmail for consumption signals"""
        if not self._init_google_services():
            return {"error": "Failed to initialize Google services"}

        results = {
            "extractions": [],
            "signals_created": 0,
            "cancellations": 0,
            "services_scanned": []
        }

        # Get last scan timestamp
        last_scan = self.config.get("email_scan", {}).get("last_scan_timestamp")
        if last_scan:
            last_scan_dt = datetime.fromisoformat(last_scan)
        else:
            last_scan_dt = datetime.now() - timedelta(days=days_back)

        # Build query for each service
        email_sources = self.config.get("email_sources", {})

        for service_name, service_config in email_sources.items():
            sender_patterns = service_config.get("sender_patterns", [])
            domain = service_config.get("domain")
            source_type = service_config.get("source_type")

            if not sender_patterns:
                continue

            # Build Gmail search query
            query_parts = []
            for pattern in sender_patterns:
                query_parts.append(f"from:{pattern}")

            # Add date filter
            date_str = last_scan_dt.strftime("%Y/%m/%d")
            query_parts.append(f"after:{date_str}")

            query = " OR ".join(query_parts)

            try:
                # Search messages
                messages_result = self.gmail_service.users().messages().list(
                    userId='me',
                    q=query
                ).execute()

                messages = messages_result.get('messages', [])

                for msg in messages:
                    msg_data = self.gmail_service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()

                    extraction = self._extract_from_email(msg_data, service_name, domain, source_type)
                    if extraction:
                        results["extractions"].append(extraction)

                results["services_scanned"].append(service_name)

            except Exception as e:
                print(f"Error scanning {service_name}: {e}")

        # Deduplicate and promote to signals
        signals, cancellations = self._process_extractions(results["extractions"])
        results["signals_created"] = len(signals)
        results["cancellations"] = cancellations

        # Update last scan timestamp
        self.config.setdefault("email_scan", {})["last_scan_timestamp"] = datetime.now().isoformat()
        self._save_config()

        return results

    def _extract_from_email(self, msg_data: Dict, service: str, domain: str, source_type: str) -> Optional[Dict]:
        """Extract consumption signal from email message"""
        try:
            headers = {h['name']: h['value'] for h in msg_data['payload'].get('headers', [])}
            subject = headers.get('Subject', '')
            from_addr = headers.get('From', '')
            date_str = headers.get('Date', '')

            # Parse date
            try:
                email_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            except:
                email_date = datetime.now()

            # Get email body
            body = self._get_email_body(msg_data['payload'])

            # Extract structured data based on service
            extraction = {
                "service": service,
                "domain": domain,
                "source_type": source_type,
                "from": from_addr,
                "subject": subject,
                "date": email_date.isoformat(),
                "body": body[:5000],  # Truncate to avoid huge records
                "email_type": self._classify_email_type(subject, body),
                "cancelled": False
            }

            # Service-specific extraction - customize for your skill
            if service == "doordash":
                extraction.update(self._extract_doordash(subject, body))
            elif service == "instacart":
                extraction.update(self._extract_instacart(subject, body))
            # Add more service extractors as needed

            return extraction

        except Exception as e:
            print(f"Error extracting from email: {e}")
            return None

    def _get_email_body(self, payload: Dict) -> str:
        """Extract email body from message payload"""
        body = ""

        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    if data:
                        body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                elif 'parts' in part:
                    body += self._get_email_body(part)

        elif 'body' in payload and 'data' in payload['body']:
            data = payload['body']['data']
            body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

        return body

    def _classify_email_type(self, subject: str, body: str) -> str:
        """Classify email type (confirmation, reminder, cancellation, receipt)"""
        subject_lower = subject.lower()
        body_lower = body.lower()

        if any(word in subject_lower for word in ['cancelled', 'canceled', 'cancel']):
            return 'cancellation'
        elif any(word in subject_lower for word in ['receipt', 'order complete', 'delivered']):
            return 'receipt'
        elif any(word in subject_lower for word in ['reminder', 'upcoming', 'tomorrow']):
            return 'reminder'
        elif any(word in subject_lower for word in ['confirmation', 'confirmed', 'booked']):
            return 'confirmation'
        else:
            return 'unknown'

    def _extract_doordash(self, subject: str, body: str) -> Dict:
        """Extract DoorDash order details - customize for your service"""
        venue_match = re.search(r'from\s+([^\n]+)', subject, re.IGNORECASE)
        venue = venue_match.group(1).strip() if venue_match else "Unknown"

        total_match = re.search(r'\$[\d,]+\.\d{2}', body)
        total = total_match.group(0) if total_match else None

        return {
            "venue_name": venue,
            "order_id": "unknown",
            "total": total,
            "items": []
        }

    def _extract_instacart(self, subject: str, body: str) -> Dict:
        """Extract Instacart order details - customize for your service"""
        store_match = re.search(r'from\s+([^\n]+)', subject, re.IGNORECASE)
        store = store_match.group(1).strip() if store_match else "Unknown"

        total_match = re.search(r'\$[\d,]+\.\d{2}', body)
        total = total_match.group(0) if total_match else None

        return {
            "venue_name": store,
            "order_id": "unknown",
            "total": total,
            "items": []
        }

    def _process_extractions(self, extractions: List[Dict]) -> tuple:
        """Deduplicate extractions and promote to signals"""
        # Group by dedup key
        groups = {}
        for extraction in extractions:
            dedup_key = self._compute_dedup_key(
                extraction['service'],
                extraction.get('order_id', 'unknown'),
                extraction['date'],
                extraction.get('venue_name', 'unknown')
            )
            if dedup_key not in groups:
                groups[dedup_key] = []
            groups[dedup_key].append(extraction)

        signals = []
        cancellations = 0

        for dedup_key, group in groups.items():
            # Check for cancellations
            if any(e.get('email_type') == 'cancellation' for e in group):
                cancellations += 1
                continue

            # Select richest extraction
            canonical = max(group, key=lambda e: len(e))

            # Create signal
            signal = {
                "signal_id": f"sig-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(signals)}",
                "domain": canonical['domain'],
                "source_type": canonical['source_type'],
                "venue_name": canonical.get('venue_name'),
                "event_date": canonical['date'],
                "strength": self._compute_base_strength(canonical['source_type']),
                "created_at": datetime.now().isoformat(),
                "extraction_source": canonical['service']
            }

            signals.append(signal)
            self._append_jsonl(self.signals_file, signal)

            # Create/update item record
            self._update_item_record(canonical)

        return signals, cancellations

    def _compute_dedup_key(self, service: str, order_id: str, event_date: str, venue_name: str) -> str:
        """Compute deduplication key for an extraction"""
        normalized_venue = venue_name.lower().strip()
        return f"{service}:{order_id}:{event_date}:{normalized_venue}"

    def _compute_base_strength(self, source_type: str) -> float:
        """Compute base strength for a signal type"""
        strength_config = self.config.get("strength", {})
        return strength_config.get(f"base_{source_type}", 0.70)

    def _update_item_record(self, extraction: Dict):
        """Create or update item record"""
        venue_name = extraction.get('venue_name')
        if not venue_name:
            return

        # Read existing items
        items = self._read_jsonl(self.items_file)

        # Find existing item
        existing_item = None
        for item in items:
            if item.get('venue_name') == venue_name:
                existing_item = item
                break

        if existing_item:
            # Update existing
            existing_item['signal_count'] = existing_item.get('signal_count', 0) + 1
            existing_item['last_seen'] = extraction['date']
            if 'visit_dates' not in existing_item:
                existing_item['visit_dates'] = []
            existing_item['visit_dates'].append(extraction['date'])
        else:
            # Create new
            item = {
                "item_id": f"item-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "venue_name": venue_name,
                "domain": extraction['domain'],
                "signal_count": 1,
                "first_seen": extraction['date'],
                "last_seen": extraction['date'],
                "visit_dates": [extraction['date']],
                "enriched": False
            }
            items.append(item)

        # Write back
        with open(self.items_file, 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')

    def get_status(self) -> Dict:
        """Get current status"""
        signals = self._read_jsonl(self.signals_file)
        items = self._read_jsonl(self.items_file)

        # Count by domain
        domain_counts = {}
        for signal in signals:
            domain = signal.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_signals": len(signals),
            "total_items": len(items),
            "domain_breakdown": domain_counts,
            "last_scan": self.config.get("email_scan", {}).get("last_scan_timestamp"),
            "email_scan_enabled": self.config.get("email_scan", {}).get("enabled", False)
        }


def main():
    """CLI entry point"""
    import sys

    skill = TasteSkill()

    if len(sys.argv) < 2:
        print("Usage: taste_implementation.py <command>")
        print("Commands: scan-email, status")
        sys.exit(1)

    command = sys.argv[1]

    if command == "scan-email":
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        result = skill.scan_email(days_back)
        print(json.dumps(result, indent=2))

    elif command == "status":
        status = skill.get_status()
        print(json.dumps(status, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()