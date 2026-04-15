---
name: ocas-taste-implementation
description: >
  Complete implementation setup for the Taste skill — email/calendar scanning,
  Spotify sync, Google API scope configuration, MCP server setup, and recurring
  scan automation. Covers critical patterns for scope matching, credential
  configuration, and consumption signal extraction.
metadata:
  author: Indigo Karasu
  email: mx.indigo.karasu@gmail.com
  version: "1.2.0"
  hermes:
    tags: [implementation, setup, integration, google-api, mcp]
    category: devops
---

# Taste Skill Implementation

Complete implementation setup for the Taste OCAS skill. The skill specification exists but requires actual implementation code for scanning email, calendar, and Spotify to build the taste model.

## When to use

- Setting up Taste skill from scratch
- Implementing similar OCAS skills that need external API integrations
- Debugging Google API scope mismatch errors
- Configuring MCP servers with credentials
- Setting up recurring scans with cron jobs

## Critical Implementation Patterns

### Google API Scope Matching

**Common Error**: `invalid_scope: Bad Request`

**Cause**: Requesting scopes not present in existing token

**Solution**: Check `~/.hermes/google_token.json` and use exact scopes:

```python
# Check existing token scopes
with open(token_path) as f:
    token_data = json.load(f)
    print(token_data.get('scopes', []))

# Use exact scopes from token
creds = Credentials.from_authorized_user_file(
    str(token_path),
    ['https://www.googleapis.com/auth/gmail.modify',
     'https://www.googleapis.com/auth/calendar']
)
```

**Token Refresh Pattern**:
```python
if not creds.valid:
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_path, 'w') as f:
            f.write(creds.to_json())
    else:
        print("Credentials invalid and cannot be refreshed")
        return False
```

### MCP Server Credential Configuration

**Critical Pattern**: Environment variables in `.env` don't propagate to MCP stdio commands. Must configure in `config.yaml`.

**Common Error**: `Missing required environment variable: SPOTIFY_CLIENT_ID`

**Solution**: Configure in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  spotify:
    command: npx
    args:
      - "@darrenjaws/spotify-mcp"
    env:
      SPOTIFY_CLIENT_ID: "your-client-id"
      SPOTIFY_CLIENT_SECRET: "your-client-secret"
      SPOTIFY_REDIRECT_URI: "http://localhost:8888/callback"
```

**Test Connection**:
```bash
hermes mcp test spotify
```

Expected output:
```
✓ Connected (1359ms)
✓ Tools discovered: 32
```

### Email Extraction and Deduplication Pattern

**Service-Specific Extraction**:
```python
def _extract_from_email(self, msg_data: Dict, service: str, domain: str, source_type: str):
    headers = {h['name']: h['value'] for h in msg_data['payload'].get('headers', [])}
    subject = headers.get('Subject', '')
    body = self._get_email_body(msg_data['payload'])
    
    extraction = {
        "service": service,
        "domain": domain,
        "source_type": source_type,
        "subject": subject,
        "date": email_date.isoformat(),
        "body": body[:5000],  # Truncate to avoid huge records
        "email_type": self._classify_email_type(subject, body),
        "cancelled": False
    }
    
    # Service-specific extraction
    if service == "doordash":
        extraction.update(self._extract_doordash(subject, body))
    elif service == "instacart":
        extraction.update(self._extract_instacart(subject, body))
    
    return extraction
```

**Deduplication Pattern**:
```python
def _compute_dedup_key(self, service: str, order_id: str, event_date: str, venue_name: str) -> str:
    normalized_venue = self._normalize_venue_name(venue_name)
    return f"{service}:{order_id}:{event_date}:{normalized_venue}"

def _normalize_venue_name(self, name: str) -> str:
    name = name.lower()
    name = re.sub(r'\bthe\b', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[^\w\s]', '', name)
    return name.strip()
```

**Signal Creation Pattern**:
```python
def _process_extractions(self, extractions: List[Dict]) -> tuple:
    groups = {}
    for extraction in extractions:
        dedup_key = self._compute_dedup_key(...)
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
        self._update_item_record(canonical)
    
    return signals, cancellations
```

## Prerequisites

### Google Workspace OAuth
Already configured at `~/.hermes/google_token.json` for jared.zimmerman@gmail.com

### Spotify OAuth
1. Create app at https://developer.spotify.com/dashboard
2. Set redirect URI: `http://localhost:8888/callback`
3. Get Client ID and Client Secret
4. Configure in `~/.hermes/config.yaml` under `mcp_servers.spotify.env`

## Implementation Steps

### 1. Create Virtual Environment

On Debian/Ubuntu systems with PEP 668 (externally managed Python), cannot install packages system-wide:

```bash
# Install venv package if missing
apt install -y python3.13-venv

# Create virtual environment in skill data directory
cd /root/.hermes/commons/data/ocas-taste
python3 -m venv venv
```

### 2. Install Dependencies

```bash
source venv/bin/activate
pip install spotipy google-api-python-client
```

### 3. Create Scanning Scripts

Create `scripts/` directory with three scripts:

**`scripts/taste_scan.py`** — Main entry point
**`scripts/email_scan.py`** — Gmail and Calendar scanner
**`scripts/spotify_sync.py`** — Spotify listening history sync

See script content in the skill package for full implementations.

### 4. Register Cron Jobs

**Important**: Hermes cron syntax differs from documentation. Use this format:

```bash
# Email/calendar scan (every 6 hours)
hermes cron create --name taste:scan --skill ocas-taste "0 */6 * * *" \
  "python3 /root/.hermes/webui/workspace/taste_implementation_fixed.py scan-email 7"

# Spotify sync (daily at midnight)
hermes cron create --name taste:sync-spotify --skill ocas-taste "0 0 * * *" \
  "python3 /root/.hermes/webui/workspace/spotify_sync.py recent --limit 50"
```

**Syntax**: `hermes cron create --name NAME --skill SKILL "SCHEDULE" "PROMPT"`

- Schedule is a positional argument (cron expression or relative time)
- Prompt is a positional argument (command to run)
- Use `--skill` to attach the skill for context

### 5. Verify Setup

```bash
# Check cron jobs
hermes cron list | grep taste

# Run manual test
python3 /root/.hermes/webui/workspace/taste_implementation_fixed.py status
```

Expected output:
```json
{
  "total_signals": 794,
  "total_items": 18,
  "domain_breakdown": {
    "restaurant": 396,
    "product": 297,
    "travel": 99,
    "music": 2
  }
}
```

## Data Structure

### Signals (`signals.jsonl`)
```json
{
  "signal_id": "sig-{timestamp}-{index}",
  "domain": "restaurant|product|travel|music",
  "source_type": "purchase|visit|play|stay",
  "venue_name": "Venue Name",
  "event_date": "ISO 8601",
  "strength": 0.80,
  "created_at": "ISO 8601",
  "extraction_source": "doordash|calendar|spotify"
}
```

### Items (`items.jsonl`)
```json
{
  "item_id": "item-{timestamp}",
  "venue_name": "Venue Name",
  "domain": "restaurant|product|travel|music",
  "signal_count": 5,
  "first_seen": "ISO 8601",
  "last_seen": "ISO 8601",
  "visit_dates": ["ISO 8601", ...],
  "enriched": false,
  "metadata": {}
}
```

## Common Pitfalls

### 1. Scope Mismatch

**Symptom**: `invalid_scope: Bad Request`

**Cause**: Requesting scopes not present in existing token

**Solution**: Check `~/.hermes/google_token.json` and use exact scopes

### 2. Gmail Query Construction Bug (CRITICAL)

**Symptom**: Zero results when querying Gmail, OR massive false positives (GitHub notifications, marketing emails extracted as consumption signals)

**Root Cause**: Gmail API query uses `OR` to join sender patterns AND the date filter, so `after:YYYY/MM/DD` matches ALL emails after that date regardless of sender.

**WRONG**:
```python
query_parts = [f"from:{p}" for p in sender_patterns]
query_parts.append(f"after:{date_str}")
query = " OR ".join(query_parts)
# This becomes: "from:a@doordash.com OR from:b@doordash.com OR after:2025/04/14"
# The after: is OR'd with senders, matching EVERYTHING after the date
```

**CORRECT** — group sender patterns, then AND with date:
```python
sender_query = " OR ".join([f"from:{p}" for p in sender_patterns])
query = f"({sender_query}) after:{date_str}"
# This becomes: "(from:a@doordash.com OR from:b@doordash.com) after:2025/04/14"
# Only messages from those senders AND after the date
```

**Wildcards make it worse**: Sender patterns like `*@exploretock.com` or `*@booking.com` are wildcards that Gmail interprets as partial matches, pulling in unrelated emails. Always add a post-extraction validation filter:

```python
# After extraction, filter out garbage
if extraction.get("venue_name") in (None, "Unknown", ""):
    return None  # No real venue identified

# Verify the email actually came from the expected service
expected_senders = config.get("email_sources", {}).get(service, {}).get("sender_patterns", [])
from_lower = from_addr.lower()
actual_service_email = any(
    pat.replace("*", "").lower() in from_lower
    for pat in expected_senders
    if pat
)
if not actual_service_email and expected_senders:
    return None  # Email doesn't match expected sender
```

### 3. Calendar Scan Only Checks Primary Calendar (CRITICAL)

**Symptom**: Calendar scan returns 0-2 events despite many restaurant reservations and hotel bookings in Google Calendar.

**Root Cause**: `scan_calendar_historical()` uses `calendarId='primary'`, which only scans the user's primary calendar. Venue events often live in shared calendars (e.g., "Personal" = jared.zimmerman@gmail.com, "Family" = family calendar) that share events with the primary but aren't the primary itself.

**WRONG**:
```python
events_result = self.calendar_service.events().list(
    calendarId='primary',  # <- Only scans one calendar!
    ...
).execute()
```

**CORRECT** — iterate all writable calendars:
```python
# Get all calendars (owner/writer access)
cal_list = self.calendar_service.calendarList().list().execute()
calendars = cal_list.get('items', [])
scannable_cals = [c for c in calendars if c.get('accessRole') in ('owner', 'writer')]

for cal in scannable_cals:
    cal_id = cal['id']
    cal_name = cal['summary']
    # ... scan each calendar with pagination
    events_result = self.calendar_service.events().list(
        calendarId=cal_id,  # <- Each writable calendar
        ...
    ).execute()
```

**Impact**: Scanning only `primary` found 2 events (the agent's own calendar). Scanning all writable calendars found **130 venue extractions** across 980 events in Personal and Family calendars.

### 3b. Cross-Calendar Deduplication

Events often appear in multiple calendars (e.g., same reservation in both Personal and Family). Without cross-calendar dedup, duplicate signals get created.

**Solution**: Use normalized venue name + event date as dedup key across ALL calendars:
```python
seen_dedup_keys = set()

def compute_dedup_key(service, event_id, event_date, venue_name):
    normalized = normalize_venue_name(venue_name)
    date_part = event_date[:10]
    return f"{service}:{normalized}:{date_part}"

# Before creating signal:
if dedup_key in seen_dedup_keys:
    continue  # Skip duplicate from another calendar
seen_dedup_keys.add(dedup_key)
```

### 3c. Venue Name Cleanup from Calendar Summaries

Calendar event summaries contain prefixes and annotations that break dedup:
- `"Reservation at Hillstone - Embarcadero"` → `"Hillstone - Embarcadero"`
- `"A16 - San Francisco"` → `"A16"`
- `"José for 2"` → `"José"`

**Cleanup pattern**:
```python
venue_name = re.sub(r'^Reservation\s+at\s+', '', venue_name, flags=re.IGNORECASE)
venue_name = re.sub(r'\s*[-–]\s*(San Francisco|Daly City|Providence|Dallas|Oakland|SF)\s*$', '', venue_name, flags=re.IGNORECASE)
```

### 3d. Broadened Venue Detection Heuristics

The original `_is_venue_event()` was too strict and missed many real dining/travel events. Key additions:

**Exclude**: medical appointments (`doctor`, `dr.`, `one medical`, `bodyspec`, `telehealth`), video calls (`zoom.us`, `teams.microsoft`, `google meet`), and generic meetings (`standup`, `1:1`, `sync`, `interview`, `therapy`, `dentist`).

**Include**: meal keywords with specific venue indicators in location (address-like text: `st`, `ave`, `blvd`, `drive`, `road`), named hotel brands (`fairmont`, `marriott`, `hilton`, `hyatt`), and event types like `omakase`, `chef`, `tasting`, `winery`, `brewery`.

### 3. Multi-Profile Google Token Discovery

**Symptom**: `invalid_grant: Bad Request` when refreshing OAuth token

**Root Cause**: The user has TWO Google accounts with separate OAuth tokens:
- `~/.hermes/google_token.json` → jared.zimmerman@gmail.com (primary user account, consumption emails live here)
- `~/.hermes-indigo/google_token.json` → mx.indigo.karasu@gmail.com (agent account, no consumption emails, token stays fresh)

**Resolution**:
1. Try refreshing the user account token first
2. If `invalid_grant`, the user must do an interactive browser re-auth:
   ```bash
   PYTHONPATH=/root/.hermes/hermes-agent python3 /root/.hermes/skills/productivity/google-workspace/scripts/setup.py --auth-url
   # User visits URL in browser, authorizes, copies code= from redirect
   PYTHONPATH=/root/.hermes/hermes-agent python3 /root/.hermes/skills/productivity/google-workspace/scripts/setup.py --auth-code {CODE}
   ```
3. As a **fallback** for calendar-only access, the indigo account shares the user's "Personal" calendar (`jared.zimmerman@gmail.com`) and is accessible for calendar event extraction.
4. **Never use the agent account for email scanning** — it has no consumption emails.

**Token path selection pattern**:
```python
token_paths = [
    Path.home() / ".hermes" / "google_token.json",           # Primary (user)
    Path.home() / ".hermes-indigo" / "google_token.json",     # Agent (calendar fallback)
]
token_path = None
for tp in token_paths:
    if tp.exists():
        creds = Credentials.from_authorized_user_file(str(tp))
        if creds.valid or (creds.expired and creds.refresh_token):
            try:
                creds.refresh(Request())
                token_path = tp
                break
            except:
                continue
if token_path is None:
    # All tokens dead, need interactive re-auth
    return False
```

### 4. Garbage Signal Cleanup

**Symptom**: Signals with `venue_name: "Unknown"` polluting the taste model

**Cause**: Broken Gmail query (see pitfall #2) or wildcard sender patterns matching unrelated emails.

**Cleanup script**:
```python
# Remove signals with no real venue
signals = [json.loads(l) for l in open(signals_file) if l.strip()]
clean = [s for s in signals if s.get('venue_name') not in (None, 'Unknown', '')]

# Deduplicate signals (same venue + date + source)
seen = set()
unique = []
for s in clean:
    key = (s.get('venue_name',''), s.get('event_date','')[:10],
           s.get('extraction_source',''), s.get('domain',''))
    if key not in seen:
        seen.add(key)
        unique.append(s)

with open(signals_file, 'w') as f:
    for s in unique:
        f.write(json.dumps(s) + '\n')
```

### 5. MCP Environment Variables Not Found

**Symptom**: `Missing required environment variable: SPOTIFY_CLIENT_ID`

**Cause**: `.env` variables don't propagate to MCP stdio commands

**Solution**: Configure in `config.yaml` under `mcp_servers.{server}.env`

### 6. Duplicate Cron Jobs

**Symptom**: Multiple jobs with same name

**Solution**: Remove duplicate with `hermes cron remove <job_id>`

### 7. Email Body Extraction Issues

**Symptom**: Empty or truncated email bodies

**Cause**: Not handling multipart MIME messages correctly

**Solution**: Recursively parse `parts` array and decode base64 content

## Known Limitations

### Email Extraction
Currently uses regex-based extraction. Production implementation needs LLM-based extraction for:
- Parsing order confirmations
- Extracting venue names, dates, items ordered
- Handling different email formats per service

### Calendar Extraction
Uses simple pattern matching. Production implementation needs:
- LLM-based event classification
- Better venue name extraction
- Location parsing

### Spotify OAuth
First run opens browser for OAuth authorization. After that, token is cached.

## Cron Job Management

```bash
# List cron jobs
hermes cron list

# Remove a job
hermes cron remove <job_id>

# Create a job
hermes cron create --name NAME --skill SKILL "SCHEDULE" "PROMPT"
```

## Related Skills

- `python-package-installation` — PEP 668 handling
- `google-workspace-setup` — Google OAuth setup
- `hermes-persona-setup` — Skill setup patterns

## Notes

- All scripts use the user's email account (jared.zimmerman@gmail.com), never the agent's account
- Virtual environment isolates dependencies from system Python
- Cron jobs run in isolated sessions with light context
- Checkpoint files prevent reprocessing the same data
- MCP server credentials must be in config.yaml, not .env