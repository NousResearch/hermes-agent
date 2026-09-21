---
name: transfermarkt-player-data
description: >
  Extract football player data, market values, transfer history, and career stats
  from Transfermarkt for CoachOS, MatchdayMaestro, Player Portfolio Builder,
  and football content engine.
category: browser
version: 1.0.0
---

# Transfermarkt Player Data

## Target
- URL: https://www.transfermarkt.co.uk/<player-slug>/profil/spieler/<player-id>
- Method: Browser (browser_navigate + browser_snapshot)
- Bot detection: Medium (Cloudflare)

## Capabilities
- Player profile: name, age, position, club, nationality
- Market value history
- Transfer history (from, to, fee, date)
- Career stats per season/competition
- No side effects: read-only, no images (copyright)

## Anti-Detection
- Bot detection level: Medium (Cloudflare)
- CAPTCHA risk: Possible
- Rate limit: ~30 requests/hour
- Strategy: 5-10s delays, rotate User-Agent quarterly
- On block: retry once after 15s delay

## Input
- Player name (free-text, will resolve to slug)
- Player ID (if known, direct URL)
- Club (optional filter)

## Workflow

### Step 1: Search for Player
```
https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query=<player_name>
```
Navigate to search, find the correct player result. Transfermarkt search returns a list of matching players/clubs.

### Step 2: Navigate to Player Profile
```
https://www.transfermarkt.co.uk/<slug>/profil/spieler/<id>
```
Profile page contains all core player data.

### Step 3: Extract Player Data
From profile page, extract:
- `name` — full name
- `age` — current age
- `position` — primary position(s)
- `club` — current club
- `nationality` — country/national team
- `market_value` — current estimated value
- `contract_end` — contract expiry date
- `joined` — date joined current club
- `shirt_number` — squad number

### Step 4: Extract Market Value History
Navigate to market value tab or extract from profile page data:
- Value history with dates and amounts
- Chart data if available

### Step 5: Extract Transfer History
Navigate to transfer history or extract from profile:
- `from_club`, `to_club`, `fee`, `date`, `season`
- Loan vs permanent flag

### Step 6: Return
```json
{
  "success": true,
  "player": {
    "id": 12345,
    "name": "Bukayo Saka",
    "age": 24,
    "position": ["Right Winger", "Left Winger"],
    "club": "Arsenal FC",
    "club_id": 11,
    "nationality": "England",
    "market_value": "€120.00m",
    "contract_end": "2027-06-30",
    "joined": "2020-07-01",
    "shirt_number": 7
  },
  "market_value_history": [
    {"date": "2025-12-01", "value": "€150.00m"},
    {"date": "2025-06-01", "value": "€140.00m"},
    {"date": "2024-12-01", "value": "€120.00m"}
  ],
  "transfer_history": [
    {
      "season": "20/21",
      "date": "2020-07-01",
      "from_club": "Arsenal U23",
      "to_club": "Arsenal FC",
      "fee": "-",
      "type": "permanent"
    }
  ],
  "url": "https://www.transfermarkt.co.uk/bukayo-saka/profil/spieler/12345"
}
```

## Use Cases

- **CoachOS (P4)**: Player profile data for coaching plans
- **MatchdayMaestro (P5)**: Player trivia, transfer insights
- **Player Portfolio Builder (P6)**: Market value benchmarks for grassroots players
- **Football content engine**: Transfer news → social posts
- **Kick-tionary**: Player profiles for educational content

## Integration Points

- **Content pipeline**: Player data → drafted social posts (CeeCee)
- **Cron**: Weekly top 5 league market value updates
- **Kanban**: Noteworthy transfers → content task

## Pitfalls

- Cloudflare may block automated sessions
- Market values are Transfermarkt estimates, not actual fees
- Some player pages are sparse (lower leagues)
- Images are copyrighted — DO NOT download or store
- URL structure uses German-style slugs (lowercase, hyphens)
- Search results may include multiple players with same name — verify by age/club
- Contract dates are often approximate ("30.06.2027")
- Transfer fees marked with "?" are estimates/rumours — flag in output

## Verification

1. Search for a known Premier League player
2. Verify market value is reasonable
3. Test with lower-league player (sparse data)
4. Test with player with no transfer history (academy product)
5. Verify URL composition with player ID
