---
name: livescore-football-scores
description: >
  Track live football scores, fixtures, and match details via LiveScore's public
  unauthenticated JSON API. Returns structured match data with scores, teams,
  competition, phases, and optional deep detail (goals, incidents, stats).
  No browser needed — direct API fetch.
category: browser
version: 1.0.0
---

# LiveScore Football Scores

## Target
- URL: https://prod-cdn-mev-api.livescore.com/v1/api/app/live/soccer/{tzOffset}
- Method: Fetch (curl/web_extract direct API call)
- Bot detection: None — public unauthenticated API

## Capabilities
- List all live matches globally (across all leagues, 24/7)
- Filter by country/league
- Per-match deep detail (scoreboard, incidents, timeline)
- Phase detection: 1st half, HT, 2nd half, FT, Extra time, Penalties
- No side effects: read-only, no odds/betting interaction

## Anti-Detection
- Bot detection level: None (public API)
- CAPTCHA risk: None
- Rate limit: Generous (no documented cap; use reasonable intervals)

## API Endpoints

### Live Matches List
```
GET https://prod-cdn-mev-api.livescore.com/v1/api/app/live/soccer/0?countryCode=GB&locale=en
```
- `tzOffsetHours`: Timezone offset (`0` for UTC, `+1` for UK winter/BST)
- `countryCode`: `GB` (UK), `US`, `DE`, etc.
- `locale`: `en` (English)

### Per-Match Scoreboard
```
GET https://prod-cdn-public-api.livescore.com/v1/api/app/scoreboard/soccer/{EID}?locale=en
```
- `EID`: Match ID from live list
- Returns: venue, last-updated, incidents summary

### Per-Match Incidents
```
GET https://prod-cdn-public-api.livescore.com/v1/api/app/incidents/soccer/{EID}?locale=en
```
- Returns: full timeline of goals, cards, subs per team

## Workflow

### Step 1: Fetch Live Matches
```bash
curl -s "https://prod-cdn-mev-api.livescore.com/v1/api/app/live/soccer/0?countryCode=GB&locale=en"
```

### Step 2: Parse Response
```json
{
  "Ts": 1779305661,
  "Stages": [
    {
      "Sid": 123,
      "Snm": "Premier League",
      "Cnm": "England",
      "Scd": "premier-league",
      "Ccd": "england",
      "CompN": "Premier League",
      "Events": [
        {
          "Eid": 456789,
          "T1": [{"ID": 1, "Nm": "Arsenal", "Abr": "ARS"}],
          "T2": [{"ID": 2, "Nm": "Chelsea", "Abr": "CHE"}],
          "Sc": [2, 1],
          "Tr1": [2, 1],
          "Trh1": [1, 0],
          "Eps": "65'",
          "Esid": 1,
          "Esd": "20260603150000"
        }
      ]
    }
  ]
}
```

### Step 3: Format Output
```json
{
  "snapshot_time": "2026-06-03T15:34:21Z",
  "live_matches": 12,
  "competitions": [
    {
      "name": "Premier League",
      "country": "England",
      "matches": [
        {
          "id": 456789,
          "home_team": "Arsenal",
          "away_team": "Chelsea",
          "home_score": 2,
          "away_score": 1,
          "half_time": {"home": 1, "away": 0},
          "elapsed": "65'",
          "phase": "2nd_half",
          "kickoff_utc": "2026-06-03T15:00:00Z",
          "url": "https://www.livescore.com/en/football/england/premier-league/arsenal-vs-chelsea/456789/"
        }
      ]
    }
  ]
}
```

### Phase Mapping (Esid)
| Esid | Description |
|------|-------------|
| 0 | Scheduled |
| 1 | 1st Half |
| 2 | Half Time |
| 3 | 2nd Half |
| 4 | Extra Time 1st Half |
| 5 | Extra Time Half Time |
| 6 | Extra Time 2nd Half |
| 7 | Penalties |
| 8 | Full Time |
| 11 | Postponed |
| 12 | Abandoned |

## Integration Points

- **MatchdayMaestro (P5)**: Live scores → match data → social posts
- **Football content engine**: Match results → Twitter/LinkedIn posts
- **Cron matchday updates**: Every 5 min during match windows → Telegram/Discord
- **CoachOS**: Live results for grassroots comparisons

## Pitfalls

- `Tr1/Tr2` = regulation score (before ET/Pens)
- `Tr1OR/Tr2OR` = extra time score (if applicable)
- `Sc` = current score (may include extra time/penalties)
- Some matches may have `Epr=0` (not yet live) in live endpoint
- API returns ALL live matches globally filtered by countryCode
- Deep detail endpoints may rate-limit faster than the live list
- Incident types (IT) documented as part of browse.sh skill if needed

## Verification

1. Fetch live matches during a Premier League matchday
2. Verify scores match actual games
3. Test per-match scoreboard endpoint
4. Test per-match incidents endpoint
5. Test during non-match hours (should return empty or scheduled fixtures)
