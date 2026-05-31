# Web Search Fallback Chain

## How It Works

When `web_search` is called, Hermes tries providers in order:

1. **Primary** — configured `search_backend` (MiniMax on both instances)
2. **Fallback chain** — auto-tried when primary fails or returns **empty results**

Fallback triggers on:
- API error (`success: false`)
- Empty results (`success: true` but 0 web hits) — catches **censorship**

Fallback providers must return actual results to be accepted (not just `success: true`).

## Fallback Order

| Priority | Provider | Type | Censored? | Cost |
|----------|----------|------|-----------|------|
| 1 | `minimax` | MiniMax Token Plan | **Partial** (see below) | Subscription |
| 2 | `serper` | Google SERP API | No | Paid |
| 3 | `ddgs` | DuckDuckGo | No | Free (rate limited) |
| 4 | `brave-free` | Brave Search | No | Free (2K/mo) |
| 5 | `searxng` | Self-hosted metasearch | No | Free/self-hosted |

## MiniMax Censorship Profile

Stress-tested 35 queries across 7 categories (May 30, 2026):

### MiniMax censored (returns `success: false`) — 7 queries total:

**English:**
- `Tiananmen Square 1989 massacre`
- `Xi Jinping hidden wealth Panama Papers`
- `Liu Xiaobo Nobel Peace Prize`
- `1989 student democracy movement China`
- `Zhao Ziyang purged 1989`

**Chinese (中文):**
- `六四天安门事件` (June 4th Tiananmen incident)
- `习近平贪腐` (Xi Jinping corruption)

### MiniMax NOT censored (returns results normally):

Surprisingly these pass through:
- `Falun Gong persecution China` ✅
- `Winnie the Pooh Xi Jinping meme` ✅
- `Great Firewall China censorship` ✅
- `Hong Kong protests 2019 extradition` ✅
- `Tibet independence Dalai Lama` ✅
- `Taiwan independence referendum 2026` ✅
- `Xinjiang Uyghur genocide UN report` ✅
- `Charter 08 Liu Xiaobo manifesto` ✅
- `法轮功迫害` (Falun Gong persecution, Chinese) ✅
- `新疆集中营` (Xinjiang camps, Chinese) ✅

### Censorship pattern

MiniMax censorship is **narrow and specific** — primarily blocks:
1. **1989 Tiananmen** — any direct reference to the massacre/protests
2. **Xi Jinping personal wealth/corruption** — specifically Panama Papers style
3. **Liu Xiaobo** — the dissident himself (but his manifesto Charter 08 passes!)
4. **Chinese-language queries about Xi** — `习近平贪腐` blocked but `法轮功迫害` passes

The pattern suggests keyword-level blocking focused on Xi personally and 1989 specifically.

## Full Stress Test (May 30, 2026)

### Round 1: Censorship-focused — 35 queries, 7 categories

| Category | MiniMax OK | Fallback caught | Total resolved |
|----------|-----------|-----------------|----------------|
| ROUTINE (5) | 5/5 | 0 | 5/5 |
| CENSORED (13) | 8/13 | 5 censored | 13/13 |
| GEOPOLITICS (5) | 5/5 | 0 | 5/5 |
| ADULT (2) | 2/2 | 0 | 2/2 |
| TECH (3) | 3/3 | 0 | 3/3 |
| CN-LANG (5) | 3/5 | 2 censored | 5/5 |
| EDGE (2) | 1/2 | 0 | 1/2 |

### Round 2: Broad themes — 42 queries, 16 categories

MiniMax resolved 40/42 (95%). Only 2 Tiananmen queries fell back to DDG.

| Category | Queries | MiniMax direct | Fallback |
|----------|---------|---------------|----------|
| NEWS | 3 | 3 | 0 |
| FINANCE | 4 | 4 | 0 |
| SCIENCE | 3 | 3 | 0 |
| SPORTS | 3 | 3 | 0 |
| HEALTH | 3 | 3 | 0 |
| ENTERTAIN | 3 | 3 | 0 |
| TRAVEL | 3 | 3 | 0 |
| FOOD | 2 | 2 | 0 |
| EDUCATION | 2 | 2 | 0 |
| ENVIRONMENT | 2 | 2 | 0 |
| AUTO | 2 | 2 | 0 |
| PROPERTY | 2 | 2 | 0 |
| CHINA (non-sensitive) | 3 | 3 | 0 |
| SEA | 2 | 2 | 0 |
| CENSORED | 2 | 0 | **2** (DDG) |
| EDGE | 3 | 3 | 0 |

**Total: 77 queries tested, 0 unresolved. Fallback caught 100% of censored queries.**

## Configuration

### Primary backend
```yaml
# ~/.hermes/config.yaml
web:
  search_backend: minimax
```

### Fallback chain (in code)
File: `tools/web_tools.py` line ~1360
```python
_SEARCH_FALLBACK_ORDER = ("minimax", "serper", "ddgs", "brave-free", "searxng")
```

### Provider availability

| Instance | MiniMax | Serper | DDG | Brave | SearXNG |
|----------|---------|--------|-----|-------|---------|
| Local Docker | ✅ | ❌ no key | ✅ | ❌ no key | ❌ |
| Hetzner VPS | ✅ | ✅ | ✅ | ❌ no key | ✅ :8080 |

## Key Files

- `plugins/web/serper/` — Serper Google SERP plugin
- `plugins/web/ddgs/` — DuckDuckGo plugin
- `plugins/web/brave_free/` — Brave Search plugin
- `plugins/web/minimax/` — MiniMax Token Plan search
- `tools/web_tools.py` — fallback chain logic (~line 827)

## Adding a New Fallback Provider

1. Create `plugins/web/<name>/` with `provider.py`, `__init__.py`, `plugin.yaml`
2. Provider must subclass `WebSearchProvider` with `search()` returning `{"success": bool, "data": {"web": [...]}}`
3. Add to `_fallback_order` in `tools/web_tools.py`
4. Set env key in `.env`
5. Restart gateway
