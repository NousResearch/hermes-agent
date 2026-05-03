# Fishing drop tables — world state → items

Cat Town's public item catalog (`GET https://api.cat.town/v2/items/master`) lists every active item with optional drop conditions keyed to season, time of day, weather, and seasonal events. This doc covers the **fishing subset** — mirroring the frontend's live filter logic so agents can answer:

- "What fish can I catch in the rain?"
- "What drops are exclusive to Winter evenings?"
- "What's special about Storm weather?"

## Endpoint

```
GET https://api.cat.town/v2/items/master?limit=1000
```

- Public, no auth. Confirmed — plain `fetch` with no Authorization header or credentials option.
- ~430 active items at the last check. One request gets the full catalog.
- Frontend caches indefinitely (`staleTime: Infinity`). Agents should cache for ≥1 hour — items change rarely.

## Response shape (fields relevant for fishing filters)

```ts
{
  items: Array<{
    id: string
    name: string                 // "Pristine Snowflake"
    itemType: string             // "Fish" | "Treasure" | "Cosmetic" | "Collectible" | ...
    rarity: string               // "Common" | "Uncommon" | "Rare" | "Epic" | "Legendary" | ...
    source: string               // "Fishing" | "Incantation" | "Gacha" | ...
    isActive: boolean            // always check — skip if false
    dropConditions: null | {
      events?: string[]          // "Halloween" | "Christmas" | "Easter" | "Valentines Day"
      seasons?: string[]         // "Spring" | "Summer" | "Autumn" | "Winter"
      timesOfDay?: string[]      // "Morning" | "Afternoon" | "Evening" | "Night"
      weathers?: string[]        // "Sun" | "Rain" | "Wind" | "Storm" | "Snow" | "Heatwave"
    }
    imageUrl: string
    flavorText: string | null
    acquisitionMethod: string | null
  }>
  pagination: { limit: number; offset: number; total: number }
}
```

## Fishing filter — the frontend's exact logic

Ported verbatim from the live game's `utils/helpers/fishingHelpers.tsx` (`getFishingItemsFor{Weather,Season,TimeOfDay}`):

```
filter_fishing_items(axis, axis_value, current_event):
  for each item in catalog:
    1. require item.isActive == true
    2. require item.source == "Fishing"
    3. require item.itemType in {"Fish", "Treasure"}     // fishing catches only
    4. require axis_value in item.dropConditions[axis]   // items *exclusive* to this axis value
    5. event gate (see below) — skip if is_event_exclusive()
  sort by: rarity DESC (Legendary → Common), then name ASC
```

### Event-exclusivity gate

```
is_event_exclusive(item, current_event):
  events = item.dropConditions.events
  if events is empty:                     return false        // no event condition → always pass
  if current_event is None:               return true         // requires an event but none active
  return current_event not in events                          // active event doesn't match
```

In plain terms: Halloween-tagged items are invisible outside Halloween. Christmas items invisible outside Christmas. Etc.

### The filter returns *exclusive* items, not all available ones

`getFishingItemsForWeather("Snow")` returns only items with `"Snow"` in their `weathers` array — 3 items as of last check. It does **not** return the hundreds of always-available items without any weather condition. This matches the game UI's "special drops this weather" panel.

If an agent is answering "what's special about Storm weather?" — use this filter verbatim. If an agent is answering "what can I possibly catch right now?" — you'd want to union weather-exclusive drops with season-exclusive drops with unconstrained drops, and that's a different question the frontend doesn't expose directly.

## Recipe: "what can I catch in this weather?"

1. Read world state from GameData (see [../world/contract.md](../world/contract.md)):
   ```
   GameData.getGameState() → (season, timeOfDay, isWeekend, worldEvent, weather)
   ```
2. Normalize enum values for the item API (weather and season strings match after lowercasing; `worldEvent` uint8 → event name or `None`):

   | Contract `timeOfDay` | Item API `timesOfDay` |
   |----------------------|-----------------------|
   | `Morning`            | `Morning`             |
   | `Daytime`            | `Afternoon`           |
   | `Evening`            | `Evening`             |
   | `Nighttime`          | `Night`               |

3. Fetch the catalog once (cacheable):
   ```
   GET https://api.cat.town/v2/items/master?limit=1000
   ```
4. Apply the filter above for the axis the user asked about.

## Live weather → exclusive fishing drops (production sweep)

| Weather  | Exclusive fishing drops                                                                                              |
|----------|----------------------------------------------------------------------------------------------------------------------|
| Sun      | Solar Pearl (Uncommon Treasure), Sun-Kissed Catfish (Epic Fish), Amberfin Catfish (Epic Fish)                        |
| Rain     | Fancy Duck (Rare Treasure), and shared Rain/Storm items below                                                        |
| Storm    | Misty Duck (Rare), Lovely Duck (Rare), King Snapper (Rare Fish), **Elusive Marlin (Legendary Fish)**                 |
| Snow     | Frozen Tusk (Epic Treasure), Snow Globe (Rare Treasure), Pristine Snowflake (Uncommon Treasure)                      |
| Wind     | Driftwood, Pirate Doubloon, Lost Compass (treasures)                                                                 |
| Heatwave | Radiant Catfish, Gilded Sundial, Amberfin Catfish, Solar Pearl                                                       |

**Weather rotates on the scale of minutes to hours — it's the highest-frequency axis.** Weather-exclusive catches are the highest-value thing to surface to a user deciding when to fish. Season rotates much slower; time-of-day in between.

## Axis coverage at a glance (last sweep)

- Items with any `weathers` condition: 21
- Items with any `seasons` condition: 274 (Spring 94, Winter 96, Autumn 68, Summer 16)
- Items with any `timesOfDay` condition: 34
- Items with any `events` condition: 46 (Halloween dominates)

Total active items in catalog: 429.

---

## Response pattern — "what can I catch today?"

The axis-exclusive filter above returns ~3 items for weather, ~8 for a time of day. Used alone for the broad question "what can I catch today?", that feels like an incomplete answer. Structure the reply in two tiers.

### Tier 1 — special drops (lead with these)

Items exclusive to the current weather or time of day (using the fishing filter above, applied per axis). These rotate fastest and are the most interesting.

### Tier 2 — standard drops (offer as a follow-up)

> **Terminology note.** "Standard" here means *not rotating on weather/time* — items in the pool that are always eligible today (subject only to season + event). It does **not** refer to the `Common` rarity tier. A standard drop can be any rarity from Common to Legendary (e.g. Alligator Gar, the Spring Legendary, is a "standard" drop in this sense).


Everything else catchable today: items with **no** weather and **no** time-of-day restrictions that still pass the season + event gates. These are the "this season" and "always-available" fish and treasures.

Filter:

```
standard_drops(current_season, current_event):
  for each item in catalog:
    require item.isActive == true
    require item.source == "Fishing"
    require item.itemType in {"Fish", "Treasure"}
    require item.dropConditions.weathers is empty or absent
    require item.dropConditions.timesOfDay is empty or absent
    if item.dropConditions.seasons is non-empty:
      require current_season in item.dropConditions.seasons
    if item.dropConditions.events is non-empty:
      require current_event in item.dropConditions.events
    include
```

### Per-season baseline (no event active)

From a production sweep, each season has exactly **26 standard drops** available — steady across the year. The seasonal-Legendary rotates:

| Season  | Common-drop count | Seasonal Legendary   |
|---------|-------------------|----------------------|
| Spring  | 26                | Alligator Gar        |
| Summer  | 26                | Muskellunge          |
| Autumn  | 26                | Freshwater Stingray  |
| Winter  | 26                | Sturgeon             |

Same Epic/Rare backbone across seasons: Diamond, Jade Figurine, Message in a Bottle, Bronze Goblet, Catfish, and ~20 others.

### Offer phrasing

After listing the special drops, end with the common-drop count and an invitation to continue. Example copy (use the real count):

> "You can also catch **~26 other standard drops** today, led by **{seasonal-Legendary}**, **Diamond** ($100 treasure), and Catfish. Want me to list the rest by rarity?"

If the user accepts, sort by the big-ticket order (see below) and list the top 10 by default — the full list runs into hundreds if events are active.

### Big-ticket sort — always surface the high-value items first

Within any list of drops, open with the most interesting items. Sort order:

1. **Rarity DESC** — Legendary → Epic → Rare → Uncommon → Common. Primary signal, especially for fish (weight data isn't in this API).
2. **`sellValue` DESC** — `sellValue` is in **cents USD** (not KIBBLE). Tiebreaker within a rarity. Concrete values from a live sweep:
    - 25,000¢ ($250): Solar Ring, Dawnbreak Ring, Moonlight Ring, Twilight Ring (Legendary, time-of-day exclusive)
    - 11,000¢ ($110): Evil Book (Legendary, Halloween + Autumn + Evening/Night)
    - 10,000¢ ($100): Frozen Tusk (Epic, Snow), Diamond (Epic, no conditions)
    - 5,500¢: Gilded Sundial (Epic, Heatwave)
    - 5,000¢: Lost Compass (Epic, Wind)
    - 4,000¢: Jade Figurine (Epic, no conditions)
    - Lower tiers: Snow Globe (3,000¢), Fancy Duck (3,000¢), Gold Band (2,500¢), etc.
3. **Name ASC** as final tiebreaker.

Note: fish typically have low or zero `sellValue` in the API — rarity carries the signal for them. Treasure sellValue is the more informative tiebreaker because treasures have real prices.

### Fish weight data — not in this API

Per-species weight ranges (e.g. "Bluegill: 0.5kg – 1.2kg") are **not** returned by `/v2/items/master`. If a user asks about weights or the heaviest fish by species, point them at Cat Town's public docs (unauthenticated):

- https://docs.cat.town/items/fishing/fish — per-fish weight ranges + conditions
- https://docs.cat.town/items/fishing/treasures — per-treasure detail pages

For agent-generated answers without a human follow-up, lean on rarity + `sellValue`. For "what's the biggest {species}" or similar weight-specific asks, route the user to those pages.

### Why two tiers

The exclusive filter alone returns 3–4 items for weather. A user expecting "here's what's catchable today" will think the answer is cut off. Leading with the rotational drops (what makes *this moment* different) and offering the broader catalog on request is both more natural and more complete.
