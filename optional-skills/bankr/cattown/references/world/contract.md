# Cat Town World State — GameData contract

All live world state (season, time of day, weather, weekend flag, current world event code) lives on one onchain contract. Read-only from an agent's perspective; no admin surface to navigate.

## Address

**Base mainnet (chain id 8453):** `0x298c0d412b95c8fc9a23FEA1E4d07A69CA3E7C34`

## Primary read — `getGameState()`

Fetches everything in one call. Use this for any "what's happening in Cat Town right now?" query — one RPC call, five fields.

```solidity
function getGameState() external view
    returns (
        uint8  season,       // 0=Spring, 1=Summer, 2=Autumn, 3=Winter
        string timeOfDay,    // "Morning" | "Daytime" | "Evening" | "Nighttime"
        bool   isWeekend,    // true on Sat/Sun UTC
        uint8  worldEvent,   // event code — event-details API is out of scope for this revision
        uint8  weather       // 0=None, 1=Sun, 2=Rain, 3=Wind, 4=Storm, 5=Snow, 6=Heatwave
    );
```

Selector: `0xb7d0628b`. Full raw calldata: `0xb7d0628b`.

Sample decoded response (Base mainnet, real): `(0, "Morning", false, 0, 1)` → Spring, morning, not weekend, no active event, sunny.

## Individual reads

Use these only if you need a single field and want to avoid decoding the tuple.

| Function                  | Selector     | Returns                      |
|---------------------------|--------------|------------------------------|
| `getCurrentSeason()`      | `0xd3b30b75` | `uint8` — Season enum        |
| `getCurrentTimeOfDay()`   | `0x22d4414d` | `string` — see values below  |
| `getCurrentWeather()`     | `0x4a9919dc` | `uint8` — Weather enum       |
| `isWeekend()`             | `0xb8963681` | `bool`                       |

Prefer `getGameState()` when you'd otherwise call more than one of these in the same request.

## Enums

### Season (`uint8`)

| Value | Name    |
|-------|---------|
| 0     | Spring  |
| 1     | Summer  |
| 2     | Autumn  |
| 3     | Winter  |

### TimeOfDay (`string`)

One of `"Morning"`, `"Daytime"`, `"Evening"`, `"Nighttime"`. Case-sensitive on the contract side — normalize to lowercase if comparing user intent.

### Weather (`uint8`)

| Value | Name     |
|-------|----------|
| 0     | None     |
| 1     | Sun      |
| 2     | Rain     |
| 3     | Wind     |
| 4     | Storm    |
| 5     | Snow     |
| 6     | Heatwave |

## Historical lookups (optional)

If you need to explain yesterday's weather or project forward:

- `getSeasonForDate(uint256 unixSeconds)` → `uint8` season
- `getWeatherForDate(uint256 unixSeconds)` → `uint8` weather

Both accept any unix timestamp and use the same enums.

## Typical agent routing

| User intent                             | Call                                    |
|-----------------------------------------|-----------------------------------------|
| "What's happening in Cat Town?"         | `getGameState()` → narrate all 5 fields |
| "What's the weather right now?"         | `getGameState()` or `getCurrentWeather()` → map enum |
| "Is it night?"                          | `getCurrentTimeOfDay()` → compare string|
| "Is the fishing competition on?"        | `isWeekend()` (competition runs Sat–Sun)|
| "What was the weather last Tuesday?"    | `getWeatherForDate(unixSeconds)`        |
