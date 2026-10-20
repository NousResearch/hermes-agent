---
name: broadlinkac
description: Control air conditioners via Broadlink RM devices — multi-brand IR control, weather monitoring, typhoon alerts, scheduled automation. Clone, pip install, import — zero-config Agent API.
---

# BroadlinkAC — AI Agent Smart AC Controller

Cross-platform AC control library for Broadlink RM series IR blasters. **Zero GUI dependency** — designed for AI agents to clone, install, and control air conditioners programmatically.

## Quick Start (Agent)

```bash
git clone https://github.com/oywq00008-cell/BroadlinkAC-For-AI-Agent.git
cd BroadlinkAC-For-AI-Agent
pip install -r requirements-core.txt
```

```python
from broadlinkac_core import init, send_ac, fetch_weather, fetch_weather_alerts

# One-time setup — all config persisted to ~/.ac_controller/config.json
init(
    api_key="your_qweather_key",
    qw_host="https://xxx.re.qweatherapi.com",
    location={"lat": 22.54, "lon": 114.05, "name": "Shenzhen"},
    brand="Gree"
)

# Control AC
send_ac("on", "cool", 26, "auto")   # Turn on, cool 26°C, auto fan
send_ac("off", "cool", 26, "auto")  # Turn off

# Get weather
weather = fetch_weather()            # Real-time temp, humidity, feels-like
alerts = fetch_weather_alerts()      # Local disaster warnings (heat/rain/typhoon)
```

## API Reference

### Setup
| Function | Description |
|----------|-------------|
| `init(api_key=None, qw_host=None, location=None, brand=None)` | Initialize config + start background scheduler. All params optional — subsequent calls read from persistent config. |

### AC Control
| Function | Description |
|----------|-------------|
| `send_ac(power, mode, temp, fan)` | Send IR command. `power`: `"on"`/`"off"`. `mode`: `"cool"`/`"heat"`/`"dry"`/`"fan"`/`"auto"`. `temp`: 16-30. `fan`: `"auto"`/`"1"`/`"2"`/`"3"` |
| `decide_ac(outdoor_temp)` | Run temperature rules → returns `(target_temp, mode)` |

### Weather & Alerts
| Function | Description |
|----------|-------------|
| `fetch_weather()` | Current weather (temp, humidity, feels-like, wind) via QWeather API |
| `fetch_weather_alerts()` | Local weather warnings — returns list of `{headline, severity, description, senderName, effectiveTime, expireTime}` |
| `city_lookup(query)` | OpenStreetMap city search → coordinates |

### Typhoon
| Function | Description |
|----------|-------------|
| `fetch_typhoons()` | Active NW Pacific typhoons from China NMC |
| `fetch_typhoon_detail(typhoon_id)` | Detailed track + forecast |

## Supported AC Brands

**hvac_ir (5 brands):** Gree, Midea, Hisense, Daikin, Mitsubishi

**Custom protocols (3 brands):** Haier, AUX, Panasonic (ported from IRremoteESP8266 C++)

Select in Settings or pass `brand=` to `init()`. Device auto-discovered on LAN via Broadlink UDP.

## Key Design

- `import broadlinkac_core` has **zero side effects** — no network I/O, no filesystem reads, no threads started
- `init()` is idempotent — safe to call multiple times
- Config auto-persisted to `~/.ac_controller/config.json`
- Runs on any device with Python 3.9+ (macOS/Windows/Linux/Raspberry Pi/NAS)

## Common Agent Tasks

### "Turn on the AC to 26°C cooling"
```python
from broadlinkac_core import init, send_ac
init()
send_ac("on", "cool", 26, "auto")
```

### "What's the temperature outside?"
```python
from broadlinkac_core import init, fetch_weather
init()
w = fetch_weather()
# Returns {"temp": "31", "text": "晴", "feelsLike": "33", "humidity": "65", ...}
```

### "Are there any weather warnings?"
```python
from broadlinkac_core import init, fetch_weather_alerts
init()
alerts = fetch_weather_alerts()
for a in alerts:
    print(f"[{a['severity']}] {a['headline']}")
```

### "Turn off the AC at 10pm"
```python
# Set off_time in config, then the built-in scheduler handles it
import broadlinkac_core.config as _cfg
from broadlinkac_core import init
init()
_cfg.config["off_time"] = "22:00"
_cfg.config["off_enabled"] = True
_cfg.save_config(_cfg.config)
```
