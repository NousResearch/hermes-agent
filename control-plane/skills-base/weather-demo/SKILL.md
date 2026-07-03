---
name: weather-demo
description: "Demo skill: fetch weather for a city from an internal weather API. Declares a per-employee token."
version: 1.0.0
metadata:
  orchard:
    data_sources:
      - name: weather-api
        url: https://weather.internal/api
    secrets:
      - env: WEATHER_TOKEN
        label: "Weather API token"
        required: true
        docs_url: https://weather.internal/docs/tokens
---

# Weather (demo)

Fetch current weather for a city from the internal weather API.

## When to use
- The user asks for weather / temperature / forecast for a place.

## How
The API token is available as the environment variable `WEATHER_TOKEN`
(injected per-employee by the control-plane — never ask the user to paste it in
chat). Call the endpoint with it:

```bash
curl -s "https://weather.internal/api/current?city=<CITY>" \
  -H "Authorization: Bearer $WEATHER_TOKEN"
```

If `WEATHER_TOKEN` is unset, tell the user to run `/secret set WEATHER_TOKEN`
in Mattermost and follow the private link.
