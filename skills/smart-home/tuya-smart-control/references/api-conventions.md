# API Conventions

## API Key Prefix → Data Center Mapping

The base URL is automatically resolved from the first two characters after `sk-` in the API key:

| Prefix | Region | Base URL |
|--------|--------|----------|
| AY | China Data Center | https://openapi.tuyacn.com |
| AZ | US West Data Center | https://openapi.tuyaus.com |
| EU | Central Europe Data Center | https://openapi.tuyaeu.com |
| IN | India Data Center | https://openapi.tuyain.com |
| UE | US East Data Center | https://openapi-ueaz.tuyaus.com |
| WE | Western Europe Data Center | https://openapi-weaz.tuyaeu.com |
| SG | Singapore Data Center | https://openapi-sg.iotbing.com |

How to obtain an API key:
- China Mainland users: Get from https://tuyasmart.com/
- International users: Get from https://tuya.ai/
- Different regions use different API service domains, which must match your account registration region

## Request Format

All APIs use the configured Base URL concatenated with the path. Examples:

```
GET {base_url}/v1.0/end-user/homes/all
POST {base_url}/v1.0/end-user/devices/{device_id}/shadow/properties/issue
```

Authentication is via the `Authorization: Bearer {Api-key}` header (handled automatically by the Python SDK).

## Response Format

APIs return a unified structure:

**Success response**:
```json
{
  "success": true,
  "t": 1710234567890,
  "result": { ... }
}
```

**Error response**:
```json
{
  "success": false,
  "code": 1108,
  "msg": "uri path invalid"
}
```

- When `success` is `true`, the result is in the `result` field
- When `success` is `false`, error details are in the `code` and `msg` fields
- The Python SDK automatically checks `success` and raises `TuyaAPIError` on failure
- HTTP 429 and transient 5xx responses are retried automatically with backoff
