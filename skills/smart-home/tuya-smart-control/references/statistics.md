# Data Statistics

## 1. Query Hourly Statistics Configuration

Query the hourly statistics capabilities configured for all devices under the current user. Use this to verify whether a device supports a specific type of data aggregation.

**Request**

```
GET /v1.0/end-user/statistics/hour/config
```

**Request Parameters**: None (user identity is automatically obtained from the login context)

**Response**

```json
{
  "success": true,
  "result": [
    {
      "dev_id": "0620068884f3eb414579",
      "dp_id": 17,
      "dp_code": "ele_usage",
      "statistic_type": "SUM",
      "interval": "hour"
    }
  ]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| dev_id | String | Device ID |
| dp_id | Integer | Data point ID |
| dp_code | String | Data point code |
| statistic_type | String | Statistic type: SUM (sum), COUNT (count), MAX (maximum), MIN (minimum) |
| interval | String | Statistics interval, fixed as "hour" |

---

## 2. Query Hourly Statistics Values

Get hourly statistics data for a specified device within a specified time range.

**Request**

```
GET /v1.0/end-user/statistics/hour/data
```

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| dev_id | String | Yes | Device ID |
| dp_code | String | Yes | Data point code (e.g. `ele_usage`) |
| statistic_type | String | Yes | Statistic type (e.g. `SUM`) |
| start_time | String | Yes | Start time, format: `yyyyMMddHH` (e.g. `2024010110`) |
| end_time | String | Yes | End time, format: `yyyyMMddHH` (e.g. `2024010123`) |

> The time range from start_time to end_time cannot exceed 24 hours, and `end_time` must be later than or equal to `start_time`.

**Request Example**

```
GET /v1.0/end-user/statistics/hour/data?dev_id=0620068884f3eb414579&dp_code=ele_usage&statistic_type=SUM&start_time=2024010110&end_time=2024010123
```

**Response**

```json
{
  "success": true,
  "result": [
    {"2024010110": "123.45"},
    {"2024010111": "234.56"},
    {"2024010112": "345.67"}
  ]
}
```

**Response Description**

The return value is an array where each element is a key-value pair:
- **key**: Time in `yyyyMMddHH` format
- **value**: Statistics value, String type

### Usage Workflow

1. First call the "Statistics Configuration Query" to confirm which statistics items are available for the device (dp_code + statistic_type)
2. Then use the dp_code and statistic_type from the configuration to call this API for the actual data
3. If you need statistics spanning more than 24 hours, make multiple requests and aggregate the results yourself
4. For CLI usage, `start_time` and `end_time` are pre-validated locally before sending the request:
   - Must follow `yyyyMMddHH`
   - Must satisfy `end_time >= start_time`
   - Must not exceed 24 hours
