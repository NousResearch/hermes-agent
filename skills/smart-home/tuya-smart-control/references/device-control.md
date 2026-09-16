# Device Control

## 1. Query Device Thing Model

Query the Thing Model definition of a specified device to understand which functional properties it supports.

**Request**

```
GET /v1.0/end-user/devices/{device_id}/model
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| device_id | String | Yes | Device ID |

**Response**

```json
{
  "success": true,
  "t": 1673423649773,
  "result": {
    "model": "{\"modelId\":\"000004****\",\"services\":[{\"code\":\"\",\"description\":\"\",\"name\":\"\",\"properties\":[{\"code\":\"switch_led\",\"description\":\"Light switch\",\"accessMode\":\"rw\",\"name\":\"Switch\",\"typeSpec\":{\"type\":\"bool\"},\"abilityId\":20},{\"code\":\"bright_value\",\"description\":\"Brightness adjustment\",\"accessMode\":\"rw\",\"name\":\"Brightness\",\"typeSpec\":{\"max\":1000,\"scale\":0,\"type\":\"value\",\"unit\":\"\",\"min\":10,\"step\":1},\"abilityId\":22}]}]}"
  }
}
```

> `result.model` is a JSON string that needs to be parsed again.

**Parsed model structure**

```json
{
  "modelId": "000004****",
  "services": [
    {
      "code": "",
      "name": "",
      "description": "",
      "properties": [
        {
          "abilityId": 20,
          "code": "switch_led",
          "name": "Switch",
          "description": "Light switch",
          "accessMode": "rw",
          "typeSpec": {
            "type": "bool"
          }
        },
        {
          "abilityId": 22,
          "code": "bright_value",
          "name": "Brightness",
          "description": "Brightness adjustment",
          "accessMode": "rw",
          "typeSpec": {
            "type": "value",
            "min": 10,
            "max": 1000,
            "step": 1,
            "unit": "",
            "scale": 0
          }
        }
      ]
    }
  ]
}
```

### Thing Model Structure

**service**

| Field | Type | Description |
|-------|------|-------------|
| code | String | Service identifier; empty string represents the default service |
| name | String | Service name |
| description | String | Service description |
| properties | List | Property definition list |

**property**

| Field | Type | Description |
|-------|------|-------------|
| abilityId | Integer | Ability identifier ID |
| code | String | Property code (used as the key when issuing properties) |
| name | String | Property display name |
| description | String | Property description |
| accessMode | String | Access mode: `ro` read-only, `wr` write-only, `rw` read-write |
| typeSpec | Object | Data type specification |

### typeSpec Type Definitions

**value (numeric)**

```json
{ "type": "value", "min": 0, "max": 100, "step": 1, "unit": "%", "scale": 0 }
```

- `scale`: Decimal multiplier. Actual value = input value / 10^scale. Example: if `scale` is `1` and raw value is `255`, actual value = 255 / 10^1 = 25.5

**bool (boolean)**

```json
{ "type": "bool" }
```

- Values: `true` or `false`

**enum (enumeration)**

```json
{ "type": "enum", "range": ["auto", "cold", "hot", "wind"] }
```

- Value must be within the `range` list

**string**

```json
{ "type": "string", "maxlen": 100 }
```

**Other types**: `float`, `double`, `date`, `raw`, `bitmap`, `struct`, `array` — refer to the Thing Model documentation for details.

**Error Codes**

| Code | Message | Description |
|------|---------|-------------|
| 40000901 | The device does not exist | Device not found |
| 40000903 | The modelId of the device does not exist | Device model not found |

---

## 2. Issue Properties

Send control commands to a specified device.

**Request**

```
POST /v1.0/end-user/devices/{device_id}/shadow/properties/issue
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| device_id | String | Yes | Device ID |

**Request Body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| properties | String | Yes | Property data in **JSON string format**. Content is a key-value map from property code (dp code) to property value |

> The value of `properties` is a string, not a JSON object. You must serialize the key-value pairs into a string before passing it.

**Request Example**

```json
{
  "properties": "{\"switch_led\": true, \"bright_value\": 500}"
}
```

**Response (success)**

```json
{
  "success": true,
  "t": 1710234567890,
  "result": {}
}
```

### Common Control Scenarios and Property Codes

| User Intent | Common Property Code | Type | Example Value |
|------------|---------------------|------|---------------|
| Turn light on/off | switch_led | bool | true / false |
| Adjust brightness | bright_value | value | 10-1000 |
| Adjust color temperature | temp_value | value | 0-1000 |
| Turn AC on/off | switch | bool | true / false |
| Set temperature | temp_set | value | 16-30 |
| Switch mode | mode | enum | "auto" / "cold" / "hot" |
| Turn plug on/off | switch_1 | bool | true / false |

> These are common examples only. The actual property codes and value ranges are determined by the "Query Device Thing Model" API response.
