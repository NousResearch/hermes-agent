# Device Management

## Rename Device

Modify a device's custom name by device ID.

**Request**

```
POST /v1.0/end-user/devices/{device_id}/attribute
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| device_id | String | Yes | Device ID |

**Request Body**

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| name | String | Yes | New device name | Max 50 characters, cannot be empty |

**Request Example**

```json
{
  "name": "Master Bedroom Smart Bedside Lamp"
}
```

**Response (success)**

```json
{
  "success": true,
  "t": 1706256000000,
  "result": {
    "device_id": "0620068884f3eb414579",
    "name": "Master Bedroom Smart Bedside Lamp"
  }
}
```

**Response (device not found)**

```json
{
  "success": false,
  "t": 1706256000000,
  "code": "DEVICE_NOT_EXIST_V2",
  "msg": "Device does not exist"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| result.device_id | String | Device ID |
| result.name | String | Updated device name |
