# Device Query

## 1. List All User Devices

Get all devices under the current user's authorization.

**Request**

```
GET /v1.0/end-user/devices/all
```

**Request Parameters**: None

**Response**

```json
{
  "success": true,
  "result": {
    "devices": [
      {
        "category": "dj",
        "category_name": "Light Source",
        "device_id": "0620068884f3eb414579",
        "name": "Living Room Smart Ceiling Light",
        "online": true,
        "product_id": "bcsbx******ss4xxx"
      }
    ],
    "total": 3
  }
}
```

**Device Fields**

| Field | Type | Description |
|-------|------|-------------|
| device_id | String | Device ID |
| name | String | Device name |
| category | String | Product category code |
| category_name | String | Category display name |
| product_id | String | Product ID |
| online | Boolean | Whether the device is online |
| total | Integer | Total device count (sibling of devices) |

---

## 2. List All Devices in a Home

Get all devices under a specified home.

**Request**

```
GET /v1.0/end-user/homes/{home_id}/devices
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| home_id | String | Yes | Home ID |

**Response**

```json
{
  "success": true,
  "result": {
    "devices": [
      {
        "category": "dj",
        "category_name": "Light Source",
        "device_id": "0620068884f3eb414579",
        "name": "Living Room Smart Ceiling Light",
        "online": true,
        "room_id": "123123"
      }
    ]
  }
}
```

**Device Fields**

| Field | Type | Description |
|-------|------|-------------|
| device_id | String | Device ID |
| name | String | Device name |
| category | String | Product category code |
| category_name | String | Category display name |
| online | Boolean | Whether the device is online |
| room_id | String | Room ID (returned when the device is assigned to a room, optional) |

---

## 3. List All Devices in a Room

Get all devices under a specified room.

**Request**

```
GET /v1.0/end-user/homes/room/{room_id}/devices
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| room_id | String | Yes | Room ID |

**Response**

```json
{
  "success": true,
  "result": {
    "devices": [
      {
        "category": "dj",
        "category_name": "Light Source",
        "device_id": "0620068884f3eb414579",
        "name": "Living Room Smart Ceiling Light",
        "online": true
      }
    ]
  }
}
```

**Device Fields**

| Field | Type | Description |
|-------|------|-------------|
| device_id | String | Device ID |
| name | String | Device name |
| category | String | Product category code |
| category_name | String | Category display name |
| online | Boolean | Whether the device is online |

---

## 4. Get Single Device Detail

Query detailed information of a single device, including current property states.

**Request**

```
GET /v1.0/end-user/devices/{device_id}/detail
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| device_id | String | Yes | Device ID |

**Response (device exists)**

```json
{
  "success": true,
  "t": 1773473029779,
  "result": {
    "device_id": "0620068884f3eb414579",
    "name": "Living Room Smart Ceiling Light",
    "category": "dj",
    "category_name": "Light Source",
    "product_name": "WiFi Smart Light",
    "online": true,
    "firmware_version": "1.0.0",
    "firmware_update_available": false,
    "properties": {
      "bright_value": 100,
      "control_data": "1000000000000000a010a",
      "countdown": 0,
      "do_not_disturb": false,
      "switch_led": true,
      "work_mode": "colour"
    }
  }
}
```

**Response (device not found)**

```json
{
  "success": true,
  "t": 0,
  "result": null
}
```

**Detail Fields**

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| device_id | String | Device ID | "" |
| name | String | Device name | "" |
| category | String | Product category code | "" |
| category_name | String | Category display name | "" |
| product_name | String | Product name | "" |
| online | Boolean | Whether the device is online | false |
| firmware_version | String | Firmware version | "" |
| firmware_update_available | Boolean | Whether a firmware update is available | false |
| properties | Map\<String, Object\> | Current values of the device's functional properties. Keys are property codes (dp codes), values are current values | {} |

> The `properties` field allows you to directly read the device's current state without making a separate Thing Model query. Property codes (keys) correspond to the `code` field of properties in the Thing Model. Value types depend on the `typeSpec` definition of each property (bool returns true/false, value returns a number, enum returns an enum string, etc.).

> All device list APIs return the full set of devices in a single response (no pagination). The `total` field indicates the total device count.
