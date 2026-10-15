# Home and Space Management

## 1. List All Homes

Get all homes the current user has created or joined.

**Request**

```
GET /v1.0/end-user/homes/all
```

**Request Parameters**: None

**Response**

```json
{
  "success": true,
  "result": {
    "homes": [
      {
        "home_id": "123456",
        "name": "Smart Home Apartment",
        "role": "admin",
        "create_time": 1593661208,
        "latitude": {"Value": "30.3"},
        "longitude": {"Value": "120.07"}
      }
    ]
  }
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| home_id | String | Home ID |
| name | String | Home name |
| role | String | User's role in this home. Values: `owner` (home creator), `admin` (administrator), `member` (regular member) |
| create_time | Long | Creation time (Unix timestamp in seconds) |
| latitude | Object | Home latitude (optional). Format: `{"Value": "30.3"}`. Only returned when the home has location set. Can be used as the `lat` parameter for the Weather Query API |
| longitude | Object | Home longitude (optional). Format: `{"Value": "120.07"}`. Only returned when the home has location set. Can be used as the `lon` parameter for the Weather Query API |

---

## 2. List All Rooms in a Home

Get all rooms under a specified home.

**Request**

```
GET /v1.0/end-user/homes/{home_id}/rooms
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
    "rooms": [
      {
        "room_id": "123123",
        "name": "Living Room"
      }
    ]
  }
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| room_id | String | Room ID |
| name | String | Room name |
