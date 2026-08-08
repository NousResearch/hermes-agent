# didi-ride-skill

English | [中文](README.md)

Unified DiDi mobility entry point — handles all transportation needs with full ride-hailing and route planning capabilities.

**ClawHub**: [didi-ride-skill-official](https://clawhub.ai/didi/didi-ride-skill-official)

> **Service Area**: Covers all cities in **Mainland China** served by DiDi. Not available in Hong Kong, Macau, Taiwan, or outside China.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Setup](#setup)
- [MCP Tools](#mcp-tools)
- [Workflow](#workflow)
- [Examples](#examples)
- [Technical Reference](#technical-reference)

---

## Quick Start

**Up and running in 3 steps:**

**Step 1 — Install mcporter**

```bash
npm install -g mcporter
```

**Step 2 — Get and configure your MCP KEY**

Scan the QR code or visit [DiDi MCP Platform](https://mcp.didichuxing.com/claw) to get your KEY, then tell the AI:

```
You: My MCP Key is xxxxxx
```

**Step 3 — Start using it**

```
You: Get me a cab to Beijing West Station
You: What's the route from Guomao to Sanlitun?
```

---

## Features

### Ride-Hailing

| Feature | Description |
|---------|-------------|
| Instant Ride | Address lookup → Price estimate → Vehicle selection → Create order |
| Scheduled Ride | Creates a scheduled task that auto-dispatches a ride at the specified time |
| Order Status | Query current order status on demand |
| Driver Location | Reverse geocode driver coordinates and display formatted location |
| Cancel Order | Show order details → User confirms → Cancel |
| Price Estimate | Compare prices across available vehicle types |
| Preferences | Save frequent addresses (home/office), vehicle type preferences, and phone number |

### Route Planning

| Feature | Description |
|---------|-------------|
| Driving | Car navigation route |
| Transit | Combined bus and subway commute options |
| Walking | Pedestrian route planning |
| Cycling | Bike route planning |
| Nearby Search | Search for points of interest around a location |

---

## Setup

### 1. Get Your MCP KEY

**Option A: Scan QR Code (Recommended, fastest)**

Open the DiDi app and scan the QR code below to instantly get your MCP KEY:

![Scan with DiDi App to get MCP Key](https://s3-yspu-cdn.didistatic.com/mcp-web/qrcode/didi_ride_skill_qrcode.png)

**Option B: Visit the website**

Go to [DiDi MCP Platform](https://mcp.didichuxing.com/claw) to obtain your MCP KEY.

### 2. Install mcporter

```bash
npm install -g mcporter
```

### 3. Configure MCP KEY

**Option A: Tell the AI in chat (Recommended)**

Just tell the AI your MCP KEY in the conversation — it will persist the configuration automatically:

```
You: My MCP Key is xxxxxx
```

**Option B: Environment variable**

```bash
export DIDI_MCP_KEY="YOUR_MCP_KEY_HERE"
```

**Option C: Config file**

Edit `~/.openclaw/openclaw.json`:

```json
{
  "skills": {
    "entries": {
      "didi-ride-skill": {
        "apiKey": "YOUR_MCP_KEY_HERE"
      }
    }
  }
}
```

### 4. Verify Configuration

```bash
# Check that the key is set
echo $DIDI_MCP_KEY

# List all available tools and their signatures (verifies key + connectivity)
export MCP_URL="https://mcp.didichuxing.com/mcp-servers?key=$DIDI_MCP_KEY"
mcporter list "$MCP_URL"

# Test map lookup
mcporter call "$MCP_URL" maps_textsearch --args '{"keywords":"Xierqi Subway Station","city":"北京市"}'
```

---

## MCP Tools

### Ride-Hailing

| Tool | Purpose |
|------|---------|
| `maps_textsearch` | Text-based address lookup, returns coordinates |
| `maps_regeocode` | Reverse geocoding (coordinates → address) |
| `taxi_estimate` | Price estimate across available vehicle types |
| `taxi_create_order` | Create a ride order |
| `taxi_query_order` | Query order status and driver info |
| `taxi_get_driver_location` | Get driver's real-time location |
| `taxi_cancel_order` | Cancel an order |
| `taxi_generate_ride_app_link` | Generate deep link to DiDi App |

### Route Planning

| Tool | Purpose |
|------|---------|
| `maps_direction_driving` | Driving route planning |
| `maps_direction_transit` | Bus/subway route planning |
| `maps_direction_walking` | Walking route planning |
| `maps_direction_bicycling` | Cycling route planning |
| `maps_place_around` | Nearby search |

---

## Workflow

### Ride-Hailing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      User requests a ride                       │
│                "Take me from Guomao to Sanlitun"                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Address lookup (maps_textsearch)                       │
│  - Resolve origin: Guomao → (116.458, 39.908)                   │
│  - Resolve destination: Sanlitun → (116.455, 39.937)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Confirm origin/destination                             │
│  - Confirm with user when address was inferred or multiple      │
│    candidates were returned                                     │
│  - No confirmation needed when user specified a precise match   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Price estimate (taxi_estimate)                         │
│  - Fetch available vehicle types and prices                     │
│  - Present options if user hasn't specified a vehicle type      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Vehicle selection                                      │
│  - Use user-specified type or preference default                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Create order (taxi_create_order)                       │
│  - Create order with selected vehicle type                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: Output order info + tracking hint                      │
│  - Display order ID, origin/destination, vehicle, estimated fare│
│  - Prompt: send "check order" to see current status             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 7: Auto status check (creates cron task)                  │
│  - Schedules a follow-up check 5 minutes later                  │
│  - Automatically queries order status and pushes notification   │
└─────────────────────────────────────────────────────────────────┘
```

### Order Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 0 | Matching | Finding a driver for you |
| 1 | Driver Accepted | Shows driver name, vehicle, plate, ETA |
| 2 | Driver Arrived | Prompt user to proceed to pickup |
| 4 | Trip In Progress | Ride has started |
| 5 | Completed | Display fare information |
| 6 | Cancelled by System | System auto-cancelled |
| 7 | Cancelled | User-cancelled |
| 3/8-12 | Other terminal states | Display corresponding status description |

### Confirmation Policy

| Action | Confirmation |
|--------|-------------|
| Instant ride | No confirmation needed — dispatches using user-specified or preferred vehicle type |
| Scheduled ride | Creates cron task directly; dispatches at scheduled time |
| Cancel order | Show order details → wait for explicit user confirmation |

---

## Examples

### Ride-Hailing

**Basic ride request**
```
You: Get me a car from Beijing West Station to Xierqi Subway Station

AI: Resolving addresses...
Origin found: Beijing West Station (116.321, 39.894)
Destination found: Xierqi Subway Station (116.297, 40.053)

Fetching prices...

1. Economy Express: ~¥32
2. Express: ~¥34
3. Premier: ~¥54

Please select a vehicle type, or say "confirm" to use the recommended option.
```

**Specify vehicle type**
```
You: Get me an Express from home to the office

AI: Dispatching Express ride...
Order created! Order ID: qo01uCXI6ojeiD
```

**Scheduled ride**
```
You: Tomorrow morning at 8am from home to the office

AI: Scheduled task set for tomorrow 08:00.
A ride will be dispatched automatically at that time.
```

**Check order**
```
You: Check order qo01uCXI6ojeiD

AI: Order ID: qo01uCXI6ojeiD
Status: Driver Accepted
Plate: 京A·XXXXX
Distance: ~1.2 km
ETA: 3 minutes
```

> Note: Only active (in-progress) orders can be queried via API. For historical orders, please check the DiDi app.

**Driver location**
```
You: Where is my driver?

AI: 🚗 Driver Location
📍 Location: Near CITIC Plaza, Jianguomenwai Ave, Chaoyang, Beijing
📏 Distance: ~1.2 km
⏱️ ETA: ~3 minutes
```

### Route Planning

**Driving**
```
You: Plan a driving route from Wangjing SOHO to Guomao

AI: 🚗 Driving Route
📍 Route: Wangjing SOHO → Guomao
📏 Distance: ~8.5 km
⏱️ Estimated time: ~25 min
```

**Transit**
```
You: What's the subway/bus route from Beijing West Station to Xierqi?

AI: 🚌 Transit Route
📍 Route: Beijing West Station → Xierqi Subway Station
⏱️ Estimated time: ~55 min
🔄 Transfer: Line 9 → Line 13
```

**Walking**
```
You: How long does it take to walk from home to the subway?

AI: 🚶 Walking Route
📍 Route: Home → Subway Station
📏 Distance: ~800 m
⏱️ Estimated time: ~10 min
```

**Cycling**
```
You: How do I bike from Wangjing to Sanlitun?

AI: 🚴 Cycling Route
📍 Route: Wangjing → Sanlitun
📏 Distance: ~6.2 km
⏱️ Estimated time: ~28 min
```

**Nearby Search**
```
You: Any coffee shops nearby?

AI: ☕ Nearby Search Results
📍 Coffee shops near your location:
1. Luckin Coffee - ~150 m away
2. Starbucks - ~320 m away
3. Manner Coffee - ~580 m away
```

---

## Technical Reference

- Full workflow: [SKILL.md](SKILL.md)
- API reference: [api_references.md](references/api_references.md)
- Error handling: [error_handling.md](references/error_handling.md)
