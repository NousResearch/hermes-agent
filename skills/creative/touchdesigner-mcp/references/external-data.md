# External Data Reference

Network and device I/O — HTTP requests, WebSockets, MQTT, Serial, TCP, UDP. For MIDI/OSC specifically see `midi-osc.md`.

Common production needs:
- API polling / webhook ingestion
- Real-time data streams (sensors, market data, chat)
- IoT device control (Arduino, ESP32, smart lights)
- Inter-application messaging
- Hosting a tiny TD-side HTTP server for remote control

---

## Web DAT — HTTP Requests

```python
web = root.create(webDAT, 'api_call')
web.par.url = 'https://api.example.com/v1/status'
web.par.fetchmethod = 'get'           # 'get' | 'post' | 'put' | 'delete'
web.par.format = 'auto'                # 'auto' | 'text' | 'json'
web.par.timeout = 5.0
```

**Triggering a request:**

`webDAT` does NOT auto-fetch on cook. Trigger explicitly:

```python
web.par.fetch.pulse()
```

Or via expression on a CHOP value-change (chopExecuteDAT — see `dat-scripting.md`).

**Authentication headers:**

Use `webclientDAT` (more flexible) or set `webDAT` headers via the headers DAT:

```python
web_headers = root.create(tableDAT, 'headers')
web_headers.appendRow(['Authorization', 'Bearer YOUR_TOKEN'])
web_headers.appendRow(['Accept', 'application/json'])
web.par.headers = web_headers.path
```

**Parsing JSON response:**

```python
import json

def onTableChange(dat):
    response = dat.text          # raw response body
    data = json.loads(response)
    # Update a tableDAT or store in a constantCHOP for downstream use
    op('/project1/api_status').par.value0 = data['count']
    return
```

Wire this in a `datExecuteDAT` watching the webDAT.

**Polling pattern:**

```python
# timerCHOP fires every N seconds
timer = root.create(timerCHOP, 'poll_timer')
timer.par.length = 5.0
timer.par.cycle = True

# chopExecuteDAT on the timer's 'cycles' channel pulses the webDAT
def offToOn(channel, sampleIndex, val, prev):
    op('/project1/api_call').par.fetch.pulse()
    return
```

---

## Web Client DAT — More Robust HTTP

`webclientDAT` is the modern replacement for `webDAT` — supports streaming responses, chunked transfer, custom auth.

```python
client = root.create(webclientDAT, 'api')
client.par.method = 'POST'
client.par.url = 'https://api.example.com/events'
client.par.uploadtype = 'json'
client.par.uploaddata = '{"event": "scene_change", "scene": 3}'
client.par.request.pulse()
```

Output goes to its child `webclient1_response` DAT. Use a `datExecuteDAT` to react.

---

## Web Server DAT — TD as HTTP Server

Hosts a tiny HTTP server inside TD. Useful for:
- Status/health endpoints
- Remote control from a phone or another machine
- Webhook receivers from external services

```python
server = root.create(webserverDAT, 'control_server')
server.par.port = 8080
server.par.active = True

# Define handler in the docked callback DAT
```

In the auto-created `webserver1_callbacks` DAT:

```python
def onHTTPRequest(webServerDAT, request, response):
    path = request['uri']
    if path == '/status':
        response['statusCode'] = 200
        response['data'] = '{"fps": 60, "scene": "active"}'
    elif path == '/scene':
        idx = int(request['args'].get('index', 0))
        op('/project1/scene_switch').par.index = idx
        response['statusCode'] = 200
        response['data'] = 'OK'
    else:
        response['statusCode'] = 404
        response['data'] = 'Not Found'
    return response
```

Test from terminal: `curl http://localhost:8080/status`.

**Security:** No auth by default. Bind to localhost only or add a token check in the callback. Never expose to the public internet without auth.

---

## WebSocket DAT — Bidirectional Real-Time

For low-latency bidirectional streams (chat, live data feeds, controllers).

### Client

```python
ws = root.create(websocketDAT, 'ws_client')
ws.par.netaddress = 'wss://api.example.com/socket'
ws.par.active = True
```

In the docked callbacks DAT:

```python
def onConnect(dat):
    dat.sendText('{"action": "subscribe", "channel": "ticks"}')
    return

def onReceiveText(dat, rowIndex, message):
    # message is a string; parse JSON, dispatch to ops
    import json
    data = json.loads(message)
    op('/project1/price_chop').par.value0 = data['price']
    return

def onDisconnect(dat):
    # Optionally schedule a reconnect
    return
```

### Server

```python
ws = root.create(websocketDAT, 'ws_server')
ws.par.mode = 'server'
ws.par.port = 9001
ws.par.active = True
```

Same callback structure with an additional `clientID` arg.

---

## MQTT — Pub/Sub for IoT

```python
mqtt = root.create(mqttClientDAT, 'iot')
mqtt.par.brokeraddress = 'broker.hivemq.com'
mqtt.par.brokerport = 1883
mqtt.par.clientid = 'td_install_01'
mqtt.par.connect.pulse()

# Subscribe in callbacks DAT:
def onConnect(dat):
    dat.subscribe('home/lights/+', qos=1)
    return

def onReceive(dat, topic, payload, qos, retained, dup):
    # payload is bytes — decode if JSON
    msg = payload.decode('utf-8')
    # Dispatch by topic
    return

# Publish from anywhere:
op('iot').publish('show/scene', 'sunset', qos=0, retain=False)
```

For Mosquitto / HiveMQ self-hosted brokers use the same setup with `tcp://192.168.x.x` and your local port.

---

## Serial DAT — Arduino, USB Devices

```python
serial = root.create(serialDAT, 'arduino')
serial.par.port = '/dev/cu.usbmodem14101'   # macOS — check Arduino IDE
# Windows: 'COM3', 'COM4', etc.
serial.par.baudrate = 115200
serial.par.active = True
```

In callbacks:

```python
def onReceive(dat, rowIndex, line):
    # Each newline-terminated line from Arduino arrives here
    parts = line.split(',')
    op('/project1/sensors').par.value0 = float(parts[0])
    op('/project1/sensors').par.value1 = float(parts[1])
    return
```

Send to Arduino:
```python
op('arduino').send('LED_ON\n')
```

---

## TCP/IP DAT — Custom Protocols

For talking to non-HTTP servers (game servers, custom protocols, legacy systems).

```python
tcp = root.create(tcpipDAT, 'show_control')
tcp.par.netaddress = '192.168.1.50'
tcp.par.port = 7000
tcp.par.protocol = 'tcp'        # 'tcp' | 'udp'
tcp.par.active = True
```

Send / receive via callbacks similar to websocketDAT.

For UDP-only (fire-and-forget, no connection), use `udpoutDAT` + `udpinDAT` — simpler but unreliable across networks.

---

## Common Patterns

### REST API → Visual

```
timerCHOP (5s loop)
   → chopExecuteDAT (pulse webDAT.par.fetch on cycle)
   → webDAT (returns JSON)
   → datExecuteDAT (parse, write to constantCHOP)
   → CHOP drives glsl uniform → visuals
```

### Webhook receiver

```
webserverDAT (port 8080, /webhook endpoint)
   → callback writes to a tableDAT log + triggers a scene change
```

### Real-time stock/crypto ticker

```
websocketDAT (subscribe to feed)
   → onReceiveText callback parses JSON
   → writes to constantCHOP
   → drives bar chart / typography animation
```

### IoT-controlled installation

```
MQTT → callback dispatches by topic
   → /lights/main → constantCHOP drives lighting render
   → /audio/volume → mathCHOP for master fader
```

### Two-way phone control

```
WebSocket server in TD
   → simple HTML page on phone connects, sends slider values
   → callback writes to ops
   → TD pushes status back via dat.sendText() to phone UI
```

---

## Pitfalls

1. **`webDAT` doesn't auto-fetch** — must explicitly pulse `par.fetch`. Easy to forget.
2. **Blocking on slow APIs** — `webDAT` runs on the cook thread. A 30s API call freezes TD for 30s. Use `webclientDAT` (async) for anything potentially slow.
3. **WebSocket reconnection** — TD does NOT auto-reconnect on disconnect. Implement backoff in `onDisconnect`.
4. **Serial port permissions on macOS** — TD needs Full Disk Access OR the port needs to be unlocked via `sudo chmod 666 /dev/cu.usbmodem...` per session.
5. **MQTT broker connection state** — `mqttClientDAT` may show `connected=true` but messages don't flow if QoS is wrong or topic ACL blocks. Check broker logs.
6. **JSON parse errors crash callbacks silently** — wrap parses in try/except and log to textport. Otherwise the callback just stops firing.
7. **Firewall on Windows** — first time `webserverDAT` binds, Windows pops a firewall dialog. Approve it or the server is unreachable.
8. **CORS** — `webserverDAT` doesn't add CORS headers by default. If serving a webapp from a different origin, add `Access-Control-Allow-Origin: *` in the response.
9. **Polling vs push** — polling burns API quota. Always prefer WebSocket / webhook / MQTT for high-frequency data.
10. **Floating-point parsing** — sensor data over Serial often comes as strings. `float()` will crash on `'\n'` or `'NaN'`. Validate before converting.

---

## Quick Recipes

| Goal | Op chain |
|---|---|
| Periodic API fetch | `timerCHOP` → `chopExecuteDAT` pulses → `webDAT` → `datExecuteDAT` parses |
| Webhook receiver | `webserverDAT` (port + path), callback writes to ops |
| Real-time stream | `websocketDAT` client → onReceiveText → CHOP/DAT |
| Arduino sensor → visual | `serialDAT` → callback → `constantCHOP` → expression on visual op |
| TD ↔ phone control | `websocketDAT` server + simple HTML page on phone |
| MQTT IoT integration | `mqttClientDAT` subscribe → callback dispatches by topic |
