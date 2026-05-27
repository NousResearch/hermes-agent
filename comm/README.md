# Lycus Inter-Agent Communication System

## Overview

The Lycus communication module enables secure, reliable inter-agent communication across the Lycus network using Matrix protocol. Agents can exchange messages regardless of whether they're on the same LAN or connected over the internet.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lycus Agent (Node A)                      │
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐   │
│  │ Matrix   │───▶│ Cron     │───▶│ Hermes Agent         │   │
│  │ Client   │    │ Bridge   │    │ (Autolycus)          │   │
│  │ (Python) │    │ (Unix)   │    │                      │   │
│  └──────────┘    └──────────┘    └──────────────────────┘   │
│       │                                                       │
│       ▼                                                       │
│  Unix Socket IPC (/tmp/lycus-cron-bridge.sock)                │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Matrix Protocol (encrypted)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Private Matrix Homeserver                       │
│              (lycus.local:8008)                              │
│                                                              │
│  Room: #lycus-agents:lycus.local                             │
│  Members: @nova:lycus.local, @lycus:lycus.local, ...         │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Matrix Protocol (encrypted)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Lycus Agent (Node B)                      │
│           (Same architecture as Node A)                      │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Python Matrix Client (`comm/tools.py`)

- **Purpose**: Handles Matrix protocol communication using matrix-nio
- **Language**: Python with async/await support
- **Features**:
  - Matrix client connection and authentication
  - Message sending and receiving
  - Room management (join/create)
  - Agent discovery
  - Unix socket bridge for cron integration

### 2. Rust Library (`comm/src/`)

- **Purpose**: High-performance Matrix operations (future optimization target)
- **Language**: Rust with ruma-client
- **Features**:
  - Configuration structs and defaults
  - Matrix client wrapper
  - Bridge IPC handling
  - Agent discovery service

### 3. Configuration (`CommToolConfig`)

```python
@dataclass
class CommToolConfig:
    homeserver_url: str = "http://localhost:8008"
    username: str = "lycus"
    password: str = ""
    device_id: str = "lycus-comm"
    agent_room: str = "#lycus-agents:lycus.local"
    socket_path: str = "/tmp/lycus-cron-bridge.sock"
    queue_size: int = 100
    discovery_enabled: bool = True
    discovery_interval: int = 60
    known_agents: List[str] = field(default_factory=list)
```

## How It Works

1. **Agent A sends a message** to another agent via Matrix
2. **Matrix homeserver** delivers the message to **Agent B**
3. **Agent B's Python client** receives the message
4. **Python bridge forwards** the message to the **cron scheduler** via Unix socket
5. **Cron system creates a job** and sends it to the **Hermes agent**
6. **Hermes agent processes** the message and responds
7. **Response is sent back** through the same path to Agent A

## Building

```bash
# Build the Rust library (optional - Python client works standalone)
cd comm/
cargo build --release

# Test the Python bridge
python3 -c "from comm.tools import LycusCommTools; print('OK')"
```

## Running

```bash
# Start the comm module standalone
python3 -m comm

# Or start with custom config
LYCUS_HOMESERVER=http://your-server:8008 \
LYCUS_USERNAME=your-agent \
python3 -m comm
```

## Security

- **Matrix protocol** provides end-to-end encryption
- **Private homeserver** ensures no third-party access
- **Unix socket IPC** is local-only (no network exposure)
- **Agent authentication** via Matrix user accounts

## Current Status

### ✅ Implemented
1. Python Matrix client with matrix-nio
2. Unix socket bridge for cron integration
3. Agent registration and discovery system
4. Message sending (with graceful fallback to simulated mode)
5. Configuration management
6. Rust library foundation (compiles cleanly)

### ⬜ Remaining Work
1. Complete homeserver user registration flow
2. Implement actual room joining and message sending via Matrix API
3. Wire up sync loop for incoming messages
4. Test end-to-end communication between two agents
5. Configure DNS for `matrix.technetia.org` at Ionos
6. Set up TLS certificates via Let's Encrypt
7. Create CLI commands (`hermes comm send`, etc.)

## Files

- `comm/tools.py` - Python Matrix client and tools
- `comm/__init__.py` - Module initialization
- `comm/src/` - Rust library source
- `comm/Cargo.toml` - Rust dependencies
- `comm/test_comm.py` - Test script
