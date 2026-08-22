# Hermes Agent - Android Optimization Plan

## Goal
Enable Hermes Agent to run on Android (Termux) with minimal resource usage while maintaining access to all tools/skills via smart bridging.

## Strategy

### 1. Dual-Mode Architecture
- **Local Mode**: Runs lightweight Python core on Android (Termux)
- **Bridge Mode**: Heavy operations proxy through VPS via WebSocket (Bridght)

### 2. Tools Classification

#### ✅ Native on Android (Lightweight)
- Text operations, file management
- SQLite (built-in)
- Basic HTTP/REST calls
- WebSocket communication
- JSON processing
- Shell commands (via Termux)
- Git operations
- Logging

#### ⚡ Bridge Mode (Proxy to VPS)
- Docker-based tools
- Browser automation (browserbase, firecrawl, browser_use)
- Image generation (heavy ML models)
- Voice processing
- Heavy AI inference
- Cloud providers (modal, daytona, singularity)
- Large file downloads/uploads
- Python environment management

### 3. Implementation Steps

#### Step 1: Create Android Setup Script
#### Step 2: Create Lightweight Core Package
#### Step 3: Create Bridge Mode Skill
#### Step 4: Optimize Memory Usage
#### Step 5: Test on Android

## File Structure
```
hermes-agent/
├── android_setup.sh          # Termux setup script
├── android_config.yaml       # Android-specific config
├── lightweight_core/         # Minimal core for Android
│   ├── __init__.py
│   ├── core.py               # Simplified AIAgent
│   ├── tools_android.py      # Android-friendly tools only
│   └── bridge_client.py      # WebSocket bridge to VPS
├── bridge_mode.py            # Bridge mode orchestrator
└── ANDROID_OPTIMIZATION.md  # This file
```
