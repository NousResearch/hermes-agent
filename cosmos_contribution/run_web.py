#!/usr/bin/env python3
"""
Run cosmos Web Interface + Emotional API Server

Starts both:
1. Web Chat Interface on port 8081
2. Emotional Token Server on port 8765

Usage:
    python run_web.py
    python run_web.py --port 8081 --host 0.0.0.0
"""

import argparse
import os
import sys
import threading
import time

# Load .env file (API keys: GEMINI_API_KEY, IBM_QUANTUM_TOKEN, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: manually parse .env if python-dotenv not installed
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip())

# IMPORT TORCH EARLY TO PREVENT DLL CONFLICTS (shm.dll error)
# Use a subprocess with timeout to detect hangs (torch 2.8.0 issue)
import subprocess
def _test_torch_import(timeout=10):
    """Test if torch can actually import without hanging."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import torch; print('OK')"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        stdout, _ = proc.communicate(timeout=timeout)
        return b"OK" in stdout
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False
    except Exception:
        return False

if os.environ.get("COSMOS_SKIP_TORCH") != "1":
    print("    [INIT]  Testing torch import (10s timeout)...")
    if _test_torch_import(timeout=10):
        try:
            import torch
            print(f"    [OK]    torch {torch.__version__} loaded")
        except ImportError:
            pass
    else:
        print("    [WARN]  torch import timed out — setting COSMOS_SKIP_TORCH=1")
        print("            TTS voice cloning disabled; server will use fallback voice.")
        os.environ["COSMOS_SKIP_TORCH"] = "1"
else:
    print("    [SKIP]  torch (COSMOS_SKIP_TORCH=1)")

# ALIAS FIX FOR WINDOWS: Map 'Cosmos' to 'cosmos' to allow lowercase imports
try:
    import Cosmos
    sys.modules['cosmos'] = Cosmos
except ImportError:
    pass

try:
    import cosmos.web.server
    print(f"DEBUG: Loaded Server Module: {cosmos.web.server.__file__}")
except ImportError as e:
    print(f"DEBUG: Server Import Failed: {e}")

def start_emotional_server(host="0.0.0.0", port=8765, full_sensory=False):
    """Start the emotional API server in a background thread."""
    try:
        # Add emotional_api to path
        emotional_api_path = os.path.join(os.path.dirname(__file__), "emotional_api")
        sys.path.insert(0, emotional_api_path)
        
        if full_sensory:
            from full_system import FullSystemController
            controller = FullSystemController()
            # Run headless to avoid blocking or GUI issues in some envs, 
            # but user likely wants the camera window if running locally.
            # We'll use a thread for the controller.run()
            controller.run(headless=False)
        else:
            from emotion_server import run_server
            run_server(host=host, port=port)
    except ImportError as e:
        print(f"⚠️  Emotional API not available: {e}")
    except Exception as e:
        print(f"⚠️  Emotional API server error: {e}")

def main():
    # Get default port from env or use 8081 (avoiding conflict with Apache on 8080)
    default_port = int(os.environ.get("cosmos_WEB_PORT", "8081"))
    
    parser = argparse.ArgumentParser(description="cosmos Web Interface + Emotional API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=default_port, help="Web interface port")
    parser.add_argument("--emotion-port", type=int, default=8765, help="Emotional API port")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no token verification)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotional API server")
    parser.add_argument("--full-sensory", action="store_true", default=True, help="Enable full sensory API (Camera/Mic)")
    args = parser.parse_args()

    # Set environment variables
    os.environ["cosmos_WEB_HOST"] = args.host
    os.environ["cosmos_WEB_PORT"] = str(args.port)

    if args.demo:
        os.environ["cosmos_DEMO_MODE"] = "true"

    # Import uvicorn
    import uvicorn

    # Detect API capabilities based on configured keys
    has_gemini = bool(os.getenv('GEMINI_API_KEY'))
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_quantum = bool(os.getenv('IBM_QUANTUM_TOKEN'))
    
    capabilities = []
    capabilities.append('Ollama (llama3.1:8b)')
    if has_gemini:
        capabilities.append('Gemini API')
    if has_openai:
        capabilities.append('ChatGPT API')
    if has_quantum:
        capabilities.append('IBM Quantum Bridge')
    
    cap_str = ' | '.join(capabilities)
    
    print(f"""
    +==================================================================+
    |             * COSMOS Neural Interface v3.0                    |
    |                  + 12D CST Emotional Engine                      |
    |                  + FULL LIVE MODE                                |
    +==================================================================+
    |  Web Interface:    http://localhost:{args.port:<5}                          |
    |  Emotional API:    http://localhost:{args.emotion_port:<5}                          |
    |  Mode:             {'FULL LIVE':<4}                                          |
    +==================================================================+
    |  Active Capabilities:                                            |
    |    {cap_str:<60} |
    +==================================================================+
    |  Emotional Endpoints:                                            |
    |    GET  /state         - Current cosmos_packet                 |
    |    GET  /stream        - SSE token stream                        |
    |    WS   /ws            - WebSocket tokens                        |
    |    GET  /system_prompt - LLM steering prompt                     |
    +==================================================================+
    """)

    # Check if emotional port is already in use
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', args.emotion_port))
    is_port_open = result == 0
    sock.close()

    if is_port_open:
        print(f"    [INFO]  Port {args.emotion_port} is busy - Emotional API already running externally. Using Remote API.")
        args.no_emotion = True

    # -------------------------------------------------------------
    # QUANTUM BRIDGE CONNECTION TEST
    # -------------------------------------------------------------
    print("    [INIT]  Testing Quantum Bridge Connection...")
    try:
        from cosmos.core.quantum_bridge import get_quantum_bridge
        qb = get_quantum_bridge()
        if qb and qb.connect():
             entropy = qb.get_entropy()
             backend_name = qb.backend.name if qb.backend else 'simulation'
             print(f"    [OK]    Quantum Bridge Active | Entropy Source: {backend_name} | Value: {entropy:.4f}")
        else:
             print("    [WARN]  Quantum Bridge Offline - System will run in DETERMINISTIC mode.")
    except Exception as e:
        print(f"    [FAIL]  Quantum Connection Error: {e}")
    # -------------------------------------------------------------

    # Start emotional API server in background thread (unless disabled)
    if not args.no_emotion:
        emotion_thread = threading.Thread(
            target=start_emotional_server,
            args=(args.host, args.emotion_port, args.full_sensory),
            daemon=True
        )
        emotion_thread.start()
        print(f"    🎭 Emotional API Server ({'FULL SENSORY' if args.full_sensory else 'BASIC'}) starting on port {args.emotion_port}...")
        time.sleep(1)  # Give it time to start
    
    # Start main web server
    if args.reload:
        print("    [WARN]  Reload is enabled, passing module string (this may fail if PYTHONPATH is ignored by Uvicorn workers).")
        uvicorn.run(
            "cosmos.web.server:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="info"
        )
    else:
        # Pass the instantiated app directly to avoid module discovery issues in child processes
        import cosmos.web.server
        import cosmos.web.server_remote # Register evolution and cognitive system hooks
        uvicorn.run(
            cosmos.web.server.app,
            host=args.host,
            port=args.port,
            log_level="info"
        )


if __name__ == "__main__":
    main()
