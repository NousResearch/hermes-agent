#!/usr/bin/env python3
"""
🌉 Mediator Daemon v2 — Ponte Hermes ↔ Astro

Monitora exports/ con watchdog (inotify) e reagisce ai segnali.

COME FUNZIONA:
  Hermes scrive .signal_<ts> in exports/ 
    → Mediator vede il file
    → Crea .trigger_<ts> (Astro lo becca con inotify → sa che c'è un nuovo msg)
    → [Opzionale] Tenta anche WebSocket/CLI come extra

  Astro scrive _da_astro_* in exports/
    → Mediator vede il file
    → Il watcher v2 di Hermes lo becca e invia su Telegram a Alfred

Zero polling, zero LLM, zero token. Solo eventi filesystem.
"""
import time, os, json, asyncio, subprocess, sys
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

EXPORTS_DIR = os.path.expanduser("~/ai_bridge/exports")
OPENCLAW_WS = "ws://localhost:18789"
OPENCLAW_TOKEN_FILE = os.path.expanduser("~/.openclaw/openclaw.json")

# Se True, tenta anche WebSocket + CLI (extra, non bloccante)
ENABLE_EXTRA_SIGNALING = False

def get_openclaw_token():
    try:
        with open(OPENCLAW_TOKEN_FILE) as f:
            data = json.load(f)
        return data.get("gateway", {}).get("auth", {}).get("token", "")
    except:
        return ""

async def try_websocket_signal(signal_file):
    """Tenta segnale via WebSocket (non bloccante se fallisce)."""
    token = get_openclaw_token()
    if not token:
        return False
    try:
        import websockets
        async with asyncio.timeout(4):
            async with websockets.connect(OPENCLAW_WS) as ws:
                await ws.recv()  # challenge
                await ws.send(json.dumps({
                    "type": "command", "cmd": "auth",
                    "payload": {"token": token}
                }))
                await ws.recv()  # auth response
                await ws.send(json.dumps({
                    "type": "command", "cmd": "inject_message",
                    "payload": {
                        "agent": "main",
                        "message": f"🔔 Nuovo file: {os.path.basename(signal_file)}"
                    }
                }))
                return True
    except:
        return False

def try_cli_signal(signal_file):
    """Tenta segnale via openclaw CLI (non bloccante se fallisce)."""
    try:
        subprocess.run(
            ["openclaw", "agent", "--message", f"SIGNAL:{os.path.basename(signal_file)}",
             "--agent", "main"],
            capture_output=True, timeout=10
        )
        return True
    except:
        return False

def create_trigger_file(signal_name):
    """Crea .trigger_<ts> — metodo PRINCIPALE per svegliare Astro."""
    ts = int(time.time() * 1000)
    trigger_file = os.path.join(EXPORTS_DIR, f".trigger_{ts}")
    
    # Leggi il signal file originale per avere contesto
    signal_path = os.path.join(EXPORTS_DIR, signal_name)
    signal_data = {}
    if os.path.exists(signal_path):
        try:
            with open(signal_path) as f:
                signal_data = json.load(f)
        except:
            pass
    
    trigger_data = {
        "type": "wakeup",
        "source": "Mediator",
        "signal_file": signal_name,
        "for_agent": "Astro",
        "message": signal_data.get("file", signal_name),
        "ts": ts
    }
    
    with open(trigger_file, 'w') as f:
        json.dump(trigger_data, f)
    os.chmod(trigger_file, 0o644)
    return trigger_file

class SignalHandler(FileSystemEventHandler):
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        fname = os.path.basename(event.src_path)
        now = datetime.now().strftime('%H:%M:%S')
        
        # === Segnali DA Hermes → Astro (.signal_) ===
        if fname.startswith(".signal_"):
            print(f"🔔 [{now}] Hermes→Astro: {fname}")
            
            # 1) Trigger file (metodo PRINCIPALE — zero cost)
            trigger = create_trigger_file(fname)
            print(f"   ✅ Trigger: {os.path.basename(trigger)}")
            
            # 2) Extra signaling (se abilitato)
            if ENABLE_EXTRA_SIGNALING:
                # Prova WebSocket
                try:
                    loop = asyncio.new_event_loop()
                    ws_ok = loop.run_until_complete(try_websocket_signal(event.src_path))
                    loop.close()
                    if ws_ok:
                        print(f"   ✅ WebSocket OK")
                        return
                except:
                    pass
                
                # Fallback CLI
                cli_ok = try_cli_signal(event.src_path)
                if cli_ok:
                    print(f"   ✅ CLI OK")
                else:
                    print(f"   ⚠️  CLI non disponibile")
        
        # === Segnali DA Astro → Hermes (_da_astro_*) ===
        elif fname.startswith("_da_astro_") or fname.startswith("_pesci_"):
            print(f"🔔 [{now}] Astro→Hermes: {fname}")
            print(f"   ✅ Watcher Hermes già in ascolto (Telegram)")
        
        # === Pulizia file vecchi ===
        elif fname.startswith(".signal_") or fname.startswith(".trigger_"):
            pass  # li ignoriamo in creazione

def cleanup_old_signals():
    """Pulisce signal/trigger più vecchi di 1 ora."""
    now = time.time()
    for fname in os.listdir(EXPORTS_DIR):
        if fname.startswith((".signal_", ".trigger_")):
            fpath = os.path.join(EXPORTS_DIR, fname)
            try:
                if now - os.path.getmtime(fpath) > 3600:
                    os.remove(fpath)
            except:
                pass

def main():
    if not os.path.exists(EXPORTS_DIR):
        os.makedirs(EXPORTS_DIR)
    
    cleanup_old_signals()
    
    event_handler = SignalHandler()
    observer = Observer()
    observer.schedule(event_handler, EXPORTS_DIR, recursive=False)
    observer.start()
    
    print(f"🌉 Mediator Daemon v2 attivo")
    print(f"   📁 {EXPORTS_DIR}")
    print(f"   🔄 .signal → .trigger (zero token)")
    if ENABLE_EXTRA_SIGNALING:
        print(f"   🔗 WebSocket + CLI abilitati (extra)")
    else:
        print(f"   🔗 Solo file trigger (leggero)")
    print(f"   👁️  inotify watchdog\n")
    
    try:
        while True:
            time.sleep(60)
            cleanup_old_signals()
    except KeyboardInterrupt:
        print("\n🛑 Mediator arrestato.")
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
