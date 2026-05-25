import subprocess
import threading
import time

print("[server] Starting system...")

# --- Start Hermes gateway ---
def start_gateway():
    print("[server] Launching Hermes Gateway...")
    return subprocess.Popen(
        ["hermes", "gateway", "run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

# --- Stream logs ---
def stream_logs(proc):
    for line in proc.stdout:
        line = line.strip()
        print(f"[gateway] {line}")

        # highlight errors
        if "error" in line.lower():
            print(f"[DEBUG-ERROR] {line}")

# --- Main loop ---
def main():
    gateway = start_gateway()

    log_thread = threading.Thread(
        target=stream_logs,
        args=(gateway,),
        daemon=True
    )
    log_thread.start()

    while True:
        time.sleep(5)

        # if gateway crashes → restart it
        if gateway.poll() is not None:
            print("[server] Gateway crashed!")
            print(f"[server] Exit code: {gateway.returncode}")

            print("[server] Restarting gateway...")
            gateway = start_gateway()

if __name__ == "__main__":
    main()
