import os
import subprocess
import threading
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- UI ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html").read()

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- LOG BUFFER ----------------
LOGS = []

def log(line):
    LOGS.append(line)
    if len(LOGS) > 300:
        LOGS.pop(0)
    print(line)

@app.get("/logs")
def logs():
    return {"logs": LOGS[-100:]}

# ---------------- GATEWAY ----------------
def run_gateway():
    while True:
        log("[gateway] starting...")
        p = subprocess.Popen(
            ["hermes", "gateway", "run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in p.stdout:
            log("[gateway] " + line.strip())

        log("[gateway] crashed → restarting")
        time.sleep(2)

# ---------------- STREAM CHAT ----------------
@app.post("/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()
    message = body.get("message", "")

    def generate():
        process = subprocess.Popen(
            ["hermes", "chat", "-z", message],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            yield f"data: {line.strip()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ---------------- START ----------------
if __name__ == "__main__":
    threading.Thread(target=run_gateway, daemon=True).start()

    port = int(os.environ.get("PORT", 8080))
    print(f"[server] running on {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)
