---
name: hermes-dashboard-start
description: Start the Hermes web UI dashboard correctly.
trigger: User asks to "start the dashboard", "open the web ui", or "launch the dashboard".
---

# Start Hermes Dashboard

The Hermes dashboard should be started using the high-level CLI wrapper rather than calling the underlying Python files directly, as the wrapper ensures the virtual environment and project paths are correctly set.

## Steps

1. Start the dashboard in the background to avoid blocking the terminal session:
   `terminal(command="hermes dashboard", background=true, notify_on_complete=true)`

## Pitfalls

- **Avoid** calling `python3 run_agent.py --dashboard`. This usually fails because it lacks the necessary environment variables and dependencies provided by the venv.
- **Avoid** calling `python3 gateway/run.py --dashboard`. The gateway runner does not support the `--dashboard` argument.
- **Always** use the `hermes` shim/command for UI-related operations.
- **First-run build delay**: On first launch (or if `web_dist/` is stale/missing), `hermes dashboard` runs `tsc -b` (TypeScript compiler) to build the React/Vite frontend before starting the server. This can take 30-60 seconds, during which NO HTTP server is listening on port 9119. The command appears to "hang" but is actually compiling. After the build completes once, subsequent starts are near-instant. Do NOT assume the dashboard is broken during this window.
- **Headless environments**: Use `--no-open` flag to prevent the dashboard from trying to open a browser: `hermes dashboard --no-open`. This avoids errors on servers without a display.
- **Package install note**: `pip install hermes-agent[web]` will fail on PEP 668 systems. Hermes is typically installed from source at `~/.hermes/hermes-agent/` with its own venv, so the web extras (FastAPI, Uvicorn) are already present. Don't chase the PyPI package — if `python3 -c "from hermes_cli.web_server import app"` succeeds, the dependencies are installed.
- **Port 9119 already in use**: A previous dashboard instance may still be bound to port 9119, causing "address already in use" errors. Kill it before restarting:
  ```python
  terminal(command="fuser -k 9119/tcp 2>/dev/null; sleep 1; fuser 9119/tcp 2>/dev/null && echo 'STILL IN USE' || echo 'PORT FREE'")
  ```
  If "STILL IN USE", escalate with `kill -9` or wait a moment for the OS to release the socket.

## Verification

- The command should return a background process session ID.
- Use `process(action="poll", session_id="...")` to verify it hasn't crashed immediately.
- After ~10 seconds (or ~60 seconds on first run while frontend compiles), verify the dashboard is responding:
  ```python
  terminal(command="sleep 10 && curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9119", timeout=20)
  ```
  Expect HTTP 200. If HTTP 000 on first run, wait longer — the TypeScript build (`tsc -b`) may still be running. Use `ps aux | grep tsc` to check if the build is still active.
- The build step can take 30-60 seconds on first run before the uvicorn server actually binds the port — `process` output may be empty during this time. Subsequent starts are near-instant.
