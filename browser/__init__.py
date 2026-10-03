"""browser — run the stock Hermes backend inside a browser tab.

Hermes core executes unmodified under Pyodide in a Web Worker. The page
(page.js) shims fetch()/WebSocket over a SharedArrayBuffer ring, so the
upstream dashboard webapp talks to the real web_server/tui_gateway code
path with zero server process. See browser/README.md for architecture.
"""
