"""
Cosmos IDE Application.

FastAPI-based web IDE with Monaco editor and terminal.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List

try:
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class CosmosIDE:
    """
    Web-based IDE with Monaco Editor and integrated terminal.

    Routes:
    - GET /           - IDE interface
    - GET /api/files  - List files
    - GET /api/files/{path}  - Read file
    - PUT /api/files/{path}  - Write file
    - WS /ws/terminal/{id}   - Terminal WebSocket
    """

    def __init__(self, workspace_path: str = "."):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI required: pip install fastapi")

        self.workspace = Path(workspace_path).resolve()
        self.app = FastAPI(title="Cosmos IDE")
        self.terminals: Dict[str, "TerminalSession"] = {}
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return self._get_ide_html()

        @self.app.get("/api/files")
        async def list_files(path: str = ""):
            """List files in a directory."""
            try:
                dir_path = self.workspace / path
                if not dir_path.exists():
                    raise HTTPException(404, "Directory not found")

                items = []
                for item in dir_path.iterdir():
                    items.append({
                        "name": item.name,
                        "path": str(item.relative_to(self.workspace)),
                        "is_dir": item.is_dir(),
                        "size": item.stat().st_size if item.is_file() else 0,
                    })

                return sorted(items, key=lambda x: (not x["is_dir"], x["name"]))

            except PermissionError:
                raise HTTPException(403, "Permission denied")

        @self.app.get("/api/files/{path:path}")
        async def read_file(path: str):
            """Read a file."""
            try:
                file_path = self.workspace / path
                if not file_path.exists():
                    raise HTTPException(404, "File not found")
                if file_path.is_dir():
                    raise HTTPException(400, "Cannot read directory")

                content = file_path.read_text(encoding="utf-8", errors="replace")
                return {
                    "path": path,
                    "content": content,
                    "language": self._detect_language(path),
                }

            except Exception as e:
                raise HTTPException(500, str(e))

        @self.app.put("/api/files/{path:path}")
        async def write_file(path: str, data: dict):
            """Write a file."""
            try:
                file_path = self.workspace / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(data.get("content", ""), encoding="utf-8")
                return {"status": "saved", "path": path}

            except Exception as e:
                raise HTTPException(500, str(e))

        @self.app.websocket("/ws/terminal/{session_id}")
        async def terminal_ws(websocket: WebSocket, session_id: str):
            """Terminal WebSocket endpoint."""
            await websocket.accept()

            from .terminal import TerminalSession
            session = TerminalSession(str(self.workspace))
            self.terminals[session_id] = session

            try:
                await session.start()

                while True:
                    data = await websocket.receive_text()
                    session.write(data.encode())

                    output = await session.read()
                    if output:
                        await websocket.send_text(output.decode(errors="replace"))

            except Exception as e:
                logger.error(f"Terminal error: {e}")
            finally:
                await session.stop()
                del self.terminals[session_id]

    def _detect_language(self, path: str) -> str:
        """Detect language for Monaco editor."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".json": "json",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".sh": "shell",
            ".bash": "shell",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".sql": "sql",
        }
        ext = Path(path).suffix.lower()
        return ext_map.get(ext, "plaintext")

    def _get_ide_html(self) -> str:
        """Get the IDE HTML template."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cosmos IDE</title>
    <link rel="stylesheet" data-name="vs/editor/editor.main"
          href="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs/editor/editor.main.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; height: 100vh; background: #1e1e1e; color: #fff; }
        .toolbar { background: #333; padding: 8px; display: flex; gap: 10px; }
        .toolbar button { background: #0078d4; border: none; color: white; padding: 6px 12px; cursor: pointer; border-radius: 4px; }
        .main { display: flex; flex: 1; overflow: hidden; }
        .sidebar { width: 250px; background: #252526; border-right: 1px solid #333; overflow-y: auto; }
        .file-tree { padding: 10px; }
        .file-item { padding: 4px 8px; cursor: pointer; }
        .file-item:hover { background: #37373d; }
        .editor-container { flex: 1; display: flex; flex-direction: column; }
        #editor { flex: 1; }
        #terminal { height: 200px; background: #000; }
    </style>
</head>
<body>
    <div class="toolbar">
        <button onclick="saveFile()">Save (Ctrl+S)</button>
        <button onclick="toggleTerminal()">Terminal</button>
        <span id="filename" style="margin-left: auto;">No file open</span>
    </div>
    <div class="main">
        <div class="sidebar">
            <div class="file-tree" id="fileTree">Loading...</div>
        </div>
        <div class="editor-container">
            <div id="editor"></div>
            <div id="terminal" style="display: none;"></div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs/loader.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
    <script>
        let editor, currentFile, term, ws;

        require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' }});
        require(['vs/editor/editor.main'], function() {
            editor = monaco.editor.create(document.getElementById('editor'), {
                value: '// Open a file to edit',
                language: 'javascript',
                theme: 'vs-dark',
                automaticLayout: true
            });

            document.addEventListener('keydown', e => {
                if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveFile(); }
            });
        });

        async function loadFiles(path = '') {
            const res = await fetch('/api/files?path=' + path);
            const files = await res.json();
            const tree = document.getElementById('fileTree');
            tree.innerHTML = files.map(f =>
                `<div class="file-item" onclick="${f.is_dir ? `loadFiles('${f.path}')` : `openFile('${f.path}')`}">
                    ${f.is_dir ? '📁' : '📄'} ${f.name}
                </div>`
            ).join('');
        }

        async function openFile(path) {
            const res = await fetch('/api/files/' + path);
            const data = await res.json();
            currentFile = path;
            document.getElementById('filename').textContent = path;
            monaco.editor.setModelLanguage(editor.getModel(), data.language);
            editor.setValue(data.content);
        }

        async function saveFile() {
            if (!currentFile) return;
            await fetch('/api/files/' + currentFile, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: editor.getValue() })
            });
            alert('Saved!');
        }

        function toggleTerminal() {
            const termEl = document.getElementById('terminal');
            if (termEl.style.display === 'none') {
                termEl.style.display = 'block';
                if (!term) {
                    term = new Terminal();
                    term.open(termEl);
                    ws = new WebSocket(`ws://${location.host}/ws/terminal/main`);
                    ws.onmessage = e => term.write(e.data);
                    term.onData(data => ws.send(data));
                }
            } else {
                termEl.style.display = 'none';
            }
        }

        loadFiles();
    </script>
</body>
</html>'''

    def run(self, host: str = "127.0.0.1", port: int = 8080):
        """Run the IDE server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
