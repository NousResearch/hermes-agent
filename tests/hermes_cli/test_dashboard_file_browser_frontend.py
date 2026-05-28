"""Dashboard file browser frontend wiring tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_TSX = ROOT / "web" / "src" / "App.tsx"
API_TS = ROOT / "web" / "src" / "lib" / "api.ts"
PAGE_TSX = ROOT / "web" / "src" / "pages" / "FileBrowserPage.tsx"
CHAT_PAGE_TSX = ROOT / "web" / "src" / "pages" / "ChatPage.tsx"


def test_dashboard_sidebar_exposes_file_browser_route():
    """The dashboard should expose File Browser from the left sidebar."""
    app = APP_TSX.read_text(encoding="utf-8")

    assert 'import FileBrowserPage from "@/pages/FileBrowserPage";' in app
    assert '"/files": FileBrowserPage' in app
    assert 'path: "/files"' in app
    assert 'label: "File Browser"' in app


def test_file_browser_page_uses_existing_files_api():
    """The page should list directories and download/delete files through /api/files."""
    page = PAGE_TSX.read_text(encoding="utf-8")
    api = API_TS.read_text(encoding="utf-8")

    assert "getFiles" in api
    assert "downloadFile" in api
    assert "deleteFile" in api
    assert "/api/files" in api
    assert "/api/files/download" in api
    assert 'method: "DELETE"' in api
    assert "api.getFiles" in page
    assert "api.downloadFile" in page
    assert "api.deleteFile" in page
    assert "api.uploadDocuments" in page
    assert "type=\"file\"" in page
    assert "multiple" in page
    assert "Upload here" in page
    assert "Delete" in page
    assert "File Browser" in page


def test_chat_page_uploads_documents_and_sends_paths_to_tui():
    """The embedded chat should upload docs and tell the PTY their paths."""
    chat_page = CHAT_PAGE_TSX.read_text(encoding="utf-8")
    api = API_TS.read_text(encoding="utf-8")

    assert "uploadDocuments" in api
    assert "uploadProfileDocuments" in api
    assert "/api/files/upload" in api
    assert "/api/profiles/" in api
    assert "type=\"file\"" in chat_page
    assert "multiple" in chat_page
    assert "api.uploadProfileDocuments" in chat_page
    assert "absolute_path" in chat_page
    assert "I attached document(s) through the dashboard" in chat_page
    assert "ws.send(message)" in chat_page
    assert "waitForOpenChatSocket" in chat_page
    assert "Upload completed for" in chat_page


def test_chat_page_keeps_active_websocket_ref_across_profile_reconnects():
    """Stale PTY sockets must not clear or close the newer live chat socket."""
    chat_page = CHAT_PAGE_TSX.read_text(encoding="utf-8")

    assert "let activeWs: WebSocket | null = null" in chat_page
    assert "if (wsRef.current === ws)" in chat_page
    assert "if (wsRef.current === activeWs)" in chat_page
    assert "wsRef.current?.close()" not in chat_page
