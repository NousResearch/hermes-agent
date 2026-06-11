"""Nextcloud Files Service -- bidirectional file sync.

Provides real local file paths for the agent by syncing files
with Nextcloud via WebDAV API, notify_push WebSocket, and inotify.
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from .base import BaseService, ServiceEvent
from .client import NextcloudFilesClient, FileInfo

logger = logging.getLogger(__name__)


class FileSyncState:
    """JSON-file-backed persistent sync state.

    Tracks remote ETags and local mtimes to enable efficient delta sync
    on restart. Analogous to NotificationStore.
    """

    def __init__(self, path: Optional[str] = None):
        default = "~/.hermes/file_sync_state.json"
        self._path = Path(os.path.expanduser(path or default))
        self._entries: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._entries = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(self._entries, dict):
                    self._entries = {}
            except Exception as e:
                logger.warning("Failed to load file sync state: %s", e)
                self._entries = {}

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def async_save(self):
        """Non-blocking save via thread pool."""
        await asyncio.to_thread(self.save)

    def get(self, path: str) -> Optional[dict]:
        return self._entries.get(path)

    def set(self, path: str, etag: str, size: int, mtime: float):
        self._entries[path] = {"etag": etag, "size": size, "mtime": mtime}
        self.save()

    def set_nosave(self, path: str, etag: str, size: int, mtime: float):
        """Set entry without saving — call save() or async_save() after batch."""
        self._entries[path] = {"etag": etag, "size": size, "mtime": mtime}

    def remove(self, path: str):
        self._entries.pop(path, None)
        self.save()

    def get_all(self) -> Dict[str, dict]:
        return dict(self._entries)


class FileWatcher:
    """Watch local directory for file changes via inotifywait subprocess.

    Yields (path, event_type) tuples for CREATE, CLOSE_WRITE, DELETE, MOVED_TO, MOVED_FROM.
    Ignores .tmp files (used for atomic downloads).
    Auto-restarts the subprocess if it dies.
    """

    def __init__(self, watch_path: Path):
        self._watch_path = Path(watch_path)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None

    def _parse_inotify_line(self, line: str):
        """Parse inotifywait CSV line: 'dir,EVENT,filename'.

        Returns (path, event_type) or None if the event should be ignored.
        """
        parts = line.strip().split(",", 2)
        if len(parts) < 3:
            return None
        directory, event_type, filename = parts

        if filename.endswith(".tmp"):
            return None

        if not filename:
            return None

        # Ignore NC-internal temp/lock files (.~hexhash, .sync_*, .nfs*)
        basename = filename.rsplit("/", 1)[-1] if "/" in filename else filename
        if basename.startswith("."):
            return None

        full_path = Path(directory) / filename
        return full_path, event_type

    async def start(self):
        """Start the inotifywait subprocess."""
        self._running = True
        self._watch_path.mkdir(parents=True, exist_ok=True)
        self._reader_task = asyncio.create_task(self._run_loop())

    async def _run_loop(self):
        """Run inotifywait, restart on failure."""
        while self._running:
            try:
                self._process = await asyncio.create_subprocess_exec(
                    "inotifywait", "-m", "-r", "-c",
                    "-e", "create,close_write,delete,moved_to,moved_from",
                    str(self._watch_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                logger.info("[FileWatcher] Started watching %s", self._watch_path)

                async for line in self._process.stdout:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        continue
                    parsed = self._parse_inotify_line(decoded)
                    if parsed:
                        await self._event_queue.put(parsed)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[FileWatcher] Error: %s", e)

            if self._running:
                logger.warning("[FileWatcher] Process died, restarting in 1s")
                await asyncio.sleep(1)

    async def get_event(self):
        """Get next file event. Blocks until available."""
        return await self._event_queue.get()

    async def stop(self):
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass


class NotifyPushListener:
    """WebSocket listener for NC notify_push file events.

    Protocol:
    1. GET capabilities -> extract notify_push.endpoints.websocket URL
    2. Connect WebSocket
    3. Send username, then app-password
    4. Receive "authenticated"
    5. Send "listen notify_file_id"
    6. Receive "notify_file_id [id1 id2 ...]" events

    Reconnects with exponential backoff on disconnect.
    """

    def __init__(self, base_url: str, username: str, password: str):
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._ws_url: Optional[str] = None
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._listener_task: Optional[asyncio.Task] = None
        self._http = None

    def _parse_push_message(self, message: str) -> list:
        """Parse a notify_push message, return file IDs or empty list."""
        if not message.startswith("notify_file_id "):
            return []
        parts = message.split()
        try:
            return [int(x) for x in parts[1:]]
        except ValueError:
            return []

    async def _discover_ws_endpoint(self, http_client) -> Optional[str]:
        """Discover WebSocket endpoint from NC capabilities."""
        url = f"{self._base_url}/ocs/v2.php/cloud/capabilities"
        try:
            resp = await http_client.get(url, headers={
                "OCS-APIRequest": "true",
                "Accept": "application/json",
            })
            resp.raise_for_status()
            caps = resp.json().get("ocs", {}).get("data", {}).get("capabilities", {})
            ws_url = caps.get("notify_push", {}).get("endpoints", {}).get("websocket")
            return ws_url
        except Exception as e:
            logger.error("[NotifyPush] Failed to discover endpoint: %s", e)
            return None

    async def start(self):
        """Start the WebSocket listener."""
        import httpx as _httpx
        self._http = _httpx.AsyncClient(
            auth=(self._username, self._password),
            timeout=10.0,
            limits=_httpx.Limits(max_keepalive_connections=0),
        )
        self._ws_url = await self._discover_ws_endpoint(self._http)
        if not self._ws_url:
            logger.warning("[NotifyPush] No WebSocket endpoint found")
            return False
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_loop())
        return True

    async def _listen_loop(self):
        """Connect to WebSocket and listen, reconnect on failure."""
        import websockets
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    await ws.send(self._username)
                    await ws.send(self._password)
                    auth_resp = await asyncio.wait_for(ws.recv(), timeout=10)
                    if auth_resp != "authenticated":
                        logger.error("[NotifyPush] Auth failed: %s", auth_resp)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue

                    await ws.send("listen notify_file_id")
                    logger.info("[NotifyPush] Connected and authenticated")
                    backoff = 1

                    async for message in ws:
                        file_ids = self._parse_push_message(message)
                        if file_ids:
                            await self._event_queue.put(file_ids)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[NotifyPush] Connection error: %s (reconnect in %ds)", e, backoff)

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def get_event(self):
        """Get next file ID list. Blocks until available."""
        return await self._event_queue.get()

    async def stop(self):
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

    async def close(self):
        await self.stop()
        if self._http:
            await self._http.aclose()


class NextcloudFilesService(BaseService):
    """Bidirectional Nextcloud file sync service."""

    name = "nextcloud_files"
    DOWNLOAD_GRACE_PERIOD = 2.0

    def __init__(self, config: dict, gateway_runner=None):
        super().__init__(config, gateway_runner)
        self._nc_url = config.get("nextcloud_url", "")
        self._username = config.get("username", "hermes")
        pw_env = config.get("app_password_env", "NEXTCLOUD_TALK_APP_PASSWORD")
        self._password = os.environ.get(pw_env, "").strip()

        local_path = config.get("local_path", "~/.hermes/nextcloud")
        self._local_path = Path(os.path.expanduser(local_path))

        deliver = config.get("deliver", "")
        if ":" in deliver:
            self._deliver_platform, self._deliver_chat_id = deliver.split(":", 1)
        else:
            self._deliver_platform = deliver
            self._deliver_chat_id = ""

        self._auto_share_with = config.get("auto_share_with", "")
        self._max_file_size = config.get("max_file_size_mb", 500) * 1024 * 1024
        self._chunk_size = config.get("chunk_size_mb", 5) * 1024 * 1024
        self._do_initial_sync = config.get("initial_sync", True)

        store_path = config.get("store_path")
        self._sync_state = FileSyncState(path=store_path)

        self._client: Optional[NextcloudFilesClient] = None
        self._push_listener: Optional[NotifyPushListener] = None
        self._file_watcher: Optional[FileWatcher] = None

        self._downloading: set = set()
        self._recently_downloaded: dict = {}
        self._uploading: set = set()
        self._recently_uploaded: dict = {}

        self._incoming_task: Optional[asyncio.Task] = None
        self._outgoing_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None

    def _on_task_done(self, task_name: str, task: asyncio.Task):
        """Log dead background tasks."""
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.error("[NC Files] Background task %s died: %s", task_name, exc, exc_info=exc)

    def _supervised_task(self, coro, name: str) -> asyncio.Task:
        """Create a task with exception logging."""
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: self._on_task_done(name, t))
        return task

    async def start(self) -> bool:
        if not self._nc_url or not self._password:
            logger.warning("[NC Files] Missing nextcloud_url or password")
            return False

        self._local_path.mkdir(parents=True, exist_ok=True)

        self._client = NextcloudFilesClient(
            self._nc_url, self._username, self._password,
        )

        self._push_listener = NotifyPushListener(
            self._nc_url, self._username, self._password,
        )
        push_ok = await self._push_listener.start()
        if push_ok:
            self._incoming_task = self._supervised_task(self._incoming_loop(), "incoming_loop")
        else:
            logger.warning("[NC Files] notify_push not available, incoming sync disabled")

        self._file_watcher = FileWatcher(self._local_path)
        await self._file_watcher.start()
        self._outgoing_task = self._supervised_task(self._outgoing_loop(), "outgoing_loop")

        if self._do_initial_sync:
            self._sync_task = self._supervised_task(self._initial_sync(), "initial_sync")

        logger.info("[NC Files] Started (local_path=%s)", self._local_path)
        return True

    async def stop(self):
        for task in [self._incoming_task, self._outgoing_task, self._sync_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._push_listener:
            await self._push_listener.close()
        if self._file_watcher:
            await self._file_watcher.stop()
        if self._client:
            await self._client.close()

        self._sync_state.save()
        logger.info("[NC Files] Stopped")

    async def on_event(self, event: ServiceEvent):
        pass

    def _to_remote_path(self, local_path: Path) -> str:
        resolved = local_path.resolve()
        if not str(resolved).startswith(str(self._local_path.resolve())):
            raise ValueError(f"Path traversal blocked: {local_path}")
        return "/" + str(local_path.relative_to(self._local_path))

    def _to_local_path(self, remote_path: str) -> Path:
        candidate = (self._local_path / remote_path.lstrip("/")).resolve()
        if not str(candidate).startswith(str(self._local_path.resolve())):
            raise ValueError(f"Path traversal blocked: {remote_path}")
        return candidate

    async def _download_file(self, remote_path: str, file_info: FileInfo) -> bool:
        local_path = self._to_local_path(remote_path)

        if file_info.size > self._max_file_size:
            logger.warning("[NC Files] Skipping %s (%.1f MB > limit %.1f MB)",
                           remote_path, file_info.size / 1e6, self._max_file_size / 1e6)
            return False

        self._downloading.add(local_path)
        try:
            ok = await self._client.download(remote_path, local_path)
            if ok:
                self._sync_state.set(
                    remote_path, etag=file_info.etag,
                    size=file_info.size, mtime=file_info.mtime,
                )
            return ok
        finally:
            self._downloading.discard(local_path)
            self._recently_downloaded[local_path] = time.monotonic()

    def _is_own_operation(self, local_path: Path) -> bool:
        """Check if a local change was caused by our own download or upload."""
        if local_path in self._downloading or local_path in self._uploading:
            return True
        for cache in (self._recently_downloaded, self._recently_uploaded):
            ts = cache.get(local_path)
            if ts and (time.monotonic() - ts) < self.DOWNLOAD_GRACE_PERIOD:
                return True
            if ts:
                del cache[local_path]
        return False

    async def _upload_file(self, local_path: Path):
        remote_path = self._to_remote_path(local_path)
        file_stat = await asyncio.to_thread(local_path.stat)
        file_size = file_stat.st_size
        self._uploading.add(local_path)

        # Ensure parent directory exists on NC (WebDAV PUT doesn't auto-create)
        parent = "/".join(remote_path.split("/")[:-1])
        if parent and parent != "/":
            await self._client.mkdir(parent)

        if file_size > self._chunk_size:
            etag = await self._client.upload_chunked(
                local_path, remote_path, chunk_size=self._chunk_size,
            )
        else:
            etag = await self._client.upload(local_path, remote_path)

        if etag:
            self._sync_state.set(
                remote_path, etag=etag,
                size=file_size, mtime=file_stat.st_mtime,
            )
            logger.info("[NC Files] Uploaded %s", remote_path)

            if self._auto_share_with:
                await self._client.share(remote_path, self._auto_share_with)
        else:
            logger.error("[NC Files] Upload failed for %s", remote_path)
        self._uploading.discard(local_path)
        self._recently_uploaded[local_path] = time.monotonic()

    async def _incoming_loop(self):
        while True:
            try:
                file_ids = await self._push_listener.get_event()
                file_infos = await self._client.resolve_file_ids(file_ids)
                for fi in file_infos:
                    if fi.is_dir:
                        continue
                    stored = self._sync_state.get(fi.path)
                    if stored and stored["etag"] == fi.etag:
                        continue
                    ok = await self._download_file(fi.path, fi)
                    if ok:
                        await self._notify_agent(fi)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[NC Files] Incoming loop error: %s", e)
                await asyncio.sleep(5)

    async def _outgoing_loop(self):
        while True:
            try:
                local_path, event_type = await self._file_watcher.get_event()

                if self._is_own_operation(local_path):
                    continue

                if event_type in ("CREATE", "CLOSE_WRITE", "MOVED_TO"):
                    if local_path.is_file():
                        # Skip if file matches sync state (downloaded by us, not written by agent)
                        remote_path = self._to_remote_path(local_path)
                        stored = self._sync_state.get(remote_path)
                        if stored and stored["size"] == local_path.stat().st_size:
                            continue
                        await self._upload_file(local_path)
                elif event_type in ("DELETE", "MOVED_FROM"):
                    remote_path = self._to_remote_path(local_path)
                    await self._client.delete(remote_path)
                    self._sync_state.remove(remote_path)
                    logger.info("[NC Files] Deleted remote %s", remote_path)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[NC Files] Outgoing loop error: %s", e)
                await asyncio.sleep(1)

    async def _initial_sync(self):
        try:
            remote_files = await self._client.propfind("/", depth="infinity")
            stored_state = self._sync_state.get_all()
            remote_paths = set()
            downloaded = 0
            uploaded = 0
            total_bytes = 0

            for fi in remote_files:
                if fi.is_dir:
                    continue
                # Skip hidden/dot files
                if fi.path.rsplit('/', 1)[-1].startswith('.'):
                    continue
                remote_paths.add(fi.path)
                stored = stored_state.get(fi.path)

                if stored and stored["etag"] == fi.etag:
                    continue

                ok = await self._download_file(fi.path, fi)
                if ok:
                    downloaded += 1
                    total_bytes += fi.size
                await asyncio.sleep(0)  # yield to event loop

            if self._local_path.exists():
                for local_file in self._local_path.rglob("*"):
                    if not local_file.is_file():
                        continue
                    if local_file.suffix == ".tmp" or local_file.name.startswith("."):
                        continue
                    remote_path = self._to_remote_path(local_file)
                    if remote_path not in remote_paths and remote_path not in stored_state:
                        await self._upload_file(local_file)
                        uploaded += 1

            for path in list(stored_state):
                if path not in remote_paths:
                    local_path = self._to_local_path(path)
                    if local_path.exists():
                        local_path.unlink()
                    self._sync_state.remove(path)

            logger.info("[NC Files] Initial sync complete: %d downloaded (%.1f MB), %d uploaded",
                        downloaded, total_bytes / 1e6, uploaded)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[NC Files] Initial sync failed: %s", e)

    async def _notify_agent(self, file_info: FileInfo):
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent, MessageType

        runner = self.gateway_runner
        if not runner:
            return

        try:
            platform = Platform(self._deliver_platform)
        except ValueError:
            logger.error("[NC Files] Unknown platform '%s'", self._deliver_platform)
            return

        adapter = runner.adapters.get(platform)
        if not adapter:
            return

        local_path = self._to_local_path(file_info.path)
        text = f"File synced: {file_info.path}\nLocal path: {local_path}"

        user_id = "service:nextcloud_files"
        chat_type = "dm"
        if hasattr(adapter, "_classify_chat"):
            chat_type = adapter._classify_chat(self._deliver_chat_id)

        source = adapter.build_source(
            chat_id=self._deliver_chat_id,
            user_id=user_id,
            user_name="NC/Files",
            chat_type=chat_type,
        )

        msg_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"ncfile_{file_info.file_id}",
        )

        await adapter.handle_message(msg_event)
