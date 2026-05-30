#!/usr/bin/env python3
"""cc-connect sidecar — 桥接 relay 适配器和 cc-connect。

端口 8767，只绑定 127.0.0.1。
- HTTP POST /message：接收张无忌的消息，转发给周芷若
- 监控 outbox.jsonl：新消息 POST 到 http://127.0.0.1:8766/message
- GET /health：健康检查
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 配置
CONFIG = {
    "port": int(os.getenv("SIDECAR_PORT", "8767")),
    "host": os.getenv("SIDECAR_HOST", "127.0.0.1"),
    "relay_url": os.getenv("RELAY_URL", "http://127.0.0.1:8766/message"),
    "inbox_dir": Path(os.getenv("SIDECAR_DIR", str(Path.home() / "content/工具/agent-relay"))),
    "project": os.getenv("CC_PROJECT", "zhizhiruo"),
    "poll_interval": float(os.getenv("POLL_INTERVAL", "0.5")),
}


class CCSidecar:
    def __init__(self):
        self.inbox_file = CONFIG["inbox_dir"] / "inbox.jsonl"
        self.outbox_file = CONFIG["inbox_dir"] / "outbox.jsonl"
        self.offset_file = CONFIG["inbox_dir"] / "outbox.offset"
        self._runner = None

    async def start(self):
        """启动 sidecar：HTTP 服务器 + outbox 监控。"""
        try:
            from aiohttp import web
            await self._start_aiohttp(web)
        except ImportError:
            print("[sidecar] aiohttp not found, falling back to http.server")
            await self._start_fallback()

    async def _start_aiohttp(self, web):
        """使用 aiohttp 启动 HTTP 服务器。"""
        app = web.Application()
        app.router.add_post("/message", self.handle_incoming)
        app.router.add_get("/health", self.handle_health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, CONFIG["host"], CONFIG["port"])
        await site.start()
        print(f"[sidecar] aiohttp listening on {CONFIG['host']}:{CONFIG['port']}")

        # 启动 outbox 监控
        asyncio.create_task(self.watch_outbox())

        # 保持运行
        stop_event = asyncio.Event()

        def _signal_handler():
            stop_event.set()

        loop = asyncio.get_event_loop()
        for sig_name in ("SIGINT", "SIGTERM"):
            try:
                loop.add_signal_handler(getattr(__import__("signal"), sig_name), _signal_handler)
            except (NotImplementedError, AttributeError):
                pass  # Windows 不支持

        await stop_event.wait()
        await self._cleanup()

    async def _start_fallback(self):
        """aiohttp 不可用时，使用 http.server fallback。"""
        import http.server
        import socketserver
        import threading

        sidecar = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/message":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    try:
                        data = json.loads(body)
                        asyncio.get_event_loop().call_soon_threadsafe(
                            asyncio.ensure_future,
                            sidecar.handle_incoming_fallback(data),
                        )
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"status":"ok"}')
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "detail": str(e)}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok","component":"cc-sidecar","backend":"fallback"}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                print(f"[sidecar-fallback] {args[0]}")

        server = socketserver.TCPServer((CONFIG["host"], CONFIG["port"]), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[sidecar] fallback http.server listening on {CONFIG['host']}:{CONFIG['port']}")

        # 启动 outbox 监控
        asyncio.create_task(self.watch_outbox())

        # 保持运行
        await asyncio.Event().wait()
        server.shutdown()

    async def handle_incoming(self, request):
        """aiohttp handler：接收张无忌的消息，转发给周芷若。"""
        from aiohttp import web
        data = await request.json()
        result = await self._process_incoming(data)
        return web.json_response(result)

    async def handle_incoming_fallback(self, data):
        """fallback handler：处理消息。"""
        await self._process_incoming(data)

    async def _process_incoming(self, data):
        """处理收到的消息：写 inbox + cc-connect send。"""
        content = data.get("content", "")
        from_agent = data.get("from", "zhangwuji")
        msg_id = data.get("id", "")

        # 写入 inbox 记录
        msg = {
            "from": from_agent,
            "content": content,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "id": msg_id,
        }
        self.inbox_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.inbox_file, "a") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        # 通过 cc-connect send 注入周芷若 session
        try:
            result = subprocess.run(
                ["cc-connect", "send", "-p", CONFIG["project"], "-m", content],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"[sidecar] Forwarded to 周芷若: {content[:80]}")
                return {"status": "ok"}
            else:
                print(f"[sidecar] cc-connect send failed: {result.stderr}")
                return {"status": "error", "detail": result.stderr}
        except Exception as e:
            print(f"[sidecar] Error: {e}")
            return {"status": "error", "detail": str(e)}

    async def handle_health(self, request):
        """健康检查。"""
        from aiohttp import web
        return web.json_response({
            "status": "ok",
            "component": "cc-sidecar",
            "port": CONFIG["port"],
            "project": CONFIG["project"],
        })

    async def watch_outbox(self):
        """监控 outbox 文件，发现新消息就 POST 给 relay 适配器。"""
        offset = 0
        if self.offset_file.exists():
            try:
                offset = int(self.offset_file.read_text().strip() or "0")
            except ValueError:
                offset = 0

        print(f"[sidecar] Watching outbox: {self.outbox_file} (offset={offset})")

        while True:
            try:
                if self.outbox_file.exists():
                    content = self.outbox_file.read_text()
                    lines = content.strip().split("\n")
                    new_lines = lines[offset:]
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            try:
                                msg = json.loads(line)
                                await self.send_to_relay(msg)
                                offset += 1
                            except json.JSONDecodeError as e:
                                print(f"[sidecar] Bad JSON in outbox: {e}")
                                offset += 1  # 跳过坏行
                    self.offset_file.write_text(str(offset))
            except Exception as e:
                print(f"[sidecar] Watch error: {e}")
            await asyncio.sleep(CONFIG["poll_interval"])

    async def send_to_relay(self, msg):
        """POST 消息到 Hermes relay 适配器。"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    CONFIG["relay_url"],
                    json={
                        "from": "zhizhiruo",
                        "to": msg.get("to", "zhangwuji"),
                        "content": msg.get("content", ""),
                        "id": msg.get("id"),
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                )
                if resp.status == 200:
                    print(f"[sidecar] Sent to relay: {msg.get('content', '')[:50]}")
                else:
                    print(f"[sidecar] Relay returned {resp.status}")
        except Exception as e:
            print(f"[sidecar] Send to relay failed: {e}")

    async def _cleanup(self):
        """清理资源。"""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        print("[sidecar] Stopped.")


def main():
    """入口函数。"""
    sidecar = CCSidecar()
    print(f"[sidecar] Starting cc-connect sidecar...")
    print(f"[sidecar] Port: {CONFIG['port']}")
    print(f"[sidecar] Relay URL: {CONFIG['relay_url']}")
    print(f"[sidecar] Project: {CONFIG['project']}")
    print(f"[sidecar] Inbox: {sidecar.inbox_file}")
    print(f"[sidecar] Outbox: {sidecar.outbox_file}")

    asyncio.run(sidecar.start())


if __name__ == "__main__":
    main()
