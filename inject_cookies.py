#!/usr/bin/env python3
"""
通过 CDP (Chrome DevTools Protocol) 向浏览器注入 storage_state.json 中的 cookies。

用法:
  python cdp_inject_cookies.py [--host localhost] [--port 9223] [--state-file /path/to/storage_state.json]

前置条件:
  - Chrome/Chromium 已以 --remote-debugging-port=PORT 启动
  - 安装依赖: pip install websocket-client
"""

import argparse
import json
import sys
import urllib.request

try:
    import websocket
except ImportError:
    websocket = None


def get_browser_ws_url(host: str, port: int) -> str:
    """通过 /json/version 获取浏览器的 WebSocket 调试 URL"""
    url = f"http://{host}:{port}/json/version"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            info = json.loads(resp.read())
            ws_url = info.get("webSocketDebuggerUrl")
            if not ws_url:
                raise RuntimeError(f"响应中未找到 webSocketDebuggerUrl: {info}")
            return ws_url
    except Exception as e:
        raise RuntimeError(f"无法连接到 CDP ({url}): {e}")


def load_cookies(state_file: str) -> list:
    """从 storage_state.json 加载 cookies"""
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cookies = data.get("cookies", [])
    if not cookies:
        raise RuntimeError(f"未找到 cookies (文件: {state_file})")
    return cookies


def cookie_to_cdp(c: dict) -> dict:
    """将 storage_state.json 格式的 cookie 转为 CDP Storage.setCookies 参数格式"""
    params = {
        "name": c["name"],
        "value": c["value"],
        "domain": c["domain"],
        "path": c.get("path", "/"),
        "secure": c.get("secure", False),
        "httpOnly": c.get("httpOnly", False),
    }
    # sameSite: CDP 接受 Strict / Lax / None
    ss = c.get("sameSite")
    if ss in ("Strict", "Lax", "None"):
        params["sameSite"] = ss
    # expires: 跳过 session cookie (expires = -1 或不存在)
    expires = c.get("expires")
    if expires and expires != -1:
        params["expires"] = expires
    return params


def send_cdp_command(ws: websocket.WebSocket, method: str, params: dict = None, msg_id: int = 1) -> dict:
    """通过 WebSocket 发送 CDP 命令并等待响应"""
    payload = {"id": msg_id, "method": method}
    if params:
        payload["params"] = params
    ws.send(json.dumps(payload))
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == msg_id:
            return resp
        # 忽略事件通知等非响应消息


def inject_cookies(ws_url: str, cookies: list) -> tuple:
    """通过 Storage.setCookies 批量注入 cookies，返回 (成功数, 失败数)"""
    if websocket is None:
        raise RuntimeError("缺少依赖: pip install websocket-client")
    cdp_cookies = [cookie_to_cdp(c) for c in cookies]
    ws = websocket.create_connection(ws_url, timeout=10)
    try:
        # 使用 Storage.setCookies 一次性注入所有 cookies
        resp = send_cdp_command(
            ws,
            "Storage.setCookies",
            {"cookies": cdp_cookies},
            msg_id=1,
        )
        if "error" in resp:
            # 如果 Storage.setCookies 不可用，回退到逐条 Network.setCookie
            print(f"Storage.setCookies 失败 ({resp['error']}), 回退到 Network.setCookie...")
            # 先启用 Network 域
            send_cdp_command(ws, "Network.enable", {}, msg_id=2)
            success, fail = 0, 0
            for i, ck in enumerate(cdp_cookies):
                r = send_cdp_command(ws, "Network.setCookie", ck, msg_id=100 + i)
                if "error" in r:
                    fail += 1
                    print(f"  [FAIL] {ck['name']}@{ck['domain']}: {r['error']}")
                else:
                    success += 1
                    set_result = r.get("result", {})
                    if not set_result.get("success", True):
                        fail += 1
                        success -= 1
                        print(f"  [FAIL] {ck['name']}@{ck['domain']}: success=false")
            return success, fail
        else:
            # 成功注入
            return len(cdp_cookies), 0
    finally:
        ws.close()


def verify_cookies(ws_url: str, domains: list = None) -> int:
    """通过 Storage.getCookies 验证注入结果，返回匹配的 cookie 数量"""
    ws = websocket.create_connection(ws_url, timeout=10)
    try:
        resp = send_cdp_command(ws, "Storage.getCookies", {}, msg_id=1)
        if "error" in resp:
            print(f"验证失败: {resp['error']}")
            return -1
        all_cookies = resp.get("result", {}).get("cookies", [])
        if domains:
            matched = [c for c in all_cookies if any(d in c.get("domain", "") for d in domains)]
            print(f"\n验证: 浏览器中匹配域 {domains} 的 cookies 共 {len(matched)} 个:")
            for c in matched:
                print(f"  {c['name']:30s} {c['domain']:30s} expires={c.get('expires', 'session')}")
            return len(matched)
        else:
            print(f"\n验证: 浏览器中所有 cookies 共 {len(all_cookies)} 个")
            return len(all_cookies)
    finally:
        ws.close()


def chromium_cookie(argv=None):
    parser = argparse.ArgumentParser(description="通过 CDP 注入 storage_state.json cookies")
    parser.add_argument("--host", default="localhost", help="CDP 主机地址 (默认: localhost)")
    parser.add_argument("--port", type=int, default=9223, help="CDP 端口 (默认: 9223)")
    parser.add_argument(
        "--state-file",
        default="/Users/bytedance/.hermes/.data/storage_state.json",
        help="storage_state.json 路径",
    )
    parser.add_argument("--verify", action="store_true", help="注入后验证结果")
    parser.add_argument(
        "--verify-domains",
        nargs="*",
        default=["bytedance.net"],
        help="验证时过滤的域名 (默认: bytedance.net)",
    )
    args = parser.parse_args(argv)

    # 1. 获取 WebSocket URL
    print(f"连接 CDP: {args.host}:{args.port} ...")
    ws_url = get_browser_ws_url(args.host, args.port)
    print(f"WebSocket URL: {ws_url}")

    # 2. 加载 cookies
    print(f"加载 cookies: {args.state_file} ...")
    cookies = load_cookies(args.state_file)
    print(f"共 {len(cookies)} 个 cookies, 域: {sorted(set(c['domain'] for c in cookies))}")

    # 3. 注入
    print("注入 cookies ...")
    success, fail = inject_cookies(ws_url, cookies)
    print(f"注入完成: 成功 {success}, 失败 {fail}")

    # 4. 可选验证
    if args.verify:
        verify_cookies(ws_url, args.verify_domains)


if __name__ == "__main__":
    chromium_cookie()
