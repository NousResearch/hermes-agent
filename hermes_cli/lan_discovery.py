"""EasyHermes 局域网自发现(LAN discovery)—— 节点互相广播,知道对方在哪;IP 变了也不怕。

这是 3b 跨节点执行的**地基**:要连某个节点(比如子连主),先从 peer 表查它当前的 `ip:port`,
不用手填、IP 变了下一次广播就刷新。

机制:**纯标准库 UDP 广播**(不引 mDNS/zeroconf 依赖,省得过网络装包):
  - 广播线程:每 ``BROADCAST_INTERVAL`` 秒发一个 JSON ``{m, uid, name, ip, port, ts}``(``m`` 是魔数标识);
    ``ip`` 是本机当前局域网 IP、``port`` 是本机后端(dashboard)端口。socket **bind 到 LAN 网卡 IP** 再发
    ``255.255.255.255`` + 本网段 ``x.y.z.255`` —— 否则 clash TUN 模式下默认路由走 utun,广播到不了局域网。
  - 监听线程:bind ``DISCOVERY_PORT`` 收包 → 更新 peer 表 ``{uid → {ip,port,name,last_seen}}``(跳过自己)。
  - ``peers()``:返回当前在线的 peer(剔除超过 ``PEER_TTL`` 秒没再广播的 = 离线/换网)。

注:LAN IP 探测会绕开 clash 的 utun 假地址(``ipconfig getifaddr en*`` + 私网段过滤),见 ``_probe_lan_ip``。

身份用协同账号的 ``user_id``(经 org_client 的 /account/me 解析,和注册表/授权同一套 id),
这样发现到的 peer 能直接和账号树(谁是主/子)对上 —— 子查到树根 uid 的 IP 就能连主账号。
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

DISCOVERY_PORT = 48900       # EasyHermes 自发现固定端口(局域网内一致)
BROADCAST_INTERVAL = 5.0     # 每 5s 广播一次自己
PEER_TTL = 20.0              # 超过 20s 没再听到 = 判离线
_MAGIC = "easyhermes-lan/1"  # 报文标识,过滤无关 UDP 噪声

_peers_lock = threading.Lock()
_peers: dict[str, dict] = {}     # uid -> {ip, port, name, last_seen}
_self_info_cache: dict = {}      # {token: {uid, name}} 缓存自身身份(免每次打 /account/me)
_ip_cache: dict = {"ip": "", "ts": 0.0}   # 本机 LAN IP 缓存(~15s,换网后自动刷新)
_IP_CACHE_TTL = 15.0


def _is_lan_ip(ip: str) -> bool:
    """是不是真·局域网私网 IP。排除回环 / 链路本地 / clash-TUN(198.18.x)/ CGNAT 这类「假地址」,
    否则广播出去 peer 根本连不上。"""
    if not ip:
        return False
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    if not (0 <= a <= 255 and 0 <= b <= 255):
        return False
    if a == 127 or (a == 169 and b == 254):     # 回环 / 链路本地(169.254)
        return False
    if a == 198 and b in (18, 19):              # clash / benchmark TUN 假网段
        return False
    if a == 100 and 64 <= b <= 127:             # CGNAT(Tailscale 等),不是真 LAN
        return False
    if a == 10:
        return True
    if a == 192 and b == 168:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    return False


def _probe_lan_ip() -> str:
    """探本机真 LAN IP。macOS 直接 ``ipconfig getifaddr en*``(物理网卡,绕开 clash 的 utun);
    其它平台退化到 socket-trick / ``hostname -I``,统一用 :func:`_is_lan_ip` 滤掉假地址。"""
    # macOS:物理网卡是 en*、clash 是 utun*,只问 en* 就天然绕开被劫持的默认路由
    if sys.platform == "darwin":
        for i in range(8):
            try:
                ip = subprocess.run(
                    ["ipconfig", "getifaddr", f"en{i}"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
            except Exception:  # noqa: BLE001
                ip = ""
            if _is_lan_ip(ip):
                return ip
    # 通用 1:socket-trick(可能被 clash 劫持成 198.18.x,故必须过滤)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if _is_lan_ip(ip):
            return ip
    except Exception:  # noqa: BLE001
        pass
    # 通用 2:hostname -I(Linux 给空格分隔的本机 IP 列表)
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
        for ip in out.split():
            if _is_lan_ip(ip):
                return ip
    except Exception:  # noqa: BLE001
        pass
    return "127.0.0.1"


def _local_ip() -> str:
    """本机当前局域网 IP(带 ~15s 缓存:省得每 5s 广播都 spawn 一次 ipconfig;换网/换 IP 最多 15s 后刷新)。"""
    now = time.monotonic()
    if _ip_cache["ip"] and now - _ip_cache["ts"] < _IP_CACHE_TTL:
        return _ip_cache["ip"]
    ip = _probe_lan_ip()
    _ip_cache["ip"], _ip_cache["ts"] = ip, now
    return ip


def _broadcast_addrs(lan_ip: str) -> list[str]:
    """广播目标:受限广播 255.255.255.255 + 本网段 /24 定向广播(belt-and-suspenders)。"""
    addrs = ["255.255.255.255"]
    parts = lan_ip.split(".")
    if len(parts) == 4:
        addrs.append(".".join(parts[:3]) + ".255")
    return addrs


def _self_identity() -> "tuple[Optional[str], str]":
    """(uid, name):取本节点的协同账号身份。失败 → (None, "")(不广播,只监听)。"""
    try:
        from hermes_cli import org_client

        base, token = org_client._cloud()  # noqa: SLF001
        if not (base and token):
            return None, ""
        cached = _self_info_cache.get(token)
        if cached:
            return cached["uid"], cached["name"]
        st, r = org_client._call("GET", "/account/me", token)  # noqa: SLF001
        if st != 200:
            return None, ""
        uid = str(r.get("user_id") or "").strip()
        name = str(r.get("name") or r.get("email") or "").strip()
        if uid:
            _self_info_cache[token] = {"uid": uid, "name": name}
        return (uid or None), name
    except Exception:  # noqa: BLE001
        return None, ""


def peers() -> list[dict]:
    """当前在线 peer(剔除过期)。每项 = {uid, ip, port, name, last_seen}。"""
    now = time.time()
    with _peers_lock:
        out = [
            {"uid": uid, **info}
            for uid, info in _peers.items()
            if now - info.get("last_seen", 0) <= PEER_TTL
        ]
    return sorted(out, key=lambda p: p.get("name") or p["uid"])


def _broadcaster(get_port, stop: threading.Event) -> None:
    """每 ``BROADCAST_INTERVAL`` 秒广播一次自己。每轮新建 socket 并 **bind 到当前 LAN 网卡 IP**:
    (1) 强制广播从 en0 出去 —— clash TUN 模式默认路由走 utun,不绑会从 utun 发、到不了局域网;
    (2) IP 变了下一轮自然 rebind 到新 IP。无 uid / 无合法 LAN IP 的轮次直接跳过(不广播假地址)。

    ``get_port`` 是个 callable,每轮现取后端实际端口 —— 发现线程在 uvicorn ``startup()`` 里就起了,
    那时端口还没绑定好(``app.state.bound_port`` 稍后才写),懒取保证拿到的是真端口而非 0。"""
    try:
        from hermes_cli.org_lan_server import ORG_LAN_PORT as _org_port  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        _org_port = 48901
    while not stop.is_set():
        uid, name = _self_identity()
        ip = _local_ip()
        if uid and _is_lan_ip(ip):
            try:
                port = int(get_port() or 0)
            except Exception:  # noqa: BLE001
                port = 0
            msg = json.dumps(
                {"m": _MAGIC, "uid": uid, "name": name, "ip": ip,
                 "port": port, "org_port": _org_port, "ts": time.time()}
            ).encode()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            try:
                sock.bind((ip, 0))  # 绑真 LAN 网卡:绕开被 clash 抢走的默认路由
            except OSError:
                pass
            for dst in _broadcast_addrs(ip):
                try:
                    sock.sendto(msg, (dst, DISCOVERY_PORT))
                except OSError as e:
                    logger.debug("LAN 广播发送失败 %s:%s", dst, e)
            sock.close()
        stop.wait(BROADCAST_INTERVAL)


def _listener(self_uid_box: dict, stop: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    try:
        sock.bind(("", DISCOVERY_PORT))
    except OSError as e:
        logger.warning("LAN 自发现监听 bind %s 失败:%s(同机另一个实例已在听?)", DISCOVERY_PORT, e)
        sock.close()
        return
    sock.settimeout(1.0)
    while not stop.is_set():
        try:
            data, _addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            p = json.loads(data)
        except (ValueError, TypeError):
            continue
        if not isinstance(p, dict) or p.get("m") != _MAGIC:
            continue
        uid = str(p.get("uid") or "").strip()
        if not uid or uid == self_uid_box.get("uid"):
            continue  # 跳过自己
        with _peers_lock:
            _peers[uid] = {
                "ip": str(p.get("ip") or "").strip(),
                "port": int(p.get("port") or 0),
                "org_port": int(p.get("org_port") or 0),
                "name": str(p.get("name") or "").strip(),
                "last_seen": time.time(),
            }
    sock.close()


def run_discovery(
    local_port: int = 0,
    stop_event: "threading.Event | None" = None,
    port_getter=None,
) -> None:
    """后台跑自发现:同时广播自己 + 监听别人。由后端生命周期拉起。

    ``port_getter``(可选)每轮取本机后端实际端口;没给就用静态 ``local_port``。"""
    stop = stop_event or threading.Event()
    get_port = port_getter or (lambda: int(local_port or 0))
    self_uid_box: dict = {}

    # 先同步解析一次自身 uid 填进 box,再起监听 —— 否则启动瞬间监听还不知道自己是谁,
    # 会把自己的广播当成 peer 列进去(要等 TTL 才消)。之后 _refresh_self 持续跟进身份变化。
    _uid0, _ = _self_identity()
    if _uid0:
        self_uid_box["uid"] = _uid0

    # 监听线程持续跑;广播线程每轮自己刷新身份。self_uid_box 让监听跳过自己(异步更新)。
    def _refresh_self():
        while not stop.is_set():
            uid, _ = _self_identity()
            if uid:
                self_uid_box["uid"] = uid
            stop.wait(BROADCAST_INTERVAL)

    threading.Thread(target=_listener, args=(self_uid_box, stop), name="kari-lan-listener", daemon=True).start()
    threading.Thread(target=_refresh_self, name="kari-lan-self", daemon=True).start()
    _broadcaster(get_port, stop)  # 前台跑广播(本函数本身在后台线程里)


def start_discovery_thread(local_port: int = 0, port_getter=None) -> "threading.Thread | None":
    """起后台自发现线程(daemon)。返回线程。"""
    th = threading.Thread(
        target=run_discovery,
        kwargs={"local_port": local_port, "port_getter": port_getter},
        name="kari-lan-discovery",
        daemon=True,
    )
    th.start()
    return th
