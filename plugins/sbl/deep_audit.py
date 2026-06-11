"""
plugins/sbl/deep_audit.py — Полный аудит системы через fd/rg/stdlib

Зависимости (runtime):
  - fd (fd-find)  — поиск конфигов, логов
  - rg (ripgrep)  — извлечение портов/хостов из конфигов
  - stdlib        — всё остальное (os, re, json, subprocess)

Зависимости (pip):
  — нет. Только fd и rg из пакетного менеджера ОС.

Установка:
  apt install fd-find ripgrep   # Debian/Ubuntu
  dnf install fd-find ripgrep   # Fedora
  brew install fd ripgrep       # macOS
"""

from __future__ import annotations
import json, logging, os, re, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Shell helper ────────────────────────────────────────────────────────────

def _run(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("[SBL/deep-audit] Command timed out: %s", cmd[:80])
        return ""
    except Exception as e:
        logger.debug("[SBL/deep-audit] Command failed: %s — %s", cmd[:80], e)
        return ""


# ─── Step 1: FMC — найти всё, что запущено и где лежат конфиги ─────────────

_NAME_MAP = {
    "nginx": "nginx", "xray": "xray", "stalwart": "stalwart",
    "fail2ban-server": "fail2ban", "postgres": "postgresql",
    "sshd": "ssh", "cron": "cron", "systemd-journald": "systemd",
    "mysqld": "mysql", "redis-server": "redis", "docker": "docker",
}

_CFG_PREFIXES = {
    "nginx": ["/etc/nginx/"], "xray": ["/usr/local/etc/xray/", "/etc/xray/"],
    "stalwart": ["/opt/stalwart/"], "fail2ban": ["/etc/fail2ban/"],
    "ssh": ["/etc/ssh/"], "certbot": ["/etc/letsencrypt/"],
    "docker": ["/etc/docker/"], "postgresql": ["/etc/postgresql/", "/var/lib/postgresql/"],
    "redis": ["/etc/redis/"], "mysql": ["/etc/mysql/"],
}


def _scan_processes() -> set[str]:
    """/proc/comm — быстрее и чище, чем ps."""
    procs: set[str] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            c = Path(f"/proc/{entry}/comm").read_text().strip()
            if c and not c.startswith("("):
                procs.add(c)
        except OSError:
            continue
    # Нормализация имён
    resolved: set[str] = set(procs)
    for p in procs:
        for k, v in _NAME_MAP.items():
            if k in p:
                resolved.add(v)
    return resolved


def _scan_configs(basedirs: list[str] | None = None) -> list[dict]:
    """Все конфиги через fd — без ограничений глубины."""
    basedirs = basedirs or ["/etc", "/opt", "/usr/local/etc"]
    raw = _run(f"fd -t f -e conf -e json -e yaml -e toml -e cfg -e ini . {' '.join(basedirs)} 2>/dev/null")
    configs = []
    for line in raw.split("\n"):
        p = line.strip()
        if not p:
            continue
        try:
            st = os.stat(p)
            configs.append({
                "path": p, "mtime": st.st_mtime, "ctime": st.st_ctime,
                "size": st.st_size,
            })
        except OSError:
            continue
    return configs


def _classify(path: str) -> tuple[str, str]:
    """Классифицирует путь по FHS и маппит на имя сервиса."""
    for svc, prefixes in _CFG_PREFIXES.items():
        for pr in prefixes:
            if path.startswith(pr):
                return svc, "SYSTEM"
    for pr in ["/home/", "/tmp/", "/root/", "/var/tmp/"]:
        if path.startswith(pr):
            return "user", "USER"
    return "unknown", "UNKNOWN"


def _scan_logs() -> dict[str, int]:
    """Какие сервисы пишут логи прямо сейчас (за последний час)."""
    raw = _run("fd -t f -e log . /var/log --changed-within 1h 2>/dev/null")
    log_map: dict[str, int] = {}
    for line in raw.split("\n"):
        p = line.strip()
        if not p:
            continue
        parts = Path(p).parts
        for part in parts[2:]:  # /var/log/<dir>/
            if part not in ("log", "", "nginx", "stalwart", "postgresql",
                            "fail2ban", "mysql", "redis"):
                continue
            log_map[part] = log_map.get(part, 0) + 1
            break
    return log_map


def _scan_ports() -> dict[str, str]:
    """ss -tlnp — кто на каких портах."""
    raw = _run("ss -tlnp 2>/dev/null | rg -v '127.0.0.1:22|::1:22'")
    ports: dict[str, str] = {}
    for line in raw.split("\n"):
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        addr = parts[3]
        port = addr.split(":")[-1]
        proc = parts[-1] if len(parts) > 4 else ""
        m = re.search(r'"(\w+)"', proc)
        ports[port] = m.group(1) if m else proc
    return ports


def _scan_certs() -> tuple[list[str], list[str]]:
    """Домены в letsencrypt + сервисы, которые их используют."""
    domains = _run("ls /etc/letsencrypt/live/ 2>/dev/null").split()
    # Кто использует letsencrypt — сканируем ссылки в /proc/*/fd/
    cert_users: set[str] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            exe = os.readlink(f"/proc/{entry}/exe")
        except OSError:
            continue
        name = Path(exe).name
        for k, v in _NAME_MAP.items():
            if k in name:
                cert_users.add(v)
    return domains, sorted(cert_users)


# ─── Step 2: Universal Probe (rg) — извлечь порты, хосты, связи ────────────

def _probe_config(path: str) -> dict[str, Any]:
    """rg-based probe: порты, хосты, ссылки из любого конфига."""
    result: dict[str, Any] = {"ports": set(), "hosts": set(), "refs": set(), "upstreams": []}

    # Порты: listen, port = N, "port": N, :PORT
    for m in re.finditer(
        r'(?:listen|port)\s+["\']?(\d{2,5})["\']?',
        Path(path).read_text(), re.IGNORECASE
    ):
        result["ports"].add(int(m.group(1)))

    # JSON-порты
    if path.endswith(".json"):
        try:
            data = json.loads(Path(path).read_text())
            _json_walk(data, result)
        except json.JSONDecodeError:
            pass

    # Хосты: server_name, host, dest, proxy_pass
    for m in re.finditer(
        r'(?:server_name|host|dest|proxy_pass)\s+["\']?([\w.-]+\.[\w.-]+)',
        Path(path).read_text(), re.IGNORECASE
    ):
        result["hosts"].add(m.group(1))

    # Ссылки на файлы
    for m in re.finditer(r'(?:/etc/|/opt/|/usr/local/etc/|/var/)\S+', Path(path).read_text()):
        result["refs"].add(m.group(0))

    # upstream
    for m in re.finditer(r'upstream\s+(\S+)', Path(path).read_text(), re.IGNORECASE):
        result["upstreams"].append(m.group(1))

    # Конвертируем сеты в списки
    for k in ("ports", "hosts", "refs"):
        result[k] = sorted(result[k])
    return result


def _json_walk(data: Any, result: dict) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float)) and 1 <= v <= 65535 and "port" in k.lower():
                result["ports"].add(int(v))
            elif isinstance(v, str) and ("/" in v or "\\" in v):
                result["refs"].add(v)
            elif isinstance(v, (dict, list)):
                _json_walk(v, result)
    elif isinstance(data, list):
        for item in data:
            _json_walk(item, result)


# ─── Step 3: Корреляция ─────────────────────────────────────────────────────

def _audit() -> dict[str, Any]:
    """Полный аудит: FMC + probe + cert граф."""

    logger.info("[SBL/deep-audit] Scanning processes...")
    processes = _scan_processes()
    logger.info("[SBL/deep-audit] Scanning configs (fd)...")
    configs = _scan_configs()
    logger.info("[SBL/deep-audit] Scanning logs (fd --changed-within 1h)...")
    log_map = _scan_logs()
    logger.info("[SBL/deep-audit] Scanning ports (ss)...")
    ports = _scan_ports()
    logger.info("[SBL/deep-audit] Scanning certs...")
    cert_domains, cert_users = _scan_certs()

    # FMC: корреляция процессов, конфигов, логов
    services: dict[str, Any] = {}
    for cfg in configs:
        svc, _ = _classify(cfg["path"])
        if svc in ("unknown", "user"):
            continue
        if svc not in services:
            services[svc] = {
                "status": "CFG_ONLY", "configs": 0, "log": False,
                "ports": [], "hosts": [], "refs": [],
                "upstreams": [], "cross": [],
            }
        s = services[svc]
        s["configs"] += 1
        if svc in processes or any(svc in p for p in processes):
            s["status"] = "ACTIVE"
        if svc in log_map or any(svc in l for l in log_map):
            s["log"] = True

    # Probe: извлечение портов/хостов из ключевых конфигов
    for svc in list(services.keys()):
        prefixes = _CFG_PREFIXES.get(svc, [])
        for cfg in configs:
            if not any(cfg["path"].startswith(pr) for pr in prefixes):
                continue
            try:
                info = _probe_config(cfg["path"])
                services[svc]["ports"] = list(set(services[svc]["ports"]) | set(info["ports"]))
                services[svc]["hosts"] = list(set(services[svc]["hosts"]) | set(info["hosts"]))
                services[svc]["refs"] = list(set(services[svc]["refs"]) | set(info["refs"]))
                services[svc]["upstreams"].extend(info["upstreams"])
            except Exception as e:
                logger.debug("[SBL/deep-audit] probe failed for %s: %s", cfg["path"], e)

    # Ports из ss
    for port, proc in ports.items():
        for svc in services:
            if svc in proc:
                if port not in services[svc]["ports"]:
                    services[svc]["ports"].append(port)

    # Cross-service через общие пути (generic — не только сертификаты)
    # Собираем все refs от всех сервисов
    all_refs: dict[str, list[str]] = {}  # path → [service1, service2, ...]
    for svc, info in services.items():
        for ref in info.get("refs", []):
            # Нормализуем: разрешаем symlinks, убираем хвостовой слеш
            norm = str(Path(ref).resolve()) if Path(ref).exists() else ref.rstrip("/")
            # Берём родительскую директорию (чтобы не плодить уникальные пути)
            parent = str(Path(norm).parent) + "/"
            all_refs.setdefault(parent, [])
            if svc not in all_refs[parent]:
                all_refs[parent].append(svc)
    
    # Добавляем cert-пути как дополнительный источник
    for cert_domain in cert_domains:
        cert_path = f"/etc/letsencrypt/live/{cert_domain}/"
        all_refs.setdefault(cert_path, [])
        for svc in cert_users:
            if svc in services and svc not in all_refs[cert_path]:
                all_refs[cert_path].append(svc)
    
    # Теперь: если на один путь ссылается ≥2 сервиса — это cross-service связь
    cross_links: dict[str, list[str]] = {}
    for path, svcs in all_refs.items():
        if len(svcs) >= 2:
            for svc in svcs:
                cross_links.setdefault(svc, [])
                for other in svcs:
                    if other != svc and other not in cross_links[svc]:
                        cross_links[svc].append(other)
    
    # Применяем cross-связи к сервисам
    for svc in services:
        if svc in cross_links:
            services[svc]["cross"] = cross_links[svc]
        elif svc in cert_users and len(cert_users) >= 2:
            # fallback: хотя бы cert-связи
            services[svc]["cross"] = [s for s in cert_users if s != svc]
    
    # Сводка
    n_cross = sum(1 for s in services.values() if s.get("cross"))

    return {
        "services": services,
        "configs_total": len(configs),
        "processes_total": len(processes),
        "cert_domains": cert_domains,
        "cert_users": cert_users,
        "ports_total": len(ports),
        "n_cross": n_cross,
    }


# ─── Format ─────────────────────────────────────────────────────────────────

def format_summary(data: dict[str, Any]) -> str:
    """Человекочитаемая сводка аудита."""
    lines = [
        "═══════════════════════════════════════════",
        "  SBL Deep Audit — полный аудит системы",
        f"  {datetime.now().isoformat()[:19]}",
        "═══════════════════════════════════════════",
        f"  Файлов конфигов: {data['configs_total']}",
        f"  Процессов:       {data['processes_total']}",
        f"  Портов (tcp):    {data['ports_total']}",
        f"  Сертификатов:    {len(data['cert_domains'])} доменов",
        "",
        "  Обнаруженные сервисы:",
        "  ─────────────────────────────────────────",
    ]
    for svc, info in sorted(data["services"].items()):
        status_icon = "✅" if info["status"] == "ACTIVE" else "📁"
        log_icon = " 📝" if info.get("log") else ""
        lines.append(f"  {status_icon} {svc}{log_icon}")
        if info.get("ports"):
            lines.append(f"     ports: {', '.join(str(p) for p in info['ports'][:10])}")
        if info.get("hosts"):
            lines.append(f"     hosts: {', '.join(info['hosts'][:5])}")
        if info.get("upstreams"):
            lines.append(f"     upstreams: {', '.join(info['upstreams'][:5])}")
        if info.get("cross"):
            lines.append(f"     ⛓️  cross: {', '.join(info['cross'])}")
    lines.append("")
    lines.append(f"  Cross-service links: {data.get('n_cross', 0)} services")
    lines.append("═══════════════════════════════════════════")
    return "\n".join(lines)
