#!/usr/bin/env python3
"""PDR 一键启动：python start_run.py

    python start_run.py            环境检查 → 密钥 → Python 环境 → 生成 DDL → 基础服务 → 建库建表 → 启动应用
    python start_run.py --check    只做环境检查
    python start_run.py --no-docker 不用 Docker，连接本机已运行的 PostgreSQL / Redis
    python start_run.py stop       停止 Docker 基础服务

目录约定：pdr/ 是 PDR 工程（配置唯一入口 pdr/config.yaml），pdr-hermes-agent/ 是智能体引擎源码。
每一步都可重复执行，已完成的会跳过；失败修好后重跑即可。
"""

from __future__ import annotations

import argparse
import os
import secrets
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDR = ROOT / "pdr"
HERMES = ROOT / "pdr-hermes-agent"
RUNTIME = PDR / ".pdr"
VENV = RUNTIME / "venv"
ENV_FILE = PDR / ".env"
COMPOSE = PDR / "deploy" / "docker-compose.yml"
MIN_PY = (3, 11)


def say(msg: str) -> None:
    print(msg, flush=True)


def step(title: str) -> None:
    say(f"\n== {title} ==")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# --------------------------------------------------------------------------- 镜像测速

def _probe(url: str, timeout: float) -> float | None:
    start = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout):
            return time.monotonic() - start
    except Exception:  # noqa: BLE001 - 任何失败都只是“该源不可用”
        return None


def pick_mirror(fixed: str, candidates: list[str], strategy: str, timeout: float) -> str | None:
    """按 config.yaml install 段选择下载源；全部不可用时返回 None（使用工具默认源）。"""
    if strategy == "fixed" and fixed:
        return fixed
    pool = ([fixed] if fixed else []) + [c for c in candidates if c != fixed]
    if strategy == "first":
        return next((u for u in pool if _probe(u, timeout) is not None), None)
    with ThreadPoolExecutor(max_workers=max(1, len(pool))) as ex:
        timings = list(ex.map(lambda u: (_probe(u, timeout), u), pool))
    ok = sorted((t, u) for t, u in timings if t is not None)
    return ok[0][1] if ok else None


# --------------------------------------------------------------------------- 1. 环境检查

def _version(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0][:60] if out.returncode == 0 and text else None


TOOLS = [
    # 名称, 检查命令, 是否必需, 缺失时的说明
    ("Node.js ≥ 20", ["node", "--version"], True, "前端构建需要，见 https://nodejs.org 或使用 nvm"),
    ("pnpm", ["pnpm", "--version"], False, "可用 corepack enable 启用；缺失时用 npm"),
    ("uv", ["uv", "--version"], False, "推荐安装以加速 Python 依赖安装：pip install uv"),
    ("Docker", ["docker", "--version"], False, "用于一键拉起 PostgreSQL/Redis/MySQL；没有时加 --no-docker"),
    ("Docker Compose", ["docker", "compose", "version"], False, "Docker Desktop 自带；Linux 安装 docker-compose-plugin"),
    ("ffmpeg", ["ffmpeg", "-version"], False, "音视频解析需要（抽帧、抽音轨）"),
]


def check_env() -> dict[str, bool]:
    step("1/7 环境检查")
    py_ok = sys.version_info >= MIN_PY
    say(f"  {'✓' if py_ok else '✗'} Python ≥ {MIN_PY[0]}.{MIN_PY[1]}  当前 {sys.version.split()[0]}")
    found = {"python": py_ok}
    for name, cmd, required, hint in TOOLS:
        ver = _version(cmd)
        found[name] = ver is not None
        mark = "✓" if ver else ("✗" if required else "–")
        say(f"  {mark} {name:<16} {ver or '未找到：' + hint}")
    daemon = found["Docker"] and subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    found["Docker Compose"] = found["Docker Compose"] and daemon
    if found["Docker"]:
        say(f"  {'✓' if daemon else '–'} {'Docker 服务':<14} {'运行中' if daemon else '未运行：请启动 Docker，或加 --no-docker'}")
    if not py_ok or not found["Node.js ≥ 20"]:
        say("\n缺少必需环境（标 ✗ 的项），请安装后重试。")
    return found


# --------------------------------------------------------------------------- 2. 密钥

def ensure_env_file() -> dict[str, str]:
    step("2/7 密钥（pdr/.env）")
    template = (PDR / ".env.example").read_text(encoding="utf-8").splitlines()
    existing = _read_env(ENV_FILE) if ENV_FILE.exists() else {}
    auto = {"PDR_PG_SUPER_PASSWORD", "PDR_PG_PASSWORD", "PDR_MYSQL_ROOT_PASSWORD", "PDR_MYSQL_PASSWORD",
            "PDR_REDIS_PASSWORD", "PDR_JWT_SECRET", "PDR_HERMES_API_KEY", "PDR_ADMIN_PASSWORD"}
    lines, generated = [], []
    for line in template:
        key = line.split("=", 1)[0].strip()
        if not key or key.startswith("#") or "=" not in line:
            lines.append(line)
            continue
        value = existing.get(key, "")
        if not value and key in auto:
            value = secrets.token_urlsafe(24)
            generated.append(key)
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name != "nt":
        ENV_FILE.chmod(0o600)
    say(f"  已生成随机密钥：{', '.join(generated)}" if generated else "  已存在，沿用现有密钥")
    return _read_env(ENV_FILE)


def _read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


# --------------------------------------------------------------------------- 3. Python 环境

def ensure_venv(cfg_install: dict, has_uv: bool) -> None:
    step("3/7 Python 环境（pdr/.pdr/venv）")
    if not venv_python().exists():
        RUNTIME.mkdir(parents=True, exist_ok=True)
        if has_uv:
            run(["uv", "venv", str(VENV), "--python", f"{sys.version_info[0]}.{sys.version_info[1]}"])
        else:
            run([sys.executable, "-m", "venv", str(VENV)])
    index = pick_mirror(cfg_install.get("pip_index", ""), cfg_install.get("pip_candidates", []),
                        cfg_install.get("strategy", "fastest"), float(cfg_install.get("probe_timeout_seconds", 3)))
    say(f"  Python 包源：{index or '默认'}")
    reqs = ["pyyaml>=6,<7"]
    req_file = PDR / "server" / "requirements.txt"
    args = ["-r", str(req_file)] if req_file.exists() else reqs
    if has_uv:
        cmd = ["uv", "pip", "install", "--python", str(venv_python()), *args]
        cmd += ["--index-url", index] if index else []
    else:
        cmd = [str(venv_python()), "-m", "pip", "install", "-q", *args]
        cmd += ["-i", index] if index else []
    run(cmd)
    say("  依赖已就绪" + ("" if req_file.exists() else "（后端依赖清单 pdr/server/requirements.txt 将在编码阶段加入）"))


# --------------------------------------------------------------------------- 4. DDL

def render_ddl() -> None:
    step("4/7 生成 DDL 并做命名规范检查")
    run([str(venv_python()), "-m", "pdr_server.db.render"], cwd=PDR / "server")


# --------------------------------------------------------------------------- 5/6. 基础服务与建表

def compose(args: list[str], env: dict[str, str], cfg: dict, **kw) -> subprocess.CompletedProcess:
    profiles = ["--profile", "mysql"] if cfg["database"]["mysql"].get("enabled") else []
    full_env = {**os.environ, **env,
                "PDR_PG_PORT": str(cfg["database"]["postgres"]["port"]),
                "PDR_MYSQL_PORT": str(cfg["database"]["mysql"]["port"]),
                "PDR_REDIS_PORT": cfg["redis"]["url"].rsplit(":", 1)[-1].split("/")[0]}
    return run(["docker", "compose", "-f", str(COMPOSE), *profiles, *args], env=full_env, **kw)


def start_services(env: dict[str, str], cfg: dict, use_docker: bool) -> None:
    step("5/7 基础服务（PostgreSQL + pgvector / Redis / MySQL）")
    if not use_docker:
        say("  --no-docker：请确认本机 PostgreSQL（含 pgvector）与 Redis 已按 pdr/config.yaml 运行")
        return
    compose(["up", "-d", "--wait"], env, cfg)
    say("  基础服务已就绪")


def apply_schema(env: dict[str, str], cfg: dict, use_docker: bool) -> None:
    step("6/7 建库建表（幂等）")
    pg = cfg["database"]["postgres"]
    if not use_docker:
        say(f"  --no-docker：请用超级用户执行 pdr/deploy/sql/postgres/pdr_ddl.sql，并创建非超级用户 {pg['user']}")
        return
    ddl = (PDR / "deploy" / "sql" / "postgres" / "pdr_ddl.sql").read_text(encoding="utf-8")
    role = pg["user"]
    grants = f"""
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
    CREATE ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{env["PDR_PG_PASSWORD"]}';
  END IF;
END $$;
"""
    for schema in ("src", "ods", "dwd", "dws", "ads", "dim"):
        grants += (f"GRANT USAGE ON SCHEMA {schema} TO {role};\n"
                   f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {role};\n"
                   f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA {schema} TO {role};\n")
    compose(["exec", "-T", "postgres", "psql", "-q", "-v", "ON_ERROR_STOP=1", "-U", "postgres", "-d", pg["dbname"]],
            env, cfg, input=(ddl + grants).encode("utf-8"), stdout=subprocess.DEVNULL)
    say(f"  {ddl.count('COMMENT ON TABLE')} 张表与行级安全策略已就绪；应用账号 {role} 为非超级用户（行级安全对其生效）")


# --------------------------------------------------------------------------- 7. 应用进程

PENDING = [
    ("MinerU 解析服务", "mineru-kit api-server"),
    ("hermes 智能体网关", "pdr-hermes-agent/ 下的 gateway（api_server 平台）"),
    ("PDR 后端与 worker", "pdr/server"),
    ("管理端前端", "pdr/web/admin"),
]
PORTAL = PDR / "web" / "portal"


def start_portal(cfg: dict) -> subprocess.Popen | None:
    """安装并启动用户端（Vue 3 + Vite 开发服务器）；mock_api 为真时使用内置示例数据。"""
    if not (PORTAL / "package.json").exists():
        return None
    pnpm = shutil.which("pnpm")
    npm = shutil.which("npm")
    if not pnpm and not npm:
        say("  – 用户端：未找到 pnpm / npm，跳过（安装 Node.js ≥ 20 后重试）")
        return None
    inst = cfg.get("install", {})
    registry = pick_mirror(inst.get("npm_registry", ""), inst.get("npm_candidates", []),
                           inst.get("strategy", "fastest"), float(inst.get("probe_timeout_seconds", 3)))
    say(f"  用户端依赖源：{registry or '默认'}")
    tool = pnpm or npm
    run([tool, "install", *(["--registry", registry] if registry else [])], cwd=PORTAL)
    web = cfg["portal_web"]
    env = {**os.environ, "PDR_PORTAL_PORT": str(web["port"]),
           "PDR_API_URL": f"http://{cfg['app']['host']}:{cfg['app']['port']}",
           "VITE_PORTAL_MOCK": "true" if web.get("mock_api", True) else "false"}
    # 子进程单独成组，退出时整组结束（包括 pnpm 拉起的 vite）
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([tool, "run", "dev"], cwd=PORTAL, env=env, **group)
    mode = "示例数据" if web.get("mock_api", True) else "连接后端"
    say(f"  ✓ 用户端已启动（{mode}）：http://127.0.0.1:{web['port']}")
    return proc


def start_apps(cfg: dict) -> list[subprocess.Popen]:
    step("7/7 应用进程")
    for name, where in PENDING:
        say(f"  – {name}：尚未实现（{where}，编码阶段交付后由此处拉起）")
    procs = [p for p in [start_portal(cfg)] if p]
    say(f"\n访问地址：用户端 {cfg['portal_web']['public_url']}（已启动）  管理端 {cfg['admin_web']['public_url']}（待实现）"
        "；两端独立，互不链接")
    return procs


def wait_for(procs: list[subprocess.Popen]) -> None:
    """前台运行，Ctrl+C 时依次停止已拉起的进程。"""
    if not procs:
        return
    say("按 Ctrl+C 停止。")

    def _interrupt(*_: object) -> None:
        raise KeyboardInterrupt

    # 后台/服务方式启动时 SIGINT 可能被继承为忽略：显式接管 Ctrl+C 与 SIGTERM
    signal.signal(signal.SIGINT, _interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupt)
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            stop_tree(p)
        say("已停止。")


def stop_tree(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# --------------------------------------------------------------------------- 入口

def load_config() -> dict:
    import yaml  # noqa: PLC0415 - 只在虚拟环境就绪后导入

    return yaml.safe_load((PDR / "config.yaml").read_text(encoding="utf-8"))


def reexec_in_venv(argv: list[str]) -> None:
    """系统 Python 缺 PyYAML 时：先建虚拟环境装好，再用虚拟环境的 Python 重新执行本脚本。"""
    if not venv_python().exists():
        RUNTIME.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", str(VENV)])
    run([str(venv_python()), "-m", "pip", "install", "-q", "pyyaml>=6,<7"])
    os.execv(str(venv_python()), [str(venv_python()), str(Path(__file__).resolve()), *argv])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="PDR 一键启动")
    parser.add_argument("action", nargs="?", default="start", choices=["start", "stop"], help="start（默认）或 stop")
    parser.add_argument("--check", action="store_true", help="只做环境检查")
    parser.add_argument("--no-docker", action="store_true", help="不用 Docker，连接本机服务")
    args = parser.parse_args(argv)

    found = check_env()
    if args.check:
        return 0 if found["python"] else 1
    if not found["python"]:
        return 1
    try:
        cfg = load_config()
    except ModuleNotFoundError:
        reexec_in_venv(argv)
    use_docker = not args.no_docker and found["Docker Compose"]
    if args.action == "stop":
        compose(["down"], _read_env(ENV_FILE) if ENV_FILE.exists() else {}, cfg)
        return 0
    if not args.no_docker and not use_docker:
        say("\n未检测到 Docker Compose，改为 --no-docker 模式。")
    env = ensure_env_file()
    ensure_venv(cfg.get("install", {}), found["uv"])
    render_ddl()
    start_services(env, cfg, use_docker)
    apply_schema(env, cfg, use_docker)
    wait_for(start_apps(cfg))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except subprocess.CalledProcessError as exc:
        say(f"\n执行失败：{' '.join(map(str, exc.cmd))}（退出码 {exc.returncode}）。修复后重新运行即可，已完成的步骤会跳过。")
        raise SystemExit(exc.returncode) from None
