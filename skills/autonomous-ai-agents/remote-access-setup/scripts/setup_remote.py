#!/usr/bin/env python3
"""Interactive Hermes Dashboard remote-access setup."""
import argparse
import getpass
import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, check=True, capture_output=True)


def import_password_hasher(hermes: str, home: Path):
    """Import the bundled hasher, re-execing in Hermes' venv when necessary."""
    source_candidates = [
        Path.home() / ".hermes" / "hermes-agent",
        home / "hermes-agent",
        Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "hermes-agent",
    ]
    for source_root in source_candidates:
        basic_plugin = source_root / "plugins" / "dashboard_auth" / "basic"
        if (basic_plugin / "__init__.py").exists() or basic_plugin.with_suffix(".py").exists():
            sys.path.insert(0, str(source_root))
            break
    try:
        from plugins.dashboard_auth.basic import hash_password
        return hash_password
    except ModuleNotFoundError as exc:
        runtime_python = Path(hermes).with_name("python.exe" if os.name == "nt" else "python")
        if (
            os.environ.get("HERMES_REMOTE_SETUP_REEXEC") != "1"
            and runtime_python.exists()
            and runtime_python.resolve() != Path(sys.executable).resolve()
        ):
            env = os.environ.copy()
            env["HERMES_REMOTE_SETUP_REEXEC"] = "1"
            os.execve(runtime_python, [str(runtime_python), *sys.argv], env)
        raise RuntimeError(f"Cannot import Hermes password hasher: {exc}") from exc

def replace_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    out, seen = [], set()
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip() + "\n")
    path.chmod(0o600)

def existing_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    return values

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("vpn", "https-local", "https-remote"), required=True)
    ap.add_argument("--username", default="admin")
    ap.add_argument("--port", type=int, default=9119)
    ap.add_argument("--public-url")
    ap.add_argument("--trusted-proxy")
    ap.add_argument("--no-service", action="store_true")
    ns = ap.parse_args()
    if ns.mode.startswith("https") and (not ns.public_url or not ns.public_url.startswith("https://")):
        ap.error("HTTPS modes require --public-url beginning with https://")
    if ns.mode == "https-remote" and not ns.trusted_proxy:
        ap.error("https-remote requires --trusted-proxy IP_OR_CIDR")
    if not 1 <= ns.port <= 65535:
        ap.error("port must be between 1 and 65535")
    hermes = shutil.which("hermes")
    if not hermes:
        print("Hermes was not found in PATH.", file=sys.stderr)
        return 2
    # The auth plugin imports Hermes runtime dependencies. Re-exec through the
    # interpreter beside the resolved Hermes launcher before prompting, so a
    # standalone/system Python cannot make the user enter the password twice.
    hermes_dir = Path(hermes).resolve().parent
    venv_python = hermes_dir / ("python.exe" if os.name == "nt" else "python")
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()
    try:
        hash_password = import_password_hasher(hermes, home)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    pw1 = getpass.getpass("Dashboard password: ")
    pw2 = getpass.getpass("Repeat password: ")
    if not pw1 or pw1 != pw2 or len(pw1) < 12:
        print("Passwords must match and contain at least 12 characters.", file=sys.stderr)
        return 2
    env_file = home / ".env"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = home / "backups" / f"remote-access-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if env_file.exists():
        shutil.copy2(env_file, backup_dir / ".env")
    old = existing_env(env_file)
    signing_secret = old.get("HERMES_DASHBOARD_BASIC_AUTH_SECRET") or secrets.token_urlsafe(48)
    replace_env(env_file, {
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME": ns.username,
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH": hash_password(pw1),
        "HERMES_DASHBOARD_BASIC_AUTH_SECRET": signing_secret,
    })
    del pw1, pw2
    if ns.public_url:
        run(hermes, "config", "set", "dashboard.public_url", ns.public_url)
    if ns.trusted_proxy:
        run(hermes, "config", "set", "dashboard.trusted_proxies", f'["{ns.trusted_proxy}"]')
    bind = "127.0.0.1" if ns.mode == "https-local" else "0.0.0.0"
    if not ns.no_service and os.name == "nt":
        launcher = home / "start-dashboard.cmd"
        if launcher.exists():
            shutil.copy2(launcher, backup_dir / launcher.name)
        launcher.write_text(
            "@echo off\r\n"
            f'"{hermes}" dashboard --host {bind} --port {ns.port} --no-open\r\n'
        )
        # HKCU Run is the native per-user equivalent of a systemd user unit:
        # no administrator rights or stored Windows account password required.
        run_key = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
        previous = subprocess.run(
            ["reg.exe", "query", run_key, "/v", "HermesDashboard"],
            text=True, capture_output=True,
        )
        if previous.returncode == 0:
            (backup_dir / "HermesDashboard-HKCU-Run.txt").write_text(previous.stdout)
        run("reg.exe", "add", run_key, "/v", "HermesDashboard", "/t", "REG_SZ",
            "/d", str(launcher), "/f")
        subprocess.Popen(
            ["cmd.exe", "/c", str(launcher)],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif not ns.no_service:
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit = unit_dir / "hermes-dashboard.service"
        if unit.exists():
            shutil.copy2(unit, backup_dir / unit.name)
        unit.write_text(
            "[Unit]\nDescription=Hermes Dashboard\nAfter=network-online.target\n\n"
            "[Service]\nType=simple\n"
            f"EnvironmentFile={env_file}\n"
            f"ExecStart={hermes} dashboard --host {bind} --port {ns.port} --no-open\n"
            "Restart=always\nRestartSec=10\nRestartPreventExitStatus=78\n\n"
            "[Install]\nWantedBy=default.target\n"
        )
        run("systemctl", "--user", "daemon-reload")
        run("systemctl", "--user", "enable", "--now", "hermes-dashboard.service")
    with (home / "CHANGELOG.md").open("a") as fh:
        fh.write(f"\n## {stamp} — Remote dashboard setup\n- What: password-gated dashboard on {bind}:{ns.port}; mode={ns.mode}.\n- Why: remote access for Hermes Mobile/Desktop/Web.\n- Backup: {backup_dir}\n")
    print("\nBackend configuration complete.")
    print(f"Bind: {bind}:{ns.port}")
    print(f"Client URL: {ns.public_url or f'http://<reachable-host>:{ns.port}'}")
    print(f"Username: {ns.username}")
    print("Password: the value entered above (not displayed or stored as plaintext)")
    if ns.public_url:
        print("The reverse proxy itself must still be configured and verified.")
    print(f"Backup: {backup_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
