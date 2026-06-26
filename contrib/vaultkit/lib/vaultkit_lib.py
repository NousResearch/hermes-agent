#!/usr/bin/env python3
"""vaultkit_lib.py — Portable KeePassXC vault toolkit (shared core).

This is the machine-agnostic heart of the vault kit. Every value that was
hardcoded in the original single-machine scripts (a specific home directory,
UID, a fixed systemd service list, a NAS backup target) is resolved here from
environment variables or a small user config file, with sensible defaults. Thin
command scripts (vault-setup, vault-unlock, vault-lock, vault-inject,
vault-breakglass) import this and add only argument parsing.

Design goals:
  * No hardcoded usernames, home dirs, UIDs, or service names.
  * Prerequisites are DETECTED, not assumed; features degrade gracefully when
    an optional dependency (TPM, age) is missing.
  * The same functions can be wrapped by an upstream `hermes vault ...` command
    later — keep side effects in small, individually-callable functions.

Config resolution order (highest priority first):
  1. Explicit environment variable (VAULTKIT_*)
  2. Config file  $VAULTKIT_CONFIG  (default: $XDG_CONFIG_HOME/vaultkit/config)
  3. Built-in default

Nothing here writes secrets to disk outside the vault and the tmpfs runtime
directory. Manifests contain key NAMES only, never values.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────
# Config resolution
# ──────────────────────────────────────────────────────────────────────────
def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))


def _xdg_runtime_dir() -> Path:
    """Runtime dir for tmpfs secret/manifest files.

    Prefer $XDG_RUNTIME_DIR (systemd sets this to /run/user/<uid>, a tmpfs
    wiped on logout). Fall back to a 0700 dir under the system temp if unset
    (e.g. a minimal container or a non-login shell).
    """
    rd = os.environ.get("XDG_RUNTIME_DIR")
    if rd and Path(rd).is_dir():
        return Path(rd)
    # Fallback: best-effort private dir. NOT necessarily tmpfs — we warn the
    # caller via runtime_is_tmpfs() so they can decide.
    uid = os.getuid() if hasattr(os, "getuid") else "0"
    fallback = Path(os.environ.get("TMPDIR", "/tmp")) / f"vaultkit-{uid}"
    fallback.mkdir(mode=0o700, exist_ok=True)
    return fallback


_CONFIG_CACHE: dict | None = None


def _load_config_file() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    path = Path(
        os.environ.get("VAULTKIT_CONFIG", str(_xdg_config_home() / "vaultkit" / "config"))
    )
    cfg: dict = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    _CONFIG_CACHE = cfg
    return cfg


def cfg(key: str, default: str = "") -> str:
    """Resolve a config value: env VAULTKIT_<KEY> > config file <KEY> > default."""
    env_key = f"VAULTKIT_{key}"
    if env_key in os.environ:
        return os.environ[env_key]
    fileval = _load_config_file().get(key)
    if fileval is not None:
        return fileval
    return default


# ──────────────────────────────────────────────────────────────────────────
# Resolved paths (all overridable, none hardcoded to a user)
# ──────────────────────────────────────────────────────────────────────────
def vault_dir() -> Path:
    """Vault directory. Default is confinement-aware.

    A snap/flatpak keepassxc-cli is sandboxed to NON-HIDDEN files under $HOME
    (the snap 'home' interface excludes dot-directories like ~/.local,
    ~/.config, ~/.cache). So the XDG-correct default ~/.local/share/vaultkit is
    unreachable for snap users. When a confined CLI is detected and the user
    hasn't overridden DIR, default to a non-hidden ~/vaultkit instead.
    """
    explicit = cfg("DIR", "")
    if explicit:
        return Path(explicit).expanduser()
    # No explicit config — choose a default the resolved CLI can actually reach.
    try:
        confined = cli_confinement() != "native"
    except Exception:
        confined = False
    if confined:
        return Path.home() / "vaultkit"          # non-hidden, snap-reachable
    return Path.home() / ".local/share/vaultkit"  # XDG default for native CLI


def vault_path() -> Path:
    return Path(cfg("VAULT", str(vault_dir() / "vault.kdbx"))).expanduser()


def static_keyfile() -> Path:
    """On-disk key file (Model B). Used when present and no TPM key exists."""
    return Path(cfg("KEYFILE", str(vault_dir() / "vault.key"))).expanduser()


def tpm_keyfile() -> Path:
    """TPM-unsealed key file in tmpfs (Model TPM), if the platform provides it."""
    return Path(cfg("TPM_KEYFILE", str(_xdg_runtime_dir() / "vaultkit-vault.key"))).expanduser()


def envfile() -> Path:
    """tmpfs env file the consuming services read via EnvironmentFile=."""
    return Path(cfg("ENVFILE", str(_xdg_runtime_dir() / "vaultkit-secrets.env"))).expanduser()


def manifest_file() -> Path:
    """tmpfs manifest of injected key NAMES (for subprocess-env default-deny)."""
    return Path(cfg("KEYS_MANIFEST", str(_xdg_runtime_dir() / "vaultkit-secrets.keys"))).expanduser()


def vault_group() -> str:
    return cfg("GROUP", "hermes")


def services() -> list[str]:
    """Services to restart after unlock. Comma/space separated, default none.

    Empty by default so the kit is useful to people who aren't running the
    Hermes gateways. Set VAULTKIT_SERVICES or SERVICES in the config file.
    """
    raw = cfg("SERVICES", "")
    return [s for s in raw.replace(",", " ").split() if s]


def keepassxc_cli() -> str:
    return cfg("CLI", os.environ.get("KPXC_CLI", "keepassxc-cli"))


def runtime_is_tmpfs() -> bool:
    """True if the runtime dir is backed by tmpfs (secrets won't hit disk)."""
    rd = _xdg_runtime_dir()
    try:
        out = subprocess.run(
            ["findmnt", "-n", "-o", "FSTYPE", "--target", str(rd)],
            capture_output=True, text=True,
        )
        return out.stdout.strip() in {"tmpfs", "ramfs"}
    except Exception:
        # findmnt absent — assume the conventional /run/user path is tmpfs.
        return str(rd).startswith("/run/user/")


# ──────────────────────────────────────────────────────────────────────────
# Prerequisite detection (degrade gracefully)
# ──────────────────────────────────────────────────────────────────────────
def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def check_prereqs(require_age: bool = False, require_tpm: bool = False) -> dict:
    """Return a dict of {feature: (ok, detail)} for the optional toolchain.

    Only keepassxc-cli is mandatory. age (breakglass) and TPM (sealed key)
    are optional and reported so the caller can decide whether to offer those
    features.
    """
    report = {}
    kp = keepassxc_cli()
    report["keepassxc-cli"] = (have(kp) or have("keepassxc.cli"),
                               f"required — install keepassxc ({kp})")
    report["age"] = (have("age") and have("age-keygen"),
                     "optional — enables encrypted breakglass export")
    # TPM: tpm2-tools present AND a TPM device node exists.
    tpm_ok = have("tpm2_createprimary") and any(
        Path(p).exists() for p in ("/dev/tpmrm0", "/dev/tpm0")
    )
    report["tpm2"] = (tpm_ok, "optional — enables TPM-sealed key file")
    return report


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────
# KeePassXC CLI helpers
# ──────────────────────────────────────────────────────────────────────────
def _resolve_cli() -> str:
    kp = keepassxc_cli()
    if have(kp):
        return kp
    if have("keepassxc.cli"):  # snap naming
        return "keepassxc.cli"
    die(f"keepassxc-cli not found (looked for {kp!r} and 'keepassxc.cli'). "
        "Install KeePassXC.")
    return kp  # unreachable


def cli_confinement() -> str:
    """Detect how the resolved keepassxc CLI is packaged.

    Returns one of: 'snap', 'flatpak', 'native'. Confined packagings (snap,
    flatpak) can ONLY touch files the sandbox exposes — in practice $HOME and
    connected removable media. They CANNOT create files under /tmp, /run/user,
    or arbitrary system paths, which produces cryptic "No such file or
    directory" / "Permission denied" errors when the vault or key file live
    outside $HOME. We detect this so the kit can warn with an actionable
    message instead of surfacing the raw CLI error.
    """
    cli = _resolve_cli()
    path = shutil.which(cli) or cli
    # snap exposes binaries under /snap/bin/ (a wrapper into /usr/bin/snap).
    if "/snap/" in path or path.startswith("/snap"):
        return "snap"
    if "flatpak" in cli or "/flatpak/" in path:
        return "flatpak"
    # Heuristic: a name like `flatpak run org.keepassxc...` set via CLI config.
    if cli.split()[:1] == ["flatpak"]:
        return "flatpak"
    return "native"


def _path_reachable_by_confined_cli(p: Path) -> bool:
    """True if a confined (snap/flatpak) CLI can reach path p.

    The snap 'home' interface exposes only NON-HIDDEN files under $HOME (plus
    ~/snap). Hidden dot-directories (~/.local, ~/.config, ~/.cache, ...) are
    blocked, as is everything outside $HOME except connected removable media
    (/media, /mnt). Verified empirically on keepassxc snap 2.7.x.
    """
    try:
        home = Path.home().resolve()
        rp = p.resolve()
    except Exception:
        return True  # can't tell — don't block
    # Removable media a snap may reach via the removable-media interface.
    for root in ("/media/", "/mnt/"):
        if str(rp).startswith(root):
            return True
    # Must be under $HOME.
    if not (str(rp).startswith(str(home) + os.sep) or rp == home):
        return False
    # ...and must not traverse a hidden (dot-prefixed) component under $HOME,
    # EXCEPT the ~/snap tree which the snap can always reach.
    rel_parts = rp.relative_to(home).parts
    if rel_parts and rel_parts[0] == "snap":
        return True
    return not any(part.startswith(".") for part in rel_parts)


def preflight_paths() -> list[str]:
    """Return a list of human-readable problems with the configured paths
    given the detected CLI confinement. Empty list == all good.

    Only the vault + key file matter here (the CLI touches those). The env
    file and manifest are written by our own Python, so their location is
    unconstrained by CLI confinement (they still SHOULD be tmpfs for secrecy,
    which runtime_is_tmpfs() covers separately).
    """
    problems: list[str] = []
    conf = cli_confinement()
    if conf == "native":
        return problems
    for label, p in (("vault", vault_path()), ("key file", static_keyfile())):
        if not _path_reachable_by_confined_cli(p):
            problems.append(
                f"{conf} keepassxc-cli CANNOT reach the {label} at {p}. "
                f"A {conf}-packaged CLI is sandboxed to your home directory. "
                f"Move it under {Path.home()} (e.g. set VAULTKIT_DIR to a path "
                f"inside your home), OR install the native keepassxc-cli "
                f"(apt/dnf/brew) which has no such restriction."
            )
    return problems


def build_auth() -> tuple[list[str], str | None]:
    """Return (auth_args, password_or_None), auto-detecting the auth model.

    Priority: TPM-unsealed key (tmpfs) > static key file > KPXC_PASSWORD env >
    interactive hidden prompt. A vault may use a key file, a password, or both.
    """
    import getpass
    auth: list[str] = []
    kf = None
    if tpm_keyfile().is_file():
        kf = tpm_keyfile()
    elif static_keyfile().is_file():
        kf = static_keyfile()
    if kf is not None:
        auth += ["-k", str(kf)]
    pw = os.environ.get("KPXC_PASSWORD")
    if pw is None and kf is None:
        pw = getpass.getpass(f"Master password for {vault_path()}: ")
    if pw is None:
        auth += ["--no-password"]  # key-file-only vault
    return auth, pw


def kpxc(auth: list[str], pw: str | None, *args: str) -> subprocess.CompletedProcess:
    cli = _resolve_cli()
    return subprocess.run(
        [cli, *args, *auth],
        input=(pw + "\n") if pw is not None else "",
        capture_output=True, text=True,
    )


def read_group_secrets(group: str | None = None) -> dict[str, str]:
    """Unlock the vault and read every entry in GROUP as {TITLE: password}."""
    group = group or vault_group()
    vp = vault_path()
    if not vp.is_file():
        # A confined CLI pointed at a vault outside $HOME also manifests as
        # "not found" — give the actionable confinement message when relevant.
        for prob in preflight_paths():
            warn(prob)
        die(f"vault not found: {vp}")
    auth, pw = build_auth()
    if kpxc(auth, pw, "ls", str(vp)).returncode != 0:
        die("failed to unlock vault (bad password or key file)")
    ls = kpxc(auth, pw, "ls", str(vp), group)
    if ls.returncode != 0:
        die(f"group {group!r} not found in vault")
    kv: dict[str, str] = {}
    for entry in ls.stdout.splitlines():
        entry = entry.strip()
        if not entry or entry.endswith("/"):
            continue
        res = kpxc(auth, pw, "show", "-s", "-a", "Password", str(vp), f"{group}/{entry}")
        val = res.stdout.rstrip("\n")
        if res.returncode == 0 and val:
            kv[entry] = val
        else:
            warn(f"empty/failed value for {group}/{entry}")
    if pw is not None:
        del pw
    if not kv:
        die("no secrets read from vault")
    return kv


# ──────────────────────────────────────────────────────────────────────────
# tmpfs writers (env file + name-only manifest)
# ──────────────────────────────────────────────────────────────────────────
def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    os.chmod(str(path), stat.S_IRUSR | stat.S_IWUSR)  # 0600


def write_runtime(kv: dict[str, str]) -> None:
    """Write the tmpfs env file AND the name-only manifest (both 0600)."""
    if not runtime_is_tmpfs():
        warn(f"runtime dir {_xdg_runtime_dir()} is not tmpfs — secret env file "
             "may touch disk. Set XDG_RUNTIME_DIR to a tmpfs path.")
    env_body = "".join(f"{k}={v}\n" for k, v in sorted(kv.items()))
    _write_private(envfile(), env_body)
    manifest_body = "".join(f"{k}\n" for k in sorted(kv.keys()))
    _write_private(manifest_file(), manifest_body)


def scrub(path: Path) -> None:
    """Overwrite-then-unlink a runtime file."""
    if not path.exists():
        return
    try:
        sz = path.stat().st_size
        with open(path, "r+b") as f:
            f.write(b"\0" * sz)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    try:
        path.unlink()
    except OSError:
        pass


def scrub_runtime() -> None:
    for p in (envfile(), manifest_file()):
        scrub(p)


def restart_services() -> tuple[list[str], list[str]]:
    """systemctl --user restart each configured service. Returns (ok, failed)."""
    ok, bad = [], []
    for s in services():
        rc = subprocess.run(
            ["systemctl", "--user", "restart", s], capture_output=True, text=True
        ).returncode
        (ok if rc == 0 else bad).append(s)
    return ok, bad


def stop_services() -> None:
    for s in services():
        subprocess.run(["systemctl", "--user", "stop", s], capture_output=True, text=True)
