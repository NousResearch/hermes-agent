"""Dev environment probes (dev family, Windows): toolchains, editors, repos.

Registration only at import time. Shared helpers and memoised reads live in apps_dev.
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import os
import re
import time

from userscan.registry import probe
from userscan.specs.apps_dev import (DEV, _LA, _PD, _PF, _PF86, _RA, _U, _appx_match, _day, _file_version,
    _is_operator_path, _isdir, _ls, _memo, _mtime, _open, _read_json, _redact, _running, _rv, _spawn,
    _subkeys, _uninst_match, _which, winreg)

# ================================================================== dev

@probe(id="apps.pkg_managers", level="L1", family=DEV, tier="T0", collect="core")
def apps_pkg_managers(h, facts):
    """Package managers present: winget (alias + DesktopAppInstaller), scoop, choco, npm/uv/pipx on PATH."""
    U, LA = _U(h), _LA(h)
    dai = _appx_match(h, r"^Microsoft\.DesktopAppInstaller$")
    out = {"winget": os.path.exists(os.path.join(LA, "Microsoft", "WindowsApps", "winget.exe")),
           "winget_appinstaller_version": dai[0]["version"] if dai else None,
           "scoop": _isdir(os.path.join(U, "scoop")) or _isdir(os.path.join(_PD(), "scoop")),
           "choco": _isdir(os.path.join(_PD(), "chocolatey")),
           "npm": bool(_which("npm")), "uv": bool(_which("uv")), "pipx": bool(_which("pipx")),
           "pnpm": bool(_which("pnpm")), "bun": bool(_which("bun"))}
    return {"present": True, **out}


@probe(id="dev.devmode_sudo_longpaths", level="L1", family=DEV, tier="T0", collect="core")
def dev_devmode_sudo_longpaths(h, facts):
    """Developer Mode, Windows sudo and LongPathsEnabled registry toggles."""
    dm = _open(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock")
    fs = _open(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
    su = _open(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Sudo")
    sudo_mode = {0: "disabled", 1: "forceNewWindow", 2: "disableInput", 3: "normal"}
    sv = _rv(su, "Enabled") if su else None
    return {"present": True, "developer_mode": _rv(dm, "AllowDevelopmentWithoutDevLicense") if dm else None,
            "long_paths": _rv(fs, "LongPathsEnabled") if fs else None,
            "sudo": sudo_mode.get(sv, sv) if sv is not None else None}


@probe(id="dev.docker", level="L1", family=DEV, tier="T0", collect="core")
def dev_docker(h, facts):
    """Docker Desktop install + settings-store.json whitelisted keys (no docker spawn)."""
    inst = os.path.join(_PF(), "Docker", "Docker")
    un = _uninst_match(h, r"^Docker Desktop")
    if not (_isdir(inst) or un):
        return None
    sel = {}
    for p in (os.path.join(_RA(h), "Docker", "settings-store.json"), os.path.join(_RA(h), "Docker", "settings.json")):
        j = _read_json(p)
        if isinstance(j, dict):
            sel = {k: j[k] for k in j if k in ("wslEngineEnabled", "WslEngineEnabled", "autoStart", "AutoStart",
                                               "kubernetesEnabled", "KubernetesEnabled", "UseContainerdSnapshotter",
                                               "useContainerdSnapshotter", "memoryMiB", "MemoryMiB", "cpus", "Cpus",
                                               "EnableDockerAI", "EnableInference")
                   and isinstance(j[k], (bool, int, str))}
            break
    cli = os.path.join(inst, "resources", "bin", "docker.exe")
    return {"present": True, "desktop_version": un[0]["version"] if un else None, "install_date": un[0]["install_date"] if un else None,
            "cli_version": (lambda v: None if v in (None, "0.0.0.0") else v)(_file_version(cli)), "settings": sel,
            "service_running": bool(_running(h, r"^com\.docker\.service|^docker desktop"))}


@probe(id="dev.docker_config_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_docker_config_presence(h, facts):
    """~/.docker/config.json presence/size (may hold registry auths; never opened)."""
    return h.meta(os.path.join(_U(h), ".docker", "config.json"))


@probe(id="dev.gh_auth_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_gh_auth_presence(h, facts):
    """GitHub CLI hosts.yml presence/size (means gh logged in; never opened)."""
    return h.meta(os.path.join(_RA(h), "GitHub CLI", "hosts.yml"))


@probe(id="dev.git", level="L1", family=DEV, tier="T0", collect="core")
def dev_git(h, facts):
    """git on PATH + Git for Windows version from registry / file version (no spawn)."""
    exe = _which("git")
    k = _open(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\GitForWindows")
    reg_ver = _rv(k, "CurrentVersion") if k else None
    inst = _rv(k, "InstallPath") if k else None
    if not (exe or reg_ver):
        return None
    base = inst or os.path.join(_PF(), "Git")
    flavor = next((d for d in ("clangarm64", "mingw64", "mingw32") if _isdir(os.path.join(base, d))), None)
    return {"present": True, "on_path": bool(exe), "git_for_windows_version": reg_ver,
            "file_version": _file_version(os.path.join(base, "cmd", "git.exe")), "flavor": flavor}


@probe(id="dev.npm_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
def dev_npm_globals(h, facts):
    """Global npm packages from %APPDATA%\\npm\\node_modules listing (no `npm root -g` spawn)."""
    root = os.path.join(_RA(h), "npm", "node_modules")
    names = _ls(root, 500)
    if names is None:
        return {"present": False}
    pk = []
    for n in names:
        if n.startswith("@"):
            pk += [f"{n}/{s}" for s in (_ls(os.path.join(root, n), 100) or [])]
        elif not n.startswith("."):
            pk.append(n)
    return {"present": True, "count": len(pk), "packages": sorted(pk)[:80]}


@probe(id="dev.npmrc_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_npmrc_presence(h, facts):
    """~/.npmrc presence/size (may hold registry tokens; never opened)."""
    return h.meta(os.path.join(_U(h), ".npmrc"))


_PATH_TOOLS = [("winget_links", r"\\WinGet\\Links"), ("winget_packages", r"\\WinGet\\Packages"),
               ("windowsapps", r"\\WindowsApps$"), ("git", r"\\Git\\(cmd|bin)"), ("nodejs", r"\\nodejs"),
               ("npm_global", r"\\npm$"), ("python", r"\\Python\d+"), ("uv_or_local_bin", r"\\\.local\\bin|\\uv\\"),
               ("hermes", r"\\hermes\\bin"), ("cargo", r"\\\.cargo\\bin"), ("go", r"\\go\\bin|\\Go\\bin"),
               ("vscode", r"Microsoft VS Code"), ("ollama", r"\\Ollama"), ("cua", r"\\Cua\\"),
               ("docker", r"\\Docker\\"), ("powershell7", r"\\PowerShell\\7"), ("dotnet", r"\\dotnet"),
               ("java", r"\\(jdk|Java|Adoptium)"), ("cuda", r"CUDA"), ("tailscale", r"Tailscale"),
               ("nvidia", r"NVIDIA"), ("gh", r"GitHub CLI"), ("obs", r"obs-studio"), ("ffmpeg", r"ffmpeg")]


@probe(id="dev.path_entries", level="L1", family=DEV, tier="T0", collect="core")
def dev_path_entries(h, facts):
    """User/machine PATH entry counts and known-tool matches (paths themselves not emitted)."""
    uk = _open(winreg.HKEY_CURRENT_USER, "Environment")
    mk = _open(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
    up = [x for x in ((_rv(uk, "Path") if uk else "") or "").split(";") if x]
    mp = [x for x in ((_rv(mk, "Path") if mk else "") or "").split(";") if x]
    tools = {"user": sorted({t for t, rx in _PATH_TOOLS for e in up if re.search(rx, e, re.I)}),
             "machine": sorted({t for t, rx in _PATH_TOOLS for e in mp if re.search(rx, e, re.I)})}
    return {"present": True, "user": len(up), "machine": len(mp), "tools": tools}


@probe(id="dev.pwsh7", level="L1", family=DEV, tier="T0", collect="core")
def dev_pwsh7(h, facts):
    """PowerShell 7 presence and file version (never spawned)."""
    exe = _which("pwsh")
    cand = os.path.join(_PF(), "PowerShell", "7", "pwsh.exe")
    path = exe if exe and "windowsapps" not in exe.lower() else (cand if os.path.exists(cand) else exe)
    if not path:
        return None
    return {"present": True, "on_path": bool(exe), "version": _file_version(path) if os.path.exists(path) else None,
            "store_alias": bool(exe and "windowsapps" in exe.lower())}


@probe(id="dev.ssh_config", level="L2", family=DEV, tier="T1", collect="core", gate="dev.path_entries")
def dev_ssh_config(h, facts):
    """~/.ssh/config Host alias count and Include count (hostnames not emitted); known_hosts line count."""
    d = os.path.join(_U(h), ".ssh")
    if not _isdir(d):
        return {"present": False}
    cfg = os.path.join(d, "config")
    aliases = includes = lines = 0
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f):
                if n > 5000:
                    break
                lines += 1
                m = re.match(r"^\s*Host\s+(.+)$", line, re.I)
                if m:
                    aliases += len([x for x in m.group(1).split() if x != "*"])
                if re.match(r"^\s*Include\s", line, re.I):
                    includes += 1
    kh = os.path.join(d, "known_hosts")
    khl = None
    if os.path.exists(kh):
        with open(kh, "rb") as f:
            khl = sum(1 for _ in zip(range(100000), f))
    return {"present": True, "config": os.path.exists(cfg), "config_lines": lines, "host_aliases": aliases,
            "includes": includes, "known_hosts_lines": khl}


_TOOLCHAIN = ["node", "npm", "pnpm", "yarn", "bun", "deno", "nvm", "fnm", "volta", "uv", "py", "pipx", "conda", "poetry",
              "rustc", "cargo", "rustup", "go", "dotnet", "java", "javac", "mvn", "gradle", "cmake", "ninja", "clang",
              "gcc", "zig", "nvcc", "gh", "pwsh", "docker", "kubectl", "terraform", "tailscale", "claude", "codex",
              "hermes", "ollama", "code", "cursor", "windsurf", "zed", "nvim", "vim", "python", "python3"]


@probe(id="dev.toolchain_presence", level="L1", family=DEV, tier="T0", collect="core")
def dev_toolchain_presence(h, facts):
    """which() for node/python/rust/go/java/native/AI CLIs + off-PATH install dirs. Git Bash by path only."""
    found, stubs = [], []
    for t in _TOOLCHAIN:
        p = _which(t)
        if not p:
            continue
        low = p.lower()
        if "\\windowsapps\\" in low and os.path.basename(low) in ("python.exe", "python3.exe"):
            try:
                if os.path.getsize(p) == 0:
                    stubs.append(t); continue
            except OSError:
                pass
        found.append(t)
    U, LA, RA, PF = _U(h), _LA(h), _RA(h), _PF()
    off = {k: _isdir(p) for k, p in [
        ("cargo_home", os.path.join(U, ".cargo")), ("rustup", os.path.join(U, ".rustup")), ("go_root", os.path.join(PF, "Go")),
        ("uv_pythons", os.path.join(RA, "uv", "python")), ("python_org", os.path.join(LA, "Programs", "Python")),
        ("dotnet_sdk", os.path.join(PF, "dotnet", "sdk")), ("java_adoptium", os.path.join(PF, "Eclipse Adoptium")),
        ("java_oracle", os.path.join(PF, "Java")), ("cuda", os.path.join(PF, "NVIDIA GPU Computing Toolkit", "CUDA")),
        ("llvm", os.path.join(PF, "LLVM")), ("android_sdk", os.path.join(LA, "Android", "Sdk")),
        ("miniconda", os.path.join(U, "miniconda3")), ("anaconda", os.path.join(U, "anaconda3")),
        ("pyenv", os.path.join(U, ".pyenv")), ("nvm_windows", os.path.join(RA, "nvm")),
        ("visual_studio", os.path.join(PF, "Microsoft Visual Studio")),
        ("vs_installer", os.path.join(_PF86(), "Microsoft Visual Studio", "Installer"))]}
    return {"present": True, "on_path": found, "store_stubs": stubs,
            "git_bash": os.path.exists(os.path.join(PF, "Git", "bin", "bash.exe")),
            "off_path": sorted(k for k, v in off.items() if v)}


@probe(id="dev.uv_tools", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_uv_tools(h, facts):
    """uv tools and uv-managed Pythons (dir listing), ~/.local/bin entries; no uv spawn."""
    RA, U = _RA(h), _U(h)
    tools = _ls(os.path.join(RA, "uv", "tools"), 200)
    py = _ls(os.path.join(RA, "uv", "python"), 200)
    lb = _ls(os.path.join(U, ".local", "bin"), 300)
    if tools is None and py is None and lb is None:
        return {"present": False}
    return {"present": True, "tools": sorted(t for t in (tools or []) if not t.startswith("."))[:60],
            "pythons": sorted(p for p in (py or []) if p.startswith(("cpython", "pypy")))[:30],
            "local_bin_count": len(lb or []), "local_bin": sorted(lb or [])[:40]}


@probe(id="dev.wsl", level="L1", family=DEV, tier="T0", collect="core")
def dev_wsl(h, facts):
    """WSL distros from the HKCU Lxss key (no wsl.exe), vhdx size, WSL package version, .wslconfig presence."""
    base = r"Software\Microsoft\Windows\CurrentVersion\Lxss"
    k = _open(winreg.HKEY_CURRENT_USER, base)
    distros = []
    if k is not None:
        default = _rv(k, "DefaultDistribution")
        for sub in _subkeys(k, 100):
            dk = _open(winreg.HKEY_CURRENT_USER, base + "\\" + sub)
            if dk is None:
                continue
            bp = (_rv(dk, "BasePath") or "").replace("\\\\?\\", "")
            vhd = None
            for fn in ("ext4.vhdx", "ext4.VHDX"):
                try:
                    vhd = round(os.path.getsize(os.path.join(bp, fn)) / 2**30, 2); break
                except OSError:
                    pass
            distros.append({"name": _rv(dk, "DistributionName"), "version": _rv(dk, "Version"),
                            "default": sub == default, "vhdx_gib": vhd})
    un = _uninst_match(h, r"^Windows Subsystem for Linux")
    appx = _appx_match(h, r"^MicrosoftCorporationII\.WindowsSubsystemForLinux$")
    pkg = (un[0]["version"] if un else None) or (appx[0]["version"] if appx else None) or \
        _file_version(os.path.join(_PF(), "WSL", "wsl.exe"), kind="file")
    inbox = os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe"))
    if not (distros or pkg or k is not None):
        return {"present": False, "inbox_launcher": inbox}
    return {"present": True, "distros": distros, "distro_count": len(distros), "wsl_package_version": pkg,
            "inbox_launcher": inbox, "wslconfig": os.path.exists(os.path.join(_U(h), ".wslconfig"))}


@probe(id="dev.wt_profiles", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_wt_profiles(h, facts):
    """Windows Terminal settings.json: profile count, source kinds, default kind, customisation counts."""
    LA = _LA(h)
    out = {}
    for label, p in [("stable", os.path.join(LA, "Packages", "Microsoft.WindowsTerminal_8wekyb3d8bbwe", "LocalState", "settings.json")),
                     ("preview", os.path.join(LA, "Packages", "Microsoft.WindowsTerminalPreview_8wekyb3d8bbwe", "LocalState", "settings.json")),
                     ("unpackaged", os.path.join(LA, "Microsoft", "Windows Terminal", "settings.json"))]:
        j = _read_json(p, jsonc=True)
        if not isinstance(j, dict):
            continue
        profs = j.get("profiles", {})
        plist = profs.get("list", []) if isinstance(profs, dict) else (profs or [])
        dflt = j.get("defaultProfile")
        kinds = collections.Counter()
        dkind = None
        for pr in plist:
            if not isinstance(pr, dict):
                continue
            src = pr.get("source") or ""
            if src:
                kind = src.split(".")[-1]
            else:
                exe = os.path.basename((pr.get("commandline") or "").split(" ")[0]).lower()
                kind = {"powershell.exe": "WindowsPowerShell", "cmd.exe": "CommandPrompt", "pwsh.exe": "PowerShell",
                        "wsl.exe": "WSL", "ssh.exe": "ssh"}.get(exe, "custom" if exe else "unknown")
            kinds[kind] += 1
            if pr.get("guid") == dflt:
                dkind = kind
        out[label] = {"profiles": len(plist), "visible": sum(1 for x in plist if isinstance(x, dict) and not x.get("hidden")),
                      "kinds": dict(kinds), "default_kind": dkind, "mtime": _mtime(p),
                      "actions": len(j.get("actions", []) or []) + len(j.get("keybindings", []) or []),
                      "schemes": len(j.get("schemes", []) or []), "theme_set": bool(j.get("theme"))}
    return {"present": bool(out), **out}


_AI_EXT = re.compile(r"copilot|continue|cline|claude|anthropic|openai|chatgpt|codeium|windsurf|tabnine|amazonq|gemini|cody|sourcegraph|supermaven|roo|kilo|ollama|aider|hermes", re.I)


def _ext_ids(d):
    j = _read_json(os.path.join(d, "extensions.json"))
    ids = []
    if isinstance(j, list):
        ids = [((e.get("identifier") or {}).get("id") or "").lower() for e in j if isinstance(e, dict)]
    if not any(ids):
        ids = [m.group(1).lower() for x in (_ls(d) or []) for m in [re.match(r"^([\w\-]+\.[\w\-]+)-\d", x)] if m]
    return sorted({i for i in ids if i})


@probe(id="editor.vscode", level="L2", family=DEV, tier="T2", collect="core", gate="dev.toolchain_presence")
def editor_vscode(h, facts):
    """VS Code-family editors: install, extension ids (AI ones flagged), workspaces opened, MCP/chat state; other editors."""
    U, LA, RA = _U(h), _LA(h), _RA(h)
    eds = {"vscode": ([os.path.join(LA, "Programs", "Microsoft VS Code"), os.path.join(_PF(), "Microsoft VS Code")],
                      os.path.join(U, ".vscode", "extensions"), os.path.join(RA, "Code", "User")),
           "vscode_insiders": ([os.path.join(LA, "Programs", "Microsoft VS Code Insiders")],
                               os.path.join(U, ".vscode-insiders", "extensions"), os.path.join(RA, "Code - Insiders", "User")),
           "cursor": ([os.path.join(LA, "Programs", "cursor")], os.path.join(U, ".cursor", "extensions"), os.path.join(RA, "Cursor", "User")),
           "windsurf": ([os.path.join(LA, "Programs", "Windsurf")], os.path.join(U, ".windsurf", "extensions"), os.path.join(RA, "Windsurf", "User")),
           "vscodium": ([os.path.join(LA, "Programs", "VSCodium")], os.path.join(U, ".vscode-oss", "extensions"), os.path.join(RA, "VSCodium", "User")),
           "kiro": ([os.path.join(LA, "Programs", "Kiro")], os.path.join(U, ".kiro", "extensions"), os.path.join(RA, "Kiro", "User"))}
    out = {}
    for name, (inst, extd, user) in eds.items():
        installed = any(_isdir(p) for p in inst)
        if not (installed or _isdir(extd) or _isdir(user)):
            continue
        ids = _ext_ids(extd) if _isdir(extd) else []
        e = {"installed": installed, "extensions": len(ids), "extension_ids": ids[:80],
             "ai_extensions": [i for i in ids if _AI_EXT.search(i)]}
        if _isdir(user):
            ws = _ls(os.path.join(user, "workspaceStorage"), 5000)
            e["workspaces_opened"] = len(ws) if ws is not None else 0
            e["settings_json"] = os.path.exists(os.path.join(user, "settings.json"))
            e["keybindings_json"] = os.path.exists(os.path.join(user, "keybindings.json"))
            e["profiles"] = len(_ls(os.path.join(user, "profiles")) or [])
            mj = _read_json(os.path.join(user, "mcp.json"), jsonc=True)
            e["mcp_servers"] = len((mj.get("servers") or mj.get("mcpServers") or {})) if isinstance(mj, dict) else 0
            e["chat_sessions"] = len(glob.glob(os.path.join(user, "globalStorage", "emptyWindowChatSessions", "*"))[:5000]) + \
                len(glob.glob(os.path.join(user, "workspaceStorage", "*", "chatSessions", "*"))[:5000])
            e["user_mtime"] = _mtime(user)
        out[name] = e
    other = {"jetbrains": [x for x in (_ls(os.path.join(RA, "JetBrains")) or []) if re.match(r"^[A-Za-z]+\d{4}\.\d", x)],
             "zed": _isdir(os.path.join(LA, "Zed")) or _isdir(os.path.join(RA, "Zed")),
             "neovim_config": _isdir(os.path.join(LA, "nvim")),
             "vimrc": os.path.exists(os.path.join(U, "_vimrc")) or os.path.exists(os.path.join(U, ".vimrc")),
             "notepadpp": _isdir(os.path.join(_PF(), "Notepad++")),
             "sublime": _isdir(os.path.join(_PF(), "Sublime Text")) or _isdir(os.path.join(RA, "Sublime Text")),
             "visual_studio": [x for x in (_ls(os.path.join(_PF(), "Microsoft Visual Studio")) or []) if x[:2].isdigit()]}
    other = {k: v for k, v in other.items() if v}
    customised = any(e["extensions"] or e.get("settings_json") or e.get("keybindings_json") for e in out.values())
    return {"present": bool(out or other), "extensions": sum(e["extensions"] for e in out.values()),
            "customised": customised, "editors": out, "other": other}


@probe(id="dev.git_global_config", level="L2", family=DEV, tier="T0", collect="extended", gate="dev.git")
def dev_git_global_config(h, facts):
    """~/.gitconfig parsed without spawn: booleans for identity/credential helper/editor + section counts only."""
    p = os.path.join(_U(h), ".gitconfig")
    if not os.path.exists(p):
        return {"present": False}
    sections, keys = collections.Counter(), set()
    cred_kind = None
    cur = None
    with open(p, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f):
            if n > 3000:
                break
            s = line.strip()
            m = re.match(r"^\[\s*([A-Za-z0-9.\-]+)", s)
            if m:
                cur = m.group(1).lower(); sections[cur] += 1; continue
            if "=" in s and cur and not s.startswith(("#", ";")):
                k, v = s.split("=", 1)
                k = k.strip().lower()
                keys.add(f"{cur}.{k}")
                if cur == "credential" and k == "helper" and v.strip():
                    lv = v.lower()
                    cred_kind = next((n for n, rx in (("gh", r"gh(\.exe)?['\"]?\s+auth"), ("manager", r"manager"),
                                                      ("wincred", r"wincred"), ("store", r"^\s*store"),
                                                      ("cache", r"^\s*cache")) if re.search(rx, lv)), "custom")
    return {"present": True, "bytes": os.path.getsize(p),
            "identity_set": "user.name" in keys and "user.email" in keys,
            "identity_name_set": "user.name" in keys, "identity_email_set": "user.email" in keys,
            "credential_helper": cred_kind, "editor_set": "core.editor" in keys,
            "signing_set": "user.signingkey" in keys or "commit.gpgsign" in keys,
            "sections": dict(sections)}


def _version_line(txt):
    for l in txt.splitlines():
        l = l.strip()
        if l:
            return _redact(l, 90)
    return None


@probe(id="dev.toolchain_versions", level="L2", family=DEV, tier="T0", collect="extended", gate="dev.toolchain_presence",
       timeout_ms=12000)
def dev_toolchain_versions(h, facts):
    """--version spawns (parallel, gated on which): node, npm, uv, dotnet, gh, AI CLIs, infra CLIs. Never bash/pwsh."""
    from concurrent.futures import ThreadPoolExecutor
    specs = [("node", ["--version"]), ("npm", ["--version"]), ("uv", ["--version"]), ("gh", ["--version"]),
             ("dotnet", ["--list-sdks"]), ("claude", ["--version"]), ("codex", ["--version"]),
             ("hermes", ["--version"]), ("ollama", ["--version"]), ("docker", ["--version"]),
             ("kubectl", ["version", "--client=true"]), ("tailscale", ["version"]), ("git", ["--version"]),
             ("rustc", ["--version"]), ("go", ["version"]), ("java", ["-version"]), ("deno", ["--version"]),
             ("bun", ["--version"]), ("pnpm", ["--version"])]
    todo = []
    for name, args in specs:
        exe = _which(name)
        if exe and "\\windowsapps\\" not in exe.lower() and os.path.basename(exe).lower() not in ("bash.exe", "pwsh.exe"):
            todo.append((name, [exe] + args))
    env = h.child_env(NO_COLOR="1", HERMES_NO_UPDATE_CHECK="1")

    def one(item):
        name, args = item
        rc, txt, ms = _spawn(h, args, timeout=8, env=env)
        if name == "dotnet":
            sdks = [l.split()[0] for l in txt.splitlines() if re.match(r"^\d", l.strip())]
            rc2, txt2, ms2 = _spawn(h, [args[0], "--list-runtimes"], timeout=8, env=env)
            rts = sorted({" ".join(l.split()[:2]) for l in txt2.splitlines() if re.match(r"^Microsoft\.", l.strip())})
            return name, {"sdks": sdks[:10], "runtimes": rts[:10], "rc": rc, "ms": round(ms + ms2, 1)}
        return name, {"version": _version_line(txt) if rc == 0 or txt else None, "rc": rc, "ms": ms}
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, r in ex.map(one, todo):
            out[name] = r
    return {"present": bool(out), "tools": out}


# ---------------- repos (deep)

_SKIP_DIRS = {"node_modules", "appdata", "$recycle.bin", "system volume information", "windows", "program files",
              "program files (x86)", "programdata", "__pycache__", ".venv", "venv", "site-packages", ".cache",
              "cache", "temp", "tmp", "msocache", "recovery", "perflogs", "onedrivetemp"}


def _is_reparse(e):
    try:
        return bool(e.stat(follow_symlinks=False).st_file_attributes & 0x400) or e.is_symlink()
    except (OSError, AttributeError):
        return e.is_symlink()


def _repos(h):
    def build():
        deadline = time.perf_counter() + 8.0
        U, LA = _U(h), _LA(h)
        hits, seen = [], set()
        visited = [0]

        def walk(root, maxdepth, skip_appdata=True):
            stack = [(root, 0)]
            while stack and time.perf_counter() < deadline and visited[0] < 60000:
                d, depth = stack.pop()
                visited[0] += 1
                try:
                    with os.scandir(d) as it:
                        ents = list(zip(range(3000), it))
                except OSError:
                    continue
                names = {e.name.lower() for _, e in ents}
                if ".git" in names and os.path.exists(os.path.join(d, ".git", "HEAD")) or \
                        (".git" in names and os.path.isfile(os.path.join(d, ".git"))):
                    rp = os.path.normcase(os.path.realpath(d))
                    if rp not in seen:
                        seen.add(rp); hits.append(d)
                    continue
                if depth >= maxdepth:
                    continue
                for _, e in ents:
                    n = e.name.lower()
                    if n in _SKIP_DIRS and (skip_appdata or n != "appdata"):
                        continue
                    try:
                        if not e.is_dir(follow_symlinks=False) or _is_reparse(e):
                            continue
                    except OSError:
                        continue
                    stack.append((e.path, depth + 1))

        walk(U, 5)
        walk(LA, 3, skip_appdata=False)
        try:
            with os.scandir("C:\\") as it:
                tops = [e.path for e in it if e.is_dir(follow_symlinks=False) and e.name.lower() not in _SKIP_DIRS
                        and e.name.lower() != "users" and not _is_reparse(e)]
        except OSError:
            tops = []
        for t in tops[:60]:
            walk(t, 3)
        repos = []
        for d in hits[:200]:
            if _is_operator_path(h, d):
                cls = "operator"
            else:
                rel = os.path.relpath(d, U) if os.path.normcase(d).startswith(os.path.normcase(U)) else None
                first = rel.split(os.sep, 1)[0] if rel else ""
                if os.path.normcase(d).startswith(os.path.normcase(LA)) or first.lower() == "appdata" or first.startswith("."):
                    cls = "tool_managed"
                else:
                    cls = "user_area"
            gd = os.path.join(d, ".git")
            remotes, branch, head_mtime, local_email = [], None, None, None
            if os.path.isdir(gd):
                try:
                    with open(os.path.join(gd, "HEAD"), encoding="utf-8", errors="replace") as f:
                        hd = f.read(200).strip()
                    branch = hd[16:] if hd.startswith("ref: refs/heads/") else "detached"
                except OSError:
                    pass
                head_mtime = _mtime(os.path.join(gd, "logs", "HEAD")) or _mtime(os.path.join(gd, "index"))
                try:
                    with open(os.path.join(gd, "config"), encoding="utf-8", errors="replace") as f:
                        cfg = f.read(100_000)
                    remotes = re.findall(r"^\s*url\s*=\s*(\S+)", cfg, re.M)
                    m = re.search(r"^\s*email\s*=\s*(\S+)", cfg, re.M)
                    local_email = m.group(1).lower() if m else None
                except OSError:
                    pass
            markers = [m for m in ("package.json", "pyproject.toml", "uv.lock", "Cargo.toml", "go.mod", "pom.xml",
                                   "build.gradle", "Dockerfile", "flake.nix", "AGENTS.md", "CLAUDE.md", ".mcp.json")
                       if os.path.exists(os.path.join(d, m))]
            repos.append({"path": d, "class": cls, "branch": branch, "head_mtime": head_mtime, "remotes": remotes,
                          "local_email": local_email, "markers": markers})
        return {"repos": repos, "visited_dirs": visited[0], "timed_out": time.perf_counter() >= deadline}
    return _memo(h, "repos", build)


def _remote_parse(url):
    m = re.match(r"^(?:\w+://)?(?:[^@/]+@)?([^/:]+)[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    return (m.group(1).lower(), m.group(2), m.group(3)) if m else (None, None, None)


@probe(id="dev.repos", level="L2", family=DEV, tier="T2", collect="deep", gate="dev.git", timeout_ms=15000)
def dev_repos(h, facts):
    """Bounded repo walk (home d5, LOCALAPPDATA d3, C:\\ tops d3): class, markers, freshness. Repo names are T2."""
    r = _repos(h)
    repos = r["repos"]
    by_cls = collections.Counter(x["class"] for x in repos)
    listed = [{"name": os.path.basename(x["path"]), "class": x["class"], "branch": x["branch"],
               "head_mtime": x["head_mtime"], "markers": x["markers"]} for x in repos if x["class"] != "operator"]
    return {"present": bool(repos), "count": len(repos), "by_class": dict(by_cls), "repos": listed[:50],
            "visited_dirs": r["visited_dirs"], "walk_timed_out": r["timed_out"]}


@probe(id="dev.repos.remotes", level="L2", family=DEV, tier="T2", collect="deep", gate="dev.git", timeout_ms=15000)
def dev_repos_remotes(h, facts):
    """Remote hosts (counts) and org/repo names for non-operator repos (T2)."""
    hosts, orgs = collections.Counter(), set()
    for x in _repos(h)["repos"]:
        if x["class"] == "operator":
            continue
        for u in x["remotes"]:
            host, org, name = _remote_parse(u)
            if host:
                hosts[host] += 1
                orgs.add(f"{host}/{org}/{name}")
    return {"present": bool(hosts), "hosts": dict(hosts), "repos": sorted(orgs)[:50]}


def _commits(h):
    def build():
        git = _which("git")
        if not git:
            return None
        ids = set()
        env_ids = os.environ.get("UIL_GIT_IDENTITY", "")
        ids |= {x.strip().lower() for x in env_ids.split(",") if x.strip()}
        gc = os.path.join(_U(h), ".gitconfig")
        try:
            with open(gc, encoding="utf-8", errors="replace") as f:
                ids |= {m.lower() for m in re.findall(r"^\s*email\s*=\s*(\S+)", f.read(100_000), re.M)}
        except OSError:
            pass
        res = {"identity_sources": {"env": bool(env_ids), "global": False, "repo_local": 0}, "by_class": {},
               "hours_local": collections.Counter(), "tz_offsets": collections.Counter(), "weekday": collections.Counter(),
               "month": collections.Counter(), "repos_with_own": {}}
        res["identity_sources"]["global"] = bool(ids - {x.strip().lower() for x in env_ids.split(",") if x.strip()})
        deadline = time.perf_counter() + 10.0
        for x in _repos(h)["repos"]:
            if x["class"] == "operator" or time.perf_counter() > deadline:
                continue
            rid = set(ids)
            if x["local_email"]:
                rid.add(x["local_email"]); res["identity_sources"]["repo_local"] += 1
            rc, txt, _ms = _spawn(h, [git, "-C", x["path"], "log", "--all", "--no-merges", "--format=%ae%x09%an%x09%at%x09%ai",
                                   "-n", "100000"], timeout=10)
            if rc != 0:
                continue
            mine = total = 0
            for line in txt.splitlines():
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                total += 1
                if parts[0].lower() in rid or parts[1].lower() in rid:
                    mine += 1
                    try:
                        t = dt.datetime.fromtimestamp(int(parts[2]))
                        res["hours_local"][t.hour] += 1
                        res["weekday"][t.strftime("%a")] += 1
                        res["month"][t.strftime("%Y-%m")] += 1
                    except (ValueError, OSError):
                        pass
                    res["tz_offsets"][parts[3].strip()[-5:]] += 1
            c = res["by_class"].setdefault(x["class"], [0, 0])
            c[0] += mine; c[1] += total
            if mine:
                res["repos_with_own"][x["class"]] = res["repos_with_own"].get(x["class"], 0) + 1
        return res
    return _memo(h, "commits", build)


@probe(id="dev.repos.my_commits", level="L2", family=DEV, tier="T1", collect="deep", gate="dev.git", timeout_ms=20000)
def dev_repos_my_commits(h, facts):
    """Authored-commit counts [mine, all] by repo class, matched on-host to an identity set (never emitted)."""
    c = _commits(h)
    if not c:
        return None
    return {"present": bool(c["by_class"]), "by_class_mine_all": c["by_class"], "identity_sources": c["identity_sources"],
            "repos_with_own_commits": c["repos_with_own"].get("user_area", 0),
            "repos_with_own_commits_by_class": c["repos_with_own"],
            "note": "clone history includes upstream; tool_managed counts are not work on this box"}


@probe(id="dev.repos.commit_hours", level="L2", family=DEV, tier="T1", collect="deep", gate="dev.git", timeout_ms=20000)
def dev_repos_commit_hours(h, facts):
    """Host-timezone hour histogram, bands, weekday, month and author-TZ offsets of the user's commits."""
    c = _commits(h)
    if not c or not c["hours_local"]:
        return None
    hrs = c["hours_local"]
    bands = {f"{a:02d}-{a + 5:02d}": sum(hrs.get(x, 0) for x in range(a, a + 6)) for a in (0, 6, 12, 18)}
    return {"present": True, "commits": sum(hrs.values()), "hours": {str(k): v for k, v in sorted(hrs.items())},
            "bands": bands, "top_hours": [k for k, _ in hrs.most_common(4)], "weekday": dict(c["weekday"]),
            "month": dict(sorted(c["month"].items())), "tz_offsets": dict(c["tz_offsets"].most_common(6))}


@probe(id="dev.shell_history", level="L2", family=DEV, tier="T1", collect="deep", gate="dev.path_entries")
def dev_shell_history(h, facts):
    """Shell history files: line count and last-write only (text never emitted)."""
    out = {}
    psr = os.path.join(_RA(h), "Microsoft", "Windows", "PowerShell", "PSReadLine")
    files = [("psreadline:" + f, os.path.join(psr, f)) for f in (_ls(psr) or []) if f.lower().endswith(".txt")]
    files += [(f, os.path.join(_U(h), f)) for f in (".bash_history", ".zsh_history", ".python_history", ".node_repl_history")]
    for label, p in files:
        try:
            st = os.stat(p)
            if st.st_size > 50_000_000:
                out[label] = {"bytes": st.st_size, "lines": None, "mtime": _day(st.st_mtime)}
                continue
            with open(p, "rb") as f:
                n = sum(1 for l in f if l.strip())
            out[label] = {"lines": n, "bytes": st.st_size, "mtime": _day(st.st_mtime)}
        except OSError:
            continue
    return {"present": bool(out), **out}
