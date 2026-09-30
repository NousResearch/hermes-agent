"""macOS user-level probes: dev.

Registration only at import time. Shared helpers and registration wrappers live in darwin_apps.
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import os
import re
import shutil
import time

from userscan.specs import apps_dev as ad
from userscan.specs import apps_dev_devenv as ad_devenv
from userscan.specs.darwin_apps import (DEV, _AS, _LIB, _U, _app, _brew_prefix, _cfg, _data, _day, _ex,
    _extra_bins, _fact, _is_op, _isdir, _ls, _mirror, _mp, _mtime, _newest_mtime, _npm_roots, _read_json,
    _spawn, _which)

# ================================================================== dev

@_mp("apps.pkg_managers", level="L1", family=DEV, tier="T0", collect="core")
def apps_pkg_managers(h, facts):
    """Package managers: Homebrew, MacPorts, nix, npm/pnpm/bun/yarn, cargo, go, uv, pipx, mas."""
    U = _U(h)
    b = {"brew": bool(_brew_prefix()), "macports": _ex("/opt/local/bin/port"),
         "nix": _ex("/nix", os.path.join(U, ".nix-profile")), "npm": bool(_which(h, "npm")), "pnpm": bool(_which(h, "pnpm")),
         "bun": bool(_which(h, "bun")), "yarn": bool(_which(h, "yarn")), "cargo": bool(_which(h, "cargo")),
         "go": bool(_which(h, "go")), "uv": bool(_which(h, "uv")), "pipx": bool(_which(h, "pipx")),
         "mas": bool(_which(h, "mas")), "fnm": bool(_which(h, "fnm")), "nvm": _isdir(os.path.join(U, ".nvm")),
         "pkgx": bool(_which(h, "pkgx")), "mise": bool(_which(h, "mise")), "asdf": _isdir(os.path.join(U, ".asdf"))}
    return {"present": True, **b, "managers": sorted(k for k, v in b.items() if v)}


def _ls_pkgs(root, cap=500):
    names = _ls(root, cap)
    if names is None:
        return None
    pk = []
    for n in names:
        if n.startswith("@"):
            pk += [f"{n}/{s}" for s in (_ls(os.path.join(root, n), 100) or []) if not s.startswith(".")]
        elif not n.startswith("."):
            pk.append(n)
    return sorted(pk)


@_mp("dev.npm_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
def dev_npm_globals(h, facts):
    """Global JS packages by manager (npm roots incl. Homebrew/fnm/nvm, pnpm, bun, yarn); directory listings, no spawn."""
    U = _U(h)
    out, allp = {}, set()
    for r in _npm_roots(h):
        pk = [p for p in (_ls_pkgs(r) or []) if p not in ("npm", "corepack")]
        if pk:
            key = "npm_user" if r.startswith(U) else "npm_system"
            out.setdefault(key, []).extend(pk)
            allp.update(pk)
    for key, r in (("pnpm", os.path.join(U, "Library", "pnpm", "global", "5", "node_modules")),
                   ("bun", os.path.join(U, ".bun", "install", "global", "node_modules")),
                   ("yarn", _cfg(h, "yarn", "global", "node_modules"))):
        pk = _ls_pkgs(r)
        if pk:
            out[key] = pk
            allp.update(pk)
    if not out:
        return {"present": False}
    return {"present": True, "count": len(allp), "packages": sorted(allp)[:80],
            "by_manager": {k: len(set(v)) for k, v in out.items()}}


@_mp("dev.lang_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
def dev_lang_globals(h, facts):
    """cargo-installed crates (~/.cargo/.crates2.json) and ~/go/bin binaries; names only."""
    U = _U(h)
    out = {}
    cj = _read_json(os.path.join(U, ".cargo", ".crates2.json"))
    if isinstance(cj, dict) and isinstance(cj.get("installs"), dict):
        out["cargo"] = sorted({k.split(" ")[0] for k in cj["installs"]})[:60]
    elif _isdir(os.path.join(U, ".cargo", "bin")):
        out["cargo_bin"] = sorted(n for n in _ls(os.path.join(U, ".cargo", "bin")) or [] if not n.startswith("."))[:60]
    gb = _ls(os.path.join(os.environ.get("GOPATH") or os.path.join(U, "go"), "bin"))
    if gb:
        out["go_bin"] = sorted(gb)[:60]
    if not out:
        return {"present": False}
    return {"present": True, "count": sum(len(v) for v in out.values()), **out}


@_mp("dev.uv_tools", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_uv_tools(h, facts):
    """uv tools and uv-managed Pythons (~/.local/share/uv listing, equals `uv tool list`), ~/.local/bin entries."""
    tools = _ls(_data(h, "uv", "tools"), 200)
    py = _ls(_data(h, "uv", "python"), 200)
    lb = _ls(os.path.join(_U(h), ".local", "bin"), 300)
    if tools is None and py is None and lb is None:
        return {"present": False}
    return {"present": True, "tools": sorted(t for t in (tools or []) if not t.startswith("."))[:60],
            "pythons": sorted(p for p in (py or []) if p.startswith(("cpython", "pypy")))[:30],
            "local_bin_count": len(lb or []), "local_bin": sorted(lb or [])[:40]}


@_mp("dev.git", level="L1", family=DEV, tier="T0", collect="core")
def dev_git(h, facts):
    """git binary (no spawn): Xcode Command Line Tools shim, Homebrew git or other."""
    exe = _which(h, "git")
    if not exe:
        return None
    rp = os.path.realpath(exe)
    kind = "homebrew" if "/homebrew/" in rp or "/Cellar/" in rp else "apple_clt" if exe == "/usr/bin/git" else \
        "user" if exe.startswith(_U(h)) else "other"
    return {"present": True, "on_path": bool(shutil.which("git")), "path_kind": kind,
            "clt_installed": _isdir("/Library/Developer/CommandLineTools"), "xdg_config": os.path.exists(_cfg(h, "git", "config"))}


_mirror("dev.git_global_config", ad_devenv.dev_git_global_config)


@_mp("dev.gh_auth_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_gh_auth_presence(h, facts):
    """GitHub CLI hosts.yml presence/size (means gh logged in; never opened)."""
    return h.meta(_cfg(h, "gh", "hosts.yml"))


@_mp("dev.docker_config_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_docker_config_presence(h, facts):
    """~/.docker/config.json presence/size (may hold registry auths; never opened)."""
    return h.meta(os.path.join(_U(h), ".docker", "config.json"))


_mirror("dev.npmrc_presence", ad_devenv.dev_npmrc_presence)

_PATH_TOOLS = [("homebrew", r"^/opt/homebrew/|^/usr/local/(s?bin)$"), ("local_bin", r"/\.local/bin$"), ("home_bin", r"^/Users/[^/]+/bin$"),
               ("bun", r"/\.bun/bin"), ("cargo", r"/\.cargo/bin"), ("go", r"/go/bin"), ("fnm", r"fnm"), ("nvm", r"\.nvm"),
               ("volta", r"\.volta"), ("pnpm", r"pnpm"), ("deno", r"\.deno"), ("nix", r"\.nix-profile|/nix/"),
               ("conda", r"conda|miniforge"), ("pyenv", r"\.pyenv"), ("rye", r"\.rye"), ("hermes", r"hermes"),
               ("lmstudio", r"\.lmstudio"), ("orbstack", r"orbstack"), ("docker", r"Docker\.app"), ("macports", r"^/opt/local/"),
               ("postgres_app", r"Postgres\.app"), ("vscode", r"Visual Studio Code\.app")]


@_mp("dev.path_entries", level="L1", family=DEV, tier="T0", collect="core")
def dev_path_entries(h, facts):
    """PATH entry count and known-tool dir matches for this process (paths not emitted); user bin dirs off PATH."""
    ents = [e for e in os.environ.get("PATH", "").split(":") if e]
    tools = sorted({t for t, rx in _PATH_TOOLS for e in ents if re.search(rx, e)})
    U = _U(h)
    off = [os.path.relpath(d, U) for d in _extra_bins(h) if d.startswith(U) and _isdir(d) and d not in ents]
    helper = len(_ls("/etc/paths.d") or [])
    return {"present": True, "entries": len(ents), "user": sum(1 for e in ents if e.startswith(U)),
            "machine": sum(1 for e in ents if not e.startswith(U)), "tools": tools, "user_bin_dirs_off_path": off,
            "paths_d_entries": helper}


_mirror("dev.ssh_config", ad_devenv.dev_ssh_config)

_TOOLCHAIN = ["node", "npm", "pnpm", "yarn", "bun", "deno", "fnm", "nvm", "volta", "uv", "pipx", "conda", "poetry",
              "rustc", "cargo", "rustup", "go", "dotnet", "java", "javac", "mvn", "gradle", "cmake", "ninja", "make", "clang",
              "gcc", "zig", "swift", "xcodebuild", "xcrun", "pod", "gh", "docker", "podman", "colima", "orb", "kubectl",
              "terraform", "tailscale", "claude", "codex", "hermes", "ollama", "code", "cursor", "windsurf", "zed", "nvim",
              "vim", "emacs", "python3", "python", "tmux", "zellij", "lazygit", "rg", "fd", "fzf", "zoxide", "mcfly",
              "atuin", "btop", "htop", "jq", "ssh", "mosh", "brew", "mas", "gcloud", "aws", "vercel", "wrangler", "flyctl"]


@_mp("dev.toolchain_presence", level="L1", family=DEV, tier="T0", collect="core")
def dev_toolchain_presence(h, facts):
    """which() for language/infra/AI/editor CLIs over PATH plus Homebrew and user tool dirs; off-PATH install dirs."""
    U = _U(h)
    found, user_found = [], []
    for t in _TOOLCHAIN:
        p = _which(h, t)
        if p:
            found.append(t)
            if p.startswith(U):
                user_found.append(t)
    off = {k: _isdir(p) for k, p in [
        ("cargo_home", os.path.join(U, ".cargo")), ("rustup", os.path.join(U, ".rustup")), ("go_root", "/usr/local/go"),
        ("gopath", os.path.join(U, "go")), ("uv_pythons", _data(h, "uv", "python")), ("pyenv", os.path.join(U, ".pyenv")),
        ("miniconda", os.path.join(U, "miniconda3")), ("anaconda", os.path.join(U, "anaconda3")),
        ("miniforge", os.path.join(U, "miniforge3")), ("nvm", os.path.join(U, ".nvm")), ("sdkman", os.path.join(U, ".sdkman")),
        ("android_sdk", _LIB(h, "Android", "sdk")), ("xcode", "/Applications/Xcode.app"),
        ("xcode_derived_data", _LIB(h, "Developer", "Xcode", "DerivedData")), ("clt", "/Library/Developer/CommandLineTools"),
        ("dotnet", os.path.join(U, ".dotnet")), ("rye", os.path.join(U, ".rye"))]}
    return {"present": True, "on_path": found, "user_installed": user_found, "off_path": sorted(k for k, v in off.items() if v)}


@_mp("dev.toolchain_versions", level="L2", family=DEV, tier="T0", collect="extended", gate="dev.toolchain_presence",
     timeout_ms=12000)
def dev_toolchain_versions(h, facts):
    """--version spawns (parallel, gated on which) for runtimes, package managers, AI CLIs and infra CLIs.
    `hermes --version` and `brew --version` are excluded (both start slow interpreters); versions come from files."""
    from concurrent.futures import ThreadPoolExecutor
    specs = [("node", ["--version"]), ("npm", ["--version"]), ("uv", ["--version"]), ("gh", ["--version"]),
             ("git", ["--version"]), ("docker", ["--version"]), ("bun", ["--version"]), ("pnpm", ["--version"]),
             ("deno", ["--version"]), ("rustc", ["--version"]), ("cargo", ["--version"]), ("go", ["version"]),
             ("python3", ["--version"]), ("claude", ["--version"]), ("codex", ["--version"]), ("ollama", ["--version"]),
             ("tmux", ["-V"]), ("nvim", ["--version"]), ("swift", ["--version"]), ("tailscale", ["version"])]
    todo = [(n, [_which(h, n)] + a) for n, a in specs if _which(h, n)]
    if "git" in dict(todo) and not _isdir("/Library/Developer/CommandLineTools") and not _isdir("/Applications/Xcode.app"):
        todo = [t for t in todo if t[0] != "git"]

    def one(item):
        name, args = item
        rc, txt, ms = _spawn(h, args, timeout=3)
        line = next((l.strip() for l in txt.splitlines() if l.strip()), None)
        return name, {"version": ad._redact(line, 90) if line else None, "rc": rc, "ms": ms}
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, r in ex.map(one, todo):
            out[name] = r
    return {"present": bool(out), "tools": out}


_EDITORS = {
    # name: (app bundle names, extension dir(s), Application Support user dir)
    "vscode": (["Visual Studio Code"], [".vscode/extensions"], "Code"),
    "vscode_insiders": (["Visual Studio Code - Insiders"], [".vscode-insiders/extensions"], "Code - Insiders"),
    "cursor": (["Cursor"], [".cursor/extensions"], "Cursor"),
    "windsurf": (["Windsurf"], [".windsurf/extensions"], "Windsurf"),
    "vscodium": (["VSCodium"], [".vscode-oss/extensions"], "VSCodium"),
    "kiro": (["Kiro"], [".kiro/extensions"], "Kiro"),
    "trae": (["Trae"], [".trae/extensions"], "Trae"),
}


@_mp("editor.vscode", level="L2", family=DEV, tier="T2", collect="core", gate="dev.toolchain_presence")
def editor_vscode(h, facts):
    """VS Code family: extension ids (AI flagged), workspaces opened, MCP/chat; other editors (JetBrains, Zed, Xcode...)."""
    U = _U(h)
    out = {}
    for name, (apps, extds, userd) in _EDITORS.items():
        b = _app(h, *apps)
        extd = next((os.path.join(U, e) for e in extds if _isdir(os.path.join(U, e))), None)
        user = _AS(h, userd, "User") if userd else None
        if not (b or extd or (user and _isdir(user))):
            continue
        ids = ad_devenv._ext_ids(extd) if extd else []
        e = {"installed": bool(b), "version": (b or {}).get("version"), "extensions": len(ids), "extension_ids": ids[:80],
             "ai_extensions": [i for i in ids if ad_devenv._AI_EXT.search(i)]}
        if user and _isdir(user):
            ws = _ls(os.path.join(user, "workspaceStorage"), 5000)
            e["workspaces_opened"] = len(ws) if ws is not None else 0
            e["settings_json"] = os.path.exists(os.path.join(user, "settings.json"))
            e["keybindings_json"] = os.path.exists(os.path.join(user, "keybindings.json"))
            e["profiles"] = len(_ls(os.path.join(user, "profiles")) or [])
            mj = _read_json(os.path.join(user, "mcp.json"), jsonc=True)
            e["mcp_servers"] = len((mj.get("servers") or mj.get("mcpServers") or {})) if isinstance(mj, dict) else 0
            e["chat_sessions"] = len(glob.glob(os.path.join(user, "workspaceStorage", "*", "chatSessions", "*"))[:5000])
            e["user_mtime"] = _mtime(user)
        out[name] = e
    other = {"jetbrains": [x for x in (_ls(_AS(h, "JetBrains")) or []) if re.match(r"^[A-Za-z]+\d{4}\.\d", x)],
             "zed": bool(_app(h, "Zed")) or _isdir(_cfg(h, "zed")), "xcode": bool(_app(h, "Xcode")),
             "neovim_config": _isdir(_cfg(h, "nvim")), "vimrc": _ex(os.path.join(U, ".vimrc"), os.path.join(U, ".vim")),
             "emacs": _ex(os.path.join(U, ".emacs.d"), os.path.join(U, ".emacs"), _cfg(h, "emacs")),
             "helix": _isdir(_cfg(h, "helix")), "sublime": _isdir(_AS(h, "Sublime Text")) or bool(_app(h, "Sublime Text")),
             "nova": bool(_app(h, "Nova")), "bbedit": bool(_app(h, "BBEdit"))}
    other = {k: v for k, v in other.items() if v}
    customised = any(e["extensions"] or e.get("settings_json") or e.get("keybindings_json") for e in out.values()) or \
        bool(other.get("neovim_config") or other.get("emacs"))
    return {"present": bool(out or other), "extensions": sum(e["extensions"] for e in out.values()),
            "customised": customised, "editors": out, "other": other}


# ---------------- repos (deep)

_REPO_SKIP = {"node_modules", ".cache", "__pycache__", ".venv", "venv", "site-packages", ".npm", ".rustup", ".cargo",
              ".bun", ".vscode", ".cursor", ".trash", "target", "dist", "build", ".next", ".git", "go", ".gradle", ".m2",
              ".nvm", ".pyenv", "miniconda3", "anaconda3", ".docker", ".ollama", ".orbstack", "library", "applications",
              "movies", "music", "pictures", ".rye", ".lmstudio", ".colima"}


def _repos_mac(h):
    """Bounded repo walk: home depth 5, 60k dirs, 8 s; ~/Library and media folders skipped;
    classes user_area / tool_managed (dot dirs) / operator."""
    def build():
        deadline = time.perf_counter() + 8.0
        U = _U(h)
        hits, seen, visited = [], set(), [0]
        stack = [(U, 0)]
        while stack and time.perf_counter() < deadline and visited[0] < 60000:
            d, depth = stack.pop()
            visited[0] += 1
            try:
                with os.scandir(d) as it:
                    ents = list(zip(range(3000), it))
            except OSError:
                continue
            names = {e.name for _, e in ents}
            if ".git" in names and (os.path.exists(os.path.join(d, ".git", "HEAD")) or os.path.isfile(os.path.join(d, ".git"))):
                rp = os.path.realpath(d)
                if rp not in seen:
                    seen.add(rp)
                    hits.append(d)
                continue
            if depth >= 5:
                continue
            for _, e in ents:
                if e.name.lower() in _REPO_SKIP or e.name.startswith(".Trash"):
                    continue
                try:
                    if not e.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                stack.append((e.path, depth + 1))
        repos = []
        for d in hits[:300]:
            rel = os.path.relpath(d, U)
            cls = "operator" if _is_op(h, d) else "tool_managed" if rel.startswith(".") else "user_area"
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
                                   "build.gradle", "Dockerfile", "flake.nix", "Package.swift", "AGENTS.md", "CLAUDE.md", ".mcp.json")
                       if os.path.exists(os.path.join(d, m))]
            if any(n.endswith(".xcodeproj") for n in (_ls(d, 300) or [])):
                markers.append("xcodeproj")
            repos.append({"path": d, "class": cls, "branch": branch, "head_mtime": head_mtime, "remotes": remotes,
                          "local_email": local_email, "markers": markers})
        return {"repos": repos, "visited_dirs": visited[0], "timed_out": time.perf_counter() >= deadline}
    return build


def _commits_mac(h):
    def build():
        git = _which(h, "git")
        if not git:
            return None
        env_ids = {x.strip().lower() for x in os.environ.get("UIL_GIT_IDENTITY", "").split(",") if x.strip()}
        ids = set(env_ids)
        for gc in (os.path.join(_U(h), ".gitconfig"), _cfg(h, "git", "config")):
            try:
                with open(gc, encoding="utf-8", errors="replace") as f:
                    ids |= {m.lower() for m in re.findall(r"^\s*email\s*=\s*(\S+)", f.read(100_000), re.M)}
            except OSError:
                pass
        res = {"identity_sources": {"env": bool(env_ids), "global": bool(ids - env_ids), "repo_local": 0}, "by_class": {},
               "hours_local": collections.Counter(), "tz_offsets": collections.Counter(), "weekday": collections.Counter(),
               "month": collections.Counter(), "repos_with_own": {}}
        # Newest HEAD first, so when the deadline cuts the scan it drops the stalest repos, not a random set.
        deadline = time.perf_counter() + 10.0
        res["repos_scanned"], res["timed_out"] = 0, False
        for x in sorted(ad_devenv._repos(h)["repos"], key=lambda r: str(r.get("head_mtime") or ""), reverse=True):
            if x["class"] == "operator":
                continue
            if time.perf_counter() > deadline:
                res["timed_out"] = True
                break
            rid = set(ids)
            if x["local_email"]:
                rid.add(x["local_email"])
                res["identity_sources"]["repo_local"] += 1
            rc, txt, _ms = _spawn(h, [git, "-C", x["path"], "log", "--all", "--no-merges",
                                   "--format=%ae%x09%an%x09%at%x09%ai", "-n", "100000"], timeout=max(0.5, deadline - time.perf_counter()))
            if rc != 0:
                continue
            res["repos_scanned"] += 1
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
            c[0] += mine
            c[1] += total
            if mine:
                res["repos_with_own"][x["class"]] = res["repos_with_own"].get(x["class"], 0) + 1
        return res
    return build


def _prime_repos(h):
    ad._memo(h, "repos", _repos_mac(h))


def _prime_commits(h):
    _prime_repos(h)
    ad._memo(h, "commits", _commits_mac(h))


# dev.repos and dev.repos.remotes need only the walk; priming commits there made them wait for git log on every repo.
_mirror("dev.repos", ad_devenv.dev_repos, _prime_repos)
_mirror("dev.repos.remotes", ad_devenv.dev_repos_remotes, _prime_repos)
_mirror("dev.repos.my_commits", ad_devenv.dev_repos_my_commits, _prime_commits)
_mirror("dev.repos.commit_hours", ad_devenv.dev_repos_commit_hours, _prime_commits)

_BUILTINS = {"cd", "export", "source", ".", "alias", "unalias", "echo", "exit", "history", "set", "unset", "eval", "exec",
             "type", "which", "for", "if", "while", "sudo", "time", "nohup", "env", "clear", "pwd", "ls", "ll", "la", "fg",
             "bg", "jobs", "kill", "z", "zi", "man", "help", "open", "defaults", "killall"}


def _cmd_name(line, known):
    t = line.strip().split()
    while t and (t[0] in ("sudo", "time", "nohup", "env", "exec") or "=" in t[0]):
        t = t[1:]
    if not t:
        return None
    c = os.path.basename(t[0])
    if not re.fullmatch(r"[A-Za-z0-9._+\-]{1,32}", c):
        return None
    return c if c in _BUILTINS or c in known else None


@_mp("dev.shell_history", level="L2", family=DEV, tier="T1", collect="extended", gate="dev.path_entries")
def dev_shell_history(h, facts):
    """zsh/bash/fish/python/node history: line counts, last write, top command names (resolved executables or
    builtins only), Apple Terminal ~/.zsh_sessions count. Command lines, arguments and paths are never emitted."""
    U = _U(h)
    files = [("zsh", os.environ.get("HISTFILE") or os.path.join(U, ".zsh_history")), ("bash", os.path.join(U, ".bash_history")),
             ("fish", _data(h, "fish", "fish_history")), ("python", os.path.join(U, ".python_history")),
             ("node", os.path.join(U, ".node_repl_history")), ("mcfly_db", _AS(h, "McFly", "history.db")),
             ("atuin_db", _data(h, "atuin", "history.db"))]
    known = set()
    for d in os.environ.get("PATH", "").split(":") + _extra_bins(h):
        known.update(_ls(d, 20000) or [])
    out, cmds, total = {}, collections.Counter(), 0
    seen = set()
    for label, p in files:
        try:
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            st = os.stat(p)
        except OSError:
            continue
        seen.add(rp)
        if label.endswith("_db"):
            out[label] = {"bytes": st.st_size, "mtime": _day(st.st_mtime)}
            continue
        if st.st_size > 50_000_000:
            out[label] = {"bytes": st.st_size, "lines": None, "mtime": _day(st.st_mtime)}
            continue
        n = 0
        with open(p, "rb") as f:
            for raw in f:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if label == "fish":
                    if not line.startswith("- cmd: "):
                        continue
                    line = line[7:]
                elif label == "zsh":
                    line = re.sub(r"^: \d+:\d+;", "", line)
                n += 1
                if label in ("bash", "zsh", "fish"):
                    c = _cmd_name(line, known)
                    if c:
                        cmds[c] += 1
        out[label] = {"lines": n, "bytes": st.st_size, "mtime": _day(st.st_mtime)}
        if label in ("bash", "zsh", "fish"):
            total += n
    zs = _ls(os.path.join(U, ".zsh_sessions"), 20000)
    if zs:
        out["zsh_sessions"] = {"files": sum(1 for x in zs if x.endswith(".history")), "mtime": _day(_newest_mtime([os.path.join(U, ".zsh_sessions")]))}
    if not out:
        return {"present": False}
    return {"present": True, "lines": total, "files": out, "top_commands": dict(cmds.most_common(15)),
            "distinct_commands": len(cmds), "login_shell": os.path.basename(os.environ.get("SHELL", ""))}


def _docker_sock(h):
    for p in (os.path.join(_U(h), ".docker", "run", "docker.sock"), "/var/run/docker.sock",
              os.path.join(_U(h), ".orbstack", "run", "docker.sock"), os.path.join(_U(h), ".colima", "default", "docker.sock")):
        try:
            import stat as _st
            if _st.S_ISSOCK(os.stat(p).st_mode):
                return p
        except OSError:
            continue
    return None


@_mp("dev.docker", level="L1", family=DEV, tier="T0", collect="core")
def dev_docker(h, facts):
    """Docker on macOS: CLI, Docker Desktop / OrbStack / Colima installs, live engine socket (daemon running)."""
    cli = _which(h, "docker")
    desktop = bool(_app(h, "Docker")) or _isdir(_LIB(h, "Group Containers", "group.com.docker"))
    orb = bool(_app(h, "OrbStack")) or _isdir(os.path.join(_U(h), ".orbstack"))
    colima = _isdir(os.path.join(_U(h), ".colima"))
    if not (cli or desktop or orb or colima):
        return None
    sock = _docker_sock(h)
    return {"present": True, "cli": bool(cli), "docker_desktop": desktop, "orbstack": orb, "colima": colima,
            "engine_socket": bool(sock), "socket_access": bool(sock and os.access(sock, os.R_OK | os.W_OK)),
            "podman": bool(_which(h, "podman"))}


def _gb(s):
    m = re.match(r"([\d.]+)\s*([kKMGT]?B)", s or "")
    if not m:
        return 0.0
    return float(m.group(1)) * {"B": 1e-9, "kB": 1e-6, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}.get(m.group(2), 0)


@_mp("dev.docker_runtime", level="L2", family=DEV, tier="T2", collect="extended", gate="dev.docker", timeout_ms=8000)
def dev_docker_runtime(h, facts):
    """`docker ps` names/images and `docker images` repos/sizes only, only when an engine socket is live."""
    d = _fact(facts, "dev.docker") or {}
    cli = _which(h, "docker")
    if not cli or not d.get("socket_access"):
        return {"present": True, "accessible": False, "count": 0}
    rc, ps, ms1 = _spawn(h, [cli, "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.RunningFor}}"], timeout=5)
    rc2, im, ms2 = _spawn(h, [cli, "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}"], timeout=5)
    if rc != 0 and rc2 != 0:
        return {"present": True, "accessible": False, "count": 0}
    running = [dict(zip(("name", "image", "up"), l.split("\t"))) for l in ps.splitlines() if l.strip()]
    images = [l.split("\t") for l in im.splitlines() if l.strip()]
    return {"present": True, "accessible": True, "count": len(running),
            "running": [{"name": r.get("name"), "image": (r.get("image") or "").split("@")[0][:80], "up": r.get("up")}
                        for r in running[:40]],
            "images": len(images), "images_gb": round(sum(_gb(x[1]) for x in images if len(x) > 1), 2),
            "image_repos": sorted({x[0].rsplit(":", 1)[0] for x in images})[:40], "ms": round(ms1 + ms2, 1)}


@_mp("dev.multiplexers", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_multiplexers(h, facts):
    """tmux / zellij / screen session counts for this user (session names not emitted); terminal apps installed."""
    uid = os.getuid()
    out = {}
    tdir = os.path.join(os.environ.get("TMUX_TMPDIR", "/private/tmp"), f"tmux-{uid}")
    socks = [os.path.join(tdir, n) for n in (_ls(tdir) or [])]
    tm = _which(h, "tmux")
    if socks or tm:
        n = 0
        for s in socks[:10]:
            rc, txt, _ = _spawn(h, [tm, "-S", s, "ls"], timeout=2) if tm else (None, "", 0)
            if rc == 0:
                n += len([l for l in txt.splitlines() if l.strip()])
        out["tmux"] = {"installed": bool(tm), "sockets": len(socks), "sessions": n,
                       "config": _ex(os.path.join(_U(h), ".tmux.conf"), _cfg(h, "tmux", "tmux.conf"))}
    zj = _which(h, "zellij")
    # no-tmp: ok — zellij puts its sockets under TMPDIR, which falls back to /tmp
    tmp = os.environ.get("TMPDIR") or "/tmp"
    zsocks = [p for p in glob.glob(os.path.join(tmp, "zellij-*", "*", "*")) if not os.path.isdir(p)]
    zres = glob.glob(_LIB(h, "Caches", "org.Zellij-Contributors.Zellij", "*", "session_info", "*"))
    if zj or zsocks or zres:
        out["zellij"] = {"installed": bool(zj), "live_sessions": len(zsocks), "resurrectable": len(zres),
                         "config": _isdir(_cfg(h, "zellij"))}
    terms = [n for n in ("iTerm", "Ghostty", "WezTerm", "Alacritty", "kitty", "Warp", "cmux", "Hyper", "Tabby")
             if _app(h, n)]
    if terms:
        out["terminal_apps"] = terms
    if not out:
        return {"present": False}
    return {"present": True, "count": sum(v.get("sessions", 0) + v.get("live_sessions", 0) for v in out.values()
                                          if isinstance(v, dict)), **out}
