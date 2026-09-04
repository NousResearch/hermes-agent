"""Server registry — per-language LSP server definitions.

Each :class:`ServerDef` matches files (by extension or basename for
extensionless files like ``Dockerfile``), resolves a project root, and
assembles the spawn command.  Auto-installation lives in
:mod:`agent.lsp.install`; nothing here probes binaries until a file in
that language is actually edited.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.lsp.workspace import nearest_root

logger = logging.getLogger("agent.lsp.servers")

# LSP languageId for ``textDocument/didOpen``, as language → extensions.  A few
# servers (typescript-language-server, vue-language-server) refuse wrong IDs.
_EXTS_BY_LANGUAGE: Dict[str, Sequence[str]] = {
    "python": (".py", ".pyi"),
    "typescript": (".ts", ".mts", ".cts"),
    "typescriptreact": (".tsx",),
    "javascript": (".js", ".mjs", ".cjs"),
    "javascriptreact": (".jsx",),
    "vue": (".vue",), "svelte": (".svelte",), "astro": (".astro",),
    "go": (".go",), "rust": (".rs",),
    "ruby": (".rb", ".rake", ".gemspec", ".ru"),
    "c": (".c", ".h"),
    "cpp": (".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"),
    "csharp": (".cs", ".csx"), "fsharp": (".fs", ".fsi", ".fsx"),
    "swift": (".swift",), "java": (".java",), "kotlin": (".kt", ".kts"),
    "yaml": (".yaml", ".yml"), "json": (".json",), "jsonc": (".jsonc",),
    "lua": (".lua",), "php": (".php",), "prisma": (".prisma",), "dart": (".dart",),
    "ocaml": (".ml", ".mli"),
    "shellscript": (".sh", ".bash", ".zsh"),
    "terraform": (".tf", ".tfvars"),
    "latex": (".tex",), "bibtex": (".bib",), "gleam": (".gleam",),
    "clojure": (".clj", ".cljc", ".edn"), "clojurescript": (".cljs",),
    "nix": (".nix",), "typst": (".typ", ".typc"), "haskell": (".hs", ".lhs"),
    "julia": (".jl",), "elixir": (".ex", ".exs"), "zig": (".zig", ".zon"),
    "dockerfile": (".dockerfile",),
    "powershell": (".ps1", ".psm1", ".psd1"),
}
LANGUAGE_BY_EXT: Dict[str, str] = {ext: lang for lang, exts in _EXTS_BY_LANGUAGE.items() for ext in exts}

_SpawnFn = Callable[[str, "ServerContext"], Optional["SpawnSpec"]]
_RootFn = Callable[[str, str], Optional[str]]


@dataclass
class SpawnSpec:
    """Result of resolving a server for a file (``None`` means skip)."""
    command: List[str]
    workspace_root: str
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    seed_diagnostics_on_first_push: bool = False


@dataclass
class ServerDef:
    """One language server: ``resolve_root(file, ws)`` → per-server root or ``None`` to skip;
    ``build_spawn(root, ctx)`` → :class:`SpawnSpec` or ``None`` when the binary can't be found."""
    server_id: str
    extensions: Tuple[str, ...]
    resolve_root: _RootFn
    build_spawn: _SpawnFn
    seed_first_push: bool = False
    description: str = ""
    # Server handles ``workspace/didChangeWorkspaceFolders``: one process
    # serves every project root (git worktrees included) as extra
    # workspaceFolders instead of one process per root.
    multi_root: bool = False

    def matches(self, file_path: str) -> bool:
        return _file_ext_or_basename(file_path) in self.extensions


@dataclass
class ServerContext:
    """User policy passed into :meth:`ServerDef.build_spawn` (install strategy, overrides)."""
    workspace_root: str
    install_strategy: str = "auto"  # "auto" | "manual" | "off"
    binary_overrides: Dict[str, List[str]] = field(default_factory=dict)
    env_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    init_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---- helpers ----

def _file_ext_or_basename(path: str) -> str:
    """Lower-cased extension, or the full basename for extensionless files (``Dockerfile``)."""
    base = os.path.basename(path)
    return os.path.splitext(base)[1].lower() or base


def _which(*names: str) -> Optional[str]:
    """Return the full path of the first command found on PATH."""
    return next((p for n in names if (p := shutil.which(n))), None)


def _root_or_workspace(file_path: str, workspace: str, markers: Sequence[str], excludes: Sequence[str] = ()) -> Optional[str]:
    """``nearest_root`` with workspace fallback; ``None`` iff an exclude marker hit."""
    ceiling = os.path.dirname(workspace) if workspace else None
    found = nearest_root(file_path, markers, excludes=excludes, ceiling=ceiling)
    if found is None and excludes and nearest_root(file_path, markers, ceiling=ceiling) is not None:
        # None is ambiguous with excludes configured: a hit without them means
        # the exclude fired (gated off); otherwise fall back to the workspace.
        return None
    return found or workspace


def _markers_root(markers: Optional[Sequence[str]], excludes: Sequence[str] = ()) -> _RootFn:
    """Root resolver over marker files; ``None`` markers means "always the workspace root"."""
    if markers is None:
        return lambda fp, ws: ws
    return lambda fp, ws: _root_or_workspace(fp, ws, markers, excludes=excludes)


def _find_binary(ctx: ServerContext, server_id: str, which: Sequence[str], install_pkg: Optional[str]) -> Optional[str]:
    """Config override → PATH → (optional) auto-install; ``None`` when nothing resolves."""
    override = ctx.binary_overrides.get(server_id)
    bin_path = override[0] if override and override[0] and os.path.exists(override[0]) else _which(*which)
    if bin_path is None and install_pkg is not None:
        from agent.lsp.install import try_install
        bin_path = try_install(install_pkg, ctx.install_strategy)
    return bin_path


def _make_spec(root: str, ctx: ServerContext, server_id: str, command: List[str],
               base_init: Optional[Dict[str, Any]] = None, seed: bool = False) -> SpawnSpec:
    init = ctx.init_overrides.get(server_id, {}) if base_init is None else {**base_init, **ctx.init_overrides.get(server_id, {})}
    return SpawnSpec(command, root, root, env=ctx.env_overrides.get(server_id, {}),
                     initialization_options=init, seed_diagnostics_on_first_push=seed)


def _simple_spawn(server_id: str, which: Sequence[str], args: Sequence[str] = (),
                  install_pkg: Optional[str] = None, base_init: Optional[Dict[str, Any]] = None,
                  seed: bool = False) -> _SpawnFn:
    """Build a spawn function for the common single-binary server shape."""
    def build(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
        bin_path = _find_binary(ctx, server_id, which, install_pkg)
        return None if bin_path is None else _make_spec(root, ctx, server_id, [bin_path, *args], base_init, seed)
    return build


# ---- bespoke spawn builders ----

def _spawn_pyright(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _find_binary(ctx, "pyright", ("pyright-langserver", "pyright"), "pyright")
    if bin_path is None:
        return None
    # If we got the cli ``pyright``, the langserver is its sibling.
    if os.path.basename(bin_path) in {"pyright", "pyright.exe"}:
        sibling = os.path.join(os.path.dirname(bin_path), "pyright-langserver")
        if os.path.exists(sibling):
            bin_path = sibling
    # Point pyright at the project venv; its default "python on PATH" rarely is.
    py = _detect_python(root)
    return _make_spec(root, ctx, "pyright", [bin_path, "--stdio"], {"python": {"pythonPath": py}} if py else {})


def _detect_python(root: str) -> Optional[str]:
    venvs = [v for v in (os.environ.get("VIRTUAL_ENV"), os.path.join(root, ".venv"), os.path.join(root, "venv")) if v]
    paths = (os.path.join(v, sub) for v in venvs for sub in ("bin/python", "bin/python3", "Scripts/python.exe"))
    return next((p for p in paths if os.path.exists(p)), None)


_warned_once: set = set()


def _warn_once(key: str, message: str) -> None:
    """Log ``message`` at WARNING the first time ``key`` is seen in this process."""
    if key not in _warned_once:
        _warned_once.add(key)
        logger.warning(message)


def _spawn_bash_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _find_binary(ctx, "bash-language-server", ("bash-language-server",), "bash-language-server")
    if bin_path is None:
        return None
    # bash-language-server delegates diagnostics to shellcheck; without it the
    # server runs but never reports anything.  Warn once so the gap is visible.
    if _which("shellcheck") is None:
        _warn_once("shellcheck", "bash-language-server: shellcheck not found on PATH — diagnostics will be empty "
                   "until shellcheck is installed (apt: shellcheck, brew: shellcheck, scoop: shellcheck).")
    return _make_spec(root, ctx, "bash-language-server", [bin_path, "start"])


def _find_pses_bundle(ctx: ServerContext) -> Optional[str]:
    """Locate the PowerShellEditorServices bundle dir (release zip, manual install).  Resolution order:
    ``lsp.servers.powershell.command[0]`` when a directory, ``init_overrides["powershell"]["bundlePath"]``,
    ``PSES_BUNDLE_PATH`` env, then ``<HERMES_HOME>/lsp/PowerShellEditorServices``."""
    from hermes_constants import get_hermes_home
    override = ctx.binary_overrides.get("powershell")
    init = ctx.init_overrides.get("powershell", {})
    candidates = [
        override[0] if override else None,
        str(init["bundlePath"]) if isinstance(init, dict) and init.get("bundlePath") else None,
        os.environ.get("PSES_BUNDLE_PATH"),
        os.path.join(str(get_hermes_home()), "lsp", "PowerShellEditorServices"),
    ]
    for cand in filter(None, candidates):
        # Accept either the bundle root or the inner module dir.
        if os.path.isfile(os.path.join(cand, "PowerShellEditorServices", "Start-EditorServices.ps1")):
            return cand
        if os.path.isfile(os.path.join(cand, "Start-EditorServices.ps1")):
            return os.path.dirname(cand)
    return None


_PSES_MISSING_MSG = (
    "powershell: pwsh found but the PowerShellEditorServices bundle is missing. Download the release zip from "
    "https://github.com/PowerShell/PowerShellEditorServices/releases, extract it, and either set "
    "lsp.servers.powershell.command to the bundle path or unzip it to <HERMES_HOME>/lsp/PowerShellEditorServices."
)


def _spawn_powershell_es(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    """Spawn PowerShellEditorServices: needs a ``pwsh``/``powershell`` host plus the module bundle."""
    pwsh = _which("pwsh", "powershell")
    if pwsh is None:
        return None
    bundle = _find_pses_bundle(ctx)
    if bundle is None:
        _warn_once("pses-bundle", _PSES_MISSING_MSG)
        return None
    start_script = os.path.join(bundle, "PowerShellEditorServices", "Start-EditorServices.ps1")
    # PSES writes connection info to the session details file on startup.
    session_dir = hermes_lsp_session_dir()
    inner = (
        f"& '{start_script}' -BundledModulesPath '{bundle}' "
        f"-LogPath '{os.path.join(session_dir, 'pses.log')}' "
        f"-SessionDetailsPath '{os.path.join(session_dir, f'pses-session-{os.getpid()}.json')}' "
        f"-FeatureFlags @() -AdditionalModules @() "
        f"-HostName Hermes -HostProfileId hermes -HostVersion 1.0.0 -Stdio -LogLevel Normal"
    )
    return SpawnSpec(
        [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", inner],
        root, root, env=ctx.env_overrides.get("powershell", {}),
        initialization_options={k: v for k, v in ctx.init_overrides.get("powershell", {}).items() if k != "bundlePath"},
    )


def hermes_lsp_session_dir() -> str:
    """Return (and create) the dir for PSES session/log scratch files."""
    from hermes_constants import get_hermes_home
    d = os.path.join(str(get_hermes_home()), "lsp", "pses")
    os.makedirs(d, exist_ok=True)
    return d


# ---- the registry ----

_JS_MARKERS = ["package-lock.json", "bun.lockb", "bun.lock", "pnpm-lock.yaml", "yarn.lock", "package.json", "tsconfig.json"]
_DENO_EXCLUDES = ["deno.json", "deno.jsonc"]
_root_typescript = _markers_root(_JS_MARKERS, _DENO_EXCLUDES)


def _server(server_id: str, extensions: Tuple[str, ...], description: str, *,
            markers: Optional[Sequence[str]] = None, excludes: Sequence[str] = (),
            resolve_root: Optional[_RootFn] = None, build_spawn: Optional[_SpawnFn] = None,
            which: Sequence[str] = (), args: Sequence[str] = (), install_pkg: Optional[str] = None,
            base_init: Optional[Dict[str, Any]] = None, seed: bool = False,
            multi_root: bool = False) -> ServerDef:
    """Registry entry factory: defaults to marker-based root + single-binary spawn."""
    return ServerDef(
        server_id, extensions,
        resolve_root or _markers_root(markers, excludes),
        build_spawn or _simple_spawn(server_id, which or (server_id,), args, install_pkg, base_init, seed),
        seed_first_push=seed, description=description, multi_root=multi_root,
    )


SERVERS: List[ServerDef] = [
    ServerDef(
        server_id="pyright",
        extensions=(".py", ".pyi"),
        resolve_root=_root_python,
        build_spawn=_spawn_pyright,
        multi_root=True,
        description="Python — Microsoft pyright",
    ),
    ServerDef(
        server_id="typescript",
        extensions=(".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"),
        resolve_root=_root_typescript,
        build_spawn=_spawn_typescript,
        seed_first_push=True,
        description="JavaScript/TypeScript — typescript-language-server",
    ),
    ServerDef(
        server_id="vue-language-server",
        extensions=(".vue",),
        resolve_root=_root_typescript,
        build_spawn=_spawn_vue,
        description="Vue.js — @vue/language-server",
    ),
    ServerDef(
        server_id="svelte-language-server",
        extensions=(".svelte",),
        resolve_root=_root_typescript,
        build_spawn=_spawn_svelte,
        description="Svelte — svelte-language-server",
    ),
    ServerDef(
        server_id="astro-language-server",
        extensions=(".astro",),
        resolve_root=_root_typescript,
        build_spawn=_spawn_astro,
        description="Astro — @astrojs/language-server",
    ),
    ServerDef(
        server_id="gopls",
        extensions=(".go",),
        resolve_root=_root_go,
        build_spawn=_spawn_gopls,
        description="Go — gopls",
    ),
    ServerDef(
        server_id="rust-analyzer",
        extensions=(".rs",),
        resolve_root=_root_rust,
        build_spawn=_spawn_rust_analyzer,
        description="Rust — rust-analyzer",
    ),
    ServerDef(
        server_id="clangd",
        extensions=(".c", ".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx"),
        resolve_root=_root_clangd,
        build_spawn=_spawn_clangd,
        description="C/C++ — clangd",
    ),
    ServerDef(
        server_id="bash-language-server",
        extensions=(".sh", ".bash", ".zsh", ".ksh"),
        resolve_root=_root_bash,
        build_spawn=_spawn_bash_ls,
        description="Bash — bash-language-server",
    ),
    ServerDef(
        server_id="yaml-language-server",
        extensions=(".yaml", ".yml"),
        resolve_root=_root_yaml,
        build_spawn=_spawn_yaml_ls,
        description="YAML — yaml-language-server",
    ),
    ServerDef(
        server_id="lua-language-server",
        extensions=(".lua",),
        resolve_root=_root_lua,
        build_spawn=_spawn_lua_ls,
        description="Lua — lua-language-server",
    ),
    ServerDef(
        server_id="intelephense",
        extensions=(".php",),
        resolve_root=_root_php,
        build_spawn=_spawn_intelephense,
        description="PHP — intelephense",
    ),
    ServerDef(
        server_id="ocaml-lsp",
        extensions=(".ml", ".mli"),
        resolve_root=_root_ocaml,
        build_spawn=_spawn_ocamllsp,
        description="OCaml — ocaml-lsp",
    ),
    ServerDef(
        server_id="dockerfile-ls",
        extensions=(".dockerfile", "Dockerfile"),
        resolve_root=_root_docker,
        build_spawn=_spawn_dockerfile_ls,
        description="Dockerfile — dockerfile-language-server-nodejs",
    ),
    ServerDef(
        server_id="terraform-ls",
        extensions=(".tf", ".tfvars"),
        resolve_root=_root_terraform,
        build_spawn=_spawn_terraform_ls,
        description="Terraform — terraform-ls",
    ),
    ServerDef(
        server_id="dart",
        extensions=(".dart",),
        resolve_root=_root_dart,
        build_spawn=_spawn_dart,
        description="Dart — built-in language server",
    ),
    ServerDef(
        server_id="haskell-language-server",
        extensions=(".hs", ".lhs"),
        resolve_root=_root_haskell,
        build_spawn=_spawn_haskell_ls,
        description="Haskell — haskell-language-server",
    ),
    ServerDef(
        server_id="julia",
        extensions=(".jl",),
        resolve_root=_root_julia,
        build_spawn=_spawn_julia,
        description="Julia — LanguageServer.jl",
    ),
    ServerDef(
        server_id="clojure-lsp",
        extensions=(".clj", ".cljs", ".cljc", ".edn"),
        resolve_root=_root_clojure,
        build_spawn=_spawn_clojure_lsp,
        description="Clojure — clojure-lsp",
    ),
    ServerDef(
        server_id="nixd",
        extensions=(".nix",),
        resolve_root=_root_nix,
        build_spawn=_spawn_nixd,
        description="Nix — nixd",
    ),
    ServerDef(
        server_id="zls",
        extensions=(".zig", ".zon"),
        resolve_root=_root_zig,
        build_spawn=_spawn_zls,
        description="Zig — zls",
    ),
    ServerDef(
        server_id="gleam",
        extensions=(".gleam",),
        resolve_root=lambda fp, ws: _root_or_workspace(fp, ws, ["gleam.toml"]),
        build_spawn=_spawn_gleam,
        description="Gleam — built-in language server",
    ),
    ServerDef(
        server_id="elixir-ls",
        extensions=(".ex", ".exs"),
        resolve_root=_root_elixir,
        build_spawn=_spawn_elixir_ls,
        description="Elixir — elixir-ls",
    ),
    ServerDef(
        server_id="prisma",
        extensions=(".prisma",),
        resolve_root=_root_prisma,
        build_spawn=_spawn_prisma,
        description="Prisma — built-in language server",
    ),
    ServerDef(
        server_id="kotlin-language-server",
        extensions=(".kt", ".kts"),
        resolve_root=_root_kotlin,
        build_spawn=_spawn_kotlin_ls,
        description="Kotlin — kotlin-language-server",
    ),
    ServerDef(
        server_id="jdtls",
        extensions=(".java",),
        resolve_root=_root_java,
        build_spawn=_spawn_jdtls,
        description="Java — Eclipse JDT Language Server",
    ),
    ServerDef(
        server_id="powershell",
        extensions=(".ps1", ".psm1", ".psd1"),
        resolve_root=_root_powershell,
        build_spawn=_spawn_powershell_es,
        description="PowerShell — PowerShellEditorServices (manual bundle)",
    ),
]


def find_server_for_file(file_path: str) -> Optional[ServerDef]:
    """Return the registry entry that handles ``file_path``, or None."""
    return next((srv for srv in SERVERS if srv.matches(file_path)), None)


def language_id_for(path: str) -> str:
    """Return the LSP languageId to send in didOpen for ``path``."""
    return LANGUAGE_BY_EXT.get(_file_ext_or_basename(path), "plaintext")


__all__ = ["ServerDef", "ServerContext", "SpawnSpec", "SERVERS", "find_server_for_file", "language_id_for", "LANGUAGE_BY_EXT"]
