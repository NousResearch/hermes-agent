"""Server registry — per-language LSP server definitions.

Each :class:`ServerDef` knows how to:

- match a file by extension (or basename for extensionless files like
  ``Dockerfile``),
- resolve a project root from a file path (often via
  :func:`agent.lsp.workspace.nearest_root`),
- assemble the spawn command (binary, args, env, cwd),
- compute LSP ``initializationOptions``.

Auto-installation is a separate concern handled by
:mod:`agent.lsp.install`.  This module describes WHAT to spawn; the
install module makes the binary appear on PATH if it isn't there.

The full set of servers ships with the package, but most are only
*invoked* when the user actually edits a file in that language.  This
keeps cold-start fast — we don't probe binaries until needed.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.lsp.workspace import nearest_root, normalize_path

logger = logging.getLogger("agent.lsp.servers")

# Language IDs per LSP spec.  Used for ``textDocument/didOpen.languageId``.
# Most servers don't care exactly, but a few (typescript-language-server,
# vue-language-server) refuse files with the wrong ID.
LANGUAGE_BY_EXT: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".ru": "ruby",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".csx": "csharp",
    ".fs": "fsharp",
    ".fsi": "fsharp",
    ".fsx": "fsharp",
    ".swift": "swift",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonc": "jsonc",
    ".lua": "lua",
    ".php": "php",
    ".prisma": "prisma",
    ".dart": "dart",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".sh": "shellscript",
    ".bash": "shellscript",
    ".zsh": "shellscript",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".tex": "latex",
    ".bib": "bibtex",
    ".gleam": "gleam",
    ".clj": "clojure",
    ".cljs": "clojurescript",
    ".cljc": "clojure",
    ".edn": "clojure",
    ".nix": "nix",
    ".typ": "typst",
    ".typc": "typst",
    ".hs": "haskell",
    ".lhs": "haskell",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".zig": "zig",
    ".zon": "zig",
    ".dockerfile": "dockerfile",
}


@dataclass
class SpawnSpec:
    """The result of resolving a server for a file.

    Returned by :meth:`ServerDef.resolve` when a server is applicable
    to a file.  ``None`` is returned instead when the server should
    be skipped (binary missing and auto-install disabled, project
    marker not found, exclude marker hit, etc.).
    """

    command: List[str]
    workspace_root: str
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    seed_diagnostics_on_first_push: bool = False


@dataclass
class ServerDef:
    """Definition of one language server.

    The :func:`resolve_root` callable receives the absolute file path
    plus the workspace root (git worktree) and returns either the
    project-specific root for this server (e.g. the directory
    containing ``pyproject.toml``) or ``None`` to skip.

    The :func:`build_spawn` callable receives the resolved root and
    returns a :class:`SpawnSpec` (or ``None`` if the binary can't be
    found and auto-install isn't configured).
    """

    server_id: str
    extensions: Tuple[str, ...]
    resolve_root: Callable[[str, str], Optional[str]]
    build_spawn: Callable[[str, "ServerContext"], Optional[SpawnSpec]]
    seed_first_push: bool = False
    description: str = ""

    def matches(self, file_path: str) -> bool:
        """Return True iff this server handles ``file_path``."""
        ext = _file_ext_or_basename(file_path)
        return ext in self.extensions


@dataclass
class ServerContext:
    """Context passed into :meth:`ServerDef.build_spawn`.

    Carries the user's auto-install policy, any user-overridden
    binary paths, and helpers the spawn builder needs.  All fields
    are optional; defaults yield "auto-install allowed, no overrides".
    """

    workspace_root: str
    install_strategy: str = "auto"  # "auto" | "manual" | "off"
    binary_overrides: Dict[str, List[str]] = field(default_factory=dict)
    env_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    init_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _file_ext_or_basename(path: str) -> str:
    """Return the lower-cased extension OR full basename for extensionless files.

    Mirrors OpenCode's ``path.parse(file).ext || file`` — files like
    ``Dockerfile`` or ``Makefile`` match by basename, while normal
    files match by extension (``.py``, ``.ts``).
    """
    base = os.path.basename(path)
    _root, ext = os.path.splitext(base)
    if ext:
        return ext.lower()
    return base


def _which(*names: str) -> Optional[str]:
    """Return the full path of the first command found on PATH."""
    for n in names:
        path = shutil.which(n)
        if path:
            return path
    return None


def _root_or_workspace(file_path: str, workspace: str, markers: Sequence[str], excludes: Sequence[str] = ()) -> Optional[str]:
    """Common pattern: try ``nearest_root``, fall back to workspace root.

    Returns ``None`` if an exclude marker matches first (server gated off).
    """
    found = nearest_root(
        file_path,
        markers,
        excludes=excludes,
        ceiling=os.path.dirname(workspace) if workspace else None,
    )
    if found is None and excludes:
        # Distinguish "no marker found" from "exclude hit": when
        # excludes are configured, None means gated off.
        # Re-check without excludes — if still None, we fall back to
        # workspace; if found, the exclude hit and we return None.
        recheck = nearest_root(
            file_path,
            markers,
            ceiling=os.path.dirname(workspace) if workspace else None,
        )
        if recheck is not None:
            return None  # exclude triggered
        return workspace
    return found or workspace


# ---------------------------------------------------------------------------
# per-server spawn builders
# ---------------------------------------------------------------------------


def _spawn_pyright(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "pyright") or _which(
        "pyright-langserver", "pyright"
    )
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("pyright", ctx.install_strategy)
        if bin_path is None:
            return None
    # If we got the cli ``pyright``, the langserver is its sibling.
    base = os.path.basename(bin_path)
    if base in ("pyright", "pyright.exe"):
        sibling = os.path.join(os.path.dirname(bin_path), "pyright-langserver")
        if os.path.exists(sibling):
            bin_path = sibling
    init: Dict[str, Any] = {}
    # Pick the project's venv interpreter if there is one — otherwise
    # pyright defaults to "python on PATH" which is rarely the venv.
    py = _detect_python(root)
    if py:
        init["python"] = {"pythonPath": py}
    if "pyright" in ctx.init_overrides:
        init.update(ctx.init_overrides["pyright"])
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("pyright", {}),
        initialization_options=init,
    )


def _detect_python(root: str) -> Optional[str]:
    candidates = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(os.environ["VIRTUAL_ENV"])
    candidates.extend([os.path.join(root, ".venv"), os.path.join(root, "venv")])
    for v in candidates:
        for sub in ("bin/python", "bin/python3", "Scripts/python.exe"):
            p = os.path.join(v, sub)
            if os.path.exists(p):
                return p
    return None


def _spawn_typescript(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "typescript") or _which("typescript-language-server")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("typescript-language-server", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("typescript", {}),
        initialization_options=ctx.init_overrides.get("typescript", {}),
        seed_diagnostics_on_first_push=True,
    )


def _spawn_gopls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "gopls") or _which("gopls")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("gopls", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("gopls", {}),
        initialization_options=ctx.init_overrides.get("gopls", {}),
    )


def _spawn_rust_analyzer(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "rust-analyzer") or _which("rust-analyzer")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("rust-analyzer", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("rust-analyzer", {}),
        initialization_options=ctx.init_overrides.get("rust-analyzer", {}),
    )


def _spawn_clangd(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "clangd") or _which("clangd")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("clangd", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "--background-index", "--clang-tidy"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("clangd", {}),
        initialization_options=ctx.init_overrides.get("clangd", {}),
    )


def _spawn_bash_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "bash-language-server") or _which("bash-language-server")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("bash-language-server", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "start"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("bash-language-server", {}),
        initialization_options=ctx.init_overrides.get("bash-language-server", {}),
    )


def _spawn_yaml_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "yaml-language-server") or _which("yaml-language-server")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("yaml-language-server", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("yaml-language-server", {}),
        initialization_options=ctx.init_overrides.get("yaml-language-server", {}),
    )


def _spawn_lua_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "lua-language-server") or _which("lua-language-server")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("lua-language-server", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("lua-language-server", {}),
        initialization_options=ctx.init_overrides.get("lua-language-server", {}),
    )


def _spawn_intelephense(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "intelephense") or _which("intelephense")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("intelephense", ctx.install_strategy)
        if bin_path is None:
            return None
    init = {"telemetry": {"enabled": False}}
    init.update(ctx.init_overrides.get("intelephense", {}))
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("intelephense", {}),
        initialization_options=init,
    )


def _spawn_ocamllsp(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "ocaml-lsp") or _which("ocamllsp")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("ocaml-lsp", {}),
        initialization_options=ctx.init_overrides.get("ocaml-lsp", {}),
    )


def _spawn_dockerfile_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "dockerfile-ls") or _which("docker-langserver")
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("dockerfile-language-server-nodejs", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("dockerfile-ls", {}),
        initialization_options=ctx.init_overrides.get("dockerfile-ls", {}),
    )


def _spawn_terraform_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "terraform-ls") or _which("terraform-ls")
    if bin_path is None:
        return None  # terraform-ls is heavy to auto-install; require user
    init = {
        "experimentalFeatures": {
            "prefillRequiredFields": True,
            "validateOnSave": True,
        }
    }
    init.update(ctx.init_overrides.get("terraform-ls", {}))
    return SpawnSpec(
        command=[bin_path, "serve"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("terraform-ls", {}),
        initialization_options=init,
    )


def _spawn_dart(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "dart") or _which("dart")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path, "language-server", "--lsp"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("dart", {}),
        initialization_options=ctx.init_overrides.get("dart", {}),
    )


def _spawn_haskell_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "haskell-language-server") or _which(
        "haskell-language-server-wrapper", "haskell-language-server"
    )
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path, "--lsp"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("haskell-language-server", {}),
        initialization_options=ctx.init_overrides.get("haskell-language-server", {}),
    )


def _spawn_julia(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "julia") or _which("julia")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[
            bin_path,
            "--startup-file=no",
            "--history-file=no",
            "-e",
            "using LanguageServer; runserver()",
        ],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("julia", {}),
        initialization_options=ctx.init_overrides.get("julia", {}),
    )


def _spawn_clojure_lsp(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "clojure-lsp") or _which("clojure-lsp")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path, "listen"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("clojure-lsp", {}),
        initialization_options=ctx.init_overrides.get("clojure-lsp", {}),
    )


def _spawn_nixd(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "nixd") or _which("nixd")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("nixd", {}),
        initialization_options=ctx.init_overrides.get("nixd", {}),
    )


def _spawn_zls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "zls") or _which("zls")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("zls", {}),
        initialization_options=ctx.init_overrides.get("zls", {}),
    )


def _spawn_gleam(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "gleam") or _which("gleam")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path, "lsp"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("gleam", {}),
        initialization_options=ctx.init_overrides.get("gleam", {}),
    )


def _spawn_elixir_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "elixir-ls") or _which("elixir-ls", "language_server.sh")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("elixir-ls", {}),
        initialization_options=ctx.init_overrides.get("elixir-ls", {}),
    )


def _spawn_prisma(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "prisma") or _which("prisma")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path, "language-server"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("prisma", {}),
        initialization_options=ctx.init_overrides.get("prisma", {}),
    )


def _spawn_kotlin_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "kotlin-language-server") or _which(
        "kotlin-language-server"
    )
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("kotlin-language-server", {}),
        initialization_options=ctx.init_overrides.get("kotlin-language-server", {}),
    )


def _spawn_jdtls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    # jdtls has a complex install flow.  We require a manual install
    # for now and look for the wrapper script that the jdtls install
    # produces.
    bin_path = _resolve_override(ctx, "jdtls") or _which("jdtls")
    if bin_path is None:
        return None
    return SpawnSpec(
        command=[bin_path],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("jdtls", {}),
        initialization_options=ctx.init_overrides.get("jdtls", {}),
    )


def _spawn_vue(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "vue-language-server") or _which(
        "vue-language-server"
    )
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("@vue/language-server", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("vue-language-server", {}),
        initialization_options=ctx.init_overrides.get("vue-language-server", {}),
    )


def _spawn_svelte(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "svelte-language-server") or _which(
        "svelteserver", "svelte-language-server"
    )
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("svelte-language-server", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("svelte-language-server", {}),
        initialization_options=ctx.init_overrides.get("svelte-language-server", {}),
    )


def _spawn_astro(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _resolve_override(ctx, "astro-language-server") or _which(
        "astro-ls", "astro-language-server"
    )
    if bin_path is None:
        from agent.lsp.install import try_install
        bin_path = try_install("@astrojs/language-server", ctx.install_strategy)
        if bin_path is None:
            return None
    return SpawnSpec(
        command=[bin_path, "--stdio"],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("astro-language-server", {}),
        initialization_options=ctx.init_overrides.get("astro-language-server", {}),
    )


def _resolve_override(ctx: ServerContext, server_id: str) -> Optional[str]:
    """User can pin a binary path in config."""
    override = ctx.binary_overrides.get(server_id)
    if override and override[0] and os.path.exists(override[0]):
        return override[0]
    return None


# ---------------------------------------------------------------------------
# root resolvers
# ---------------------------------------------------------------------------


def _root_python(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path,
        workspace,
        ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile", "pyrightconfig.json"],
    )


def _root_typescript(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path,
        workspace,
        [
            "package-lock.json",
            "bun.lockb",
            "bun.lock",
            "pnpm-lock.yaml",
            "yarn.lock",
            "package.json",
            "tsconfig.json",
        ],
        excludes=["deno.json", "deno.jsonc"],
    )


def _root_go(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path,
        workspace,
        ["go.work", "go.mod", "go.sum"],
    )


def _root_rust(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["Cargo.toml", "Cargo.lock"])


def _root_ruby(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["Gemfile"])


def _root_clangd(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path,
        workspace,
        ["compile_commands.json", "compile_flags.txt", ".clangd"],
    )


def _root_bash(file_path: str, workspace: str) -> str:
    return workspace


def _root_yaml(file_path: str, workspace: str) -> str:
    return workspace


def _root_lua(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path,
        workspace,
        [".luarc.json", ".luarc.jsonc", ".luacheckrc", ".stylua.toml", "stylua.toml", "selene.toml", "selene.yml"],
    )


def _root_php(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["composer.json", "composer.lock", ".php-version"])


def _root_ocaml(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["dune-project", "dune-workspace", ".merlin", "opam"])


def _root_docker(file_path: str, workspace: str) -> str:
    return workspace


def _root_terraform(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, [".terraform.lock.hcl", "terraform.tfstate"])


def _root_dart(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["pubspec.yaml", "analysis_options.yaml"])


def _root_haskell(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["stack.yaml", "cabal.project", "hie.yaml"])


def _root_julia(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["Project.toml", "Manifest.toml"])


def _root_clojure(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path, workspace, ["deps.edn", "project.clj", "shadow-cljs.edn", "bb.edn", "build.boot"]
    )


def _root_nix(file_path: str, workspace: str) -> str:
    found = nearest_root(file_path, ["flake.nix"])
    return found or workspace


def _root_zig(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["build.zig"])


def _root_elixir(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(file_path, workspace, ["mix.exs", "mix.lock"])


def _root_prisma(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path, workspace, ["schema.prisma", "prisma/schema.prisma"]
    )


def _root_kotlin(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path,
        workspace,
        ["settings.gradle", "settings.gradle.kts", "build.gradle", "build.gradle.kts", "pom.xml"],
    )


def _root_java(file_path: str, workspace: str) -> Optional[str]:
    return _root_or_workspace(
        file_path,
        workspace,
        ["pom.xml", "build.gradle", "build.gradle.kts", ".project", ".classpath", "settings.gradle"],
    )


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


SERVERS: List[ServerDef] = [
    ServerDef(
        server_id="pyright",
        extensions=(".py", ".pyi"),
        resolve_root=_root_python,
        build_spawn=_spawn_pyright,
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
]


def find_server_for_file(file_path: str) -> Optional[ServerDef]:
    """Return the registry entry that handles ``file_path``, or None."""
    for srv in SERVERS:
        if srv.matches(file_path):
            return srv
    return None


def language_id_for(path: str) -> str:
    """Return the LSP languageId to send in didOpen for ``path``."""
    ext = _file_ext_or_basename(path)
    return LANGUAGE_BY_EXT.get(ext, "plaintext")


__all__ = [
    "ServerDef",
    "ServerContext",
    "SpawnSpec",
    "SERVERS",
    "find_server_for_file",
    "language_id_for",
    "LANGUAGE_BY_EXT",
]
