"""@-reference expansion (``@file:``, ``@folder:``, ``@diff``, ``@git:``, ``@url:`` + plugin prefixes)."""

from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent.model_metadata import estimate_tokens_rough
from hermes_cli._subprocess_compat import IS_WINDOWS, harden_git_argv, noninteractive_git_env, windows_hide_flags
from hermes_cli.sizefmt import format_bytes

# ── Plugin context-reference provider API ────────────────────────────────────

# --------------------------------------------------------------------------- Plugin context-reference
# provider API (Issue #26193) ---------------------------------------------------------------------------
BUILTIN_PREFIXES = frozenset({"diff", "staged", "file", "folder", "git", "url"})

_context_reference_providers: dict[str, "ContextReferenceProvider"] = {}


class ContextCompletionItem:
    """A single autocomplete result from a context reference provider."""

    __slots__ = ("text", "display", "meta")

    def __init__(self, text: str, display: str = "", meta: str = "") -> None:
        self.text = text
        self.display = display or text
        self.meta = meta


class ContextReferenceProvider(ABC):
    """Base class for plugin @-prefix providers, registered via ``PluginContext.register_context_reference()``."""

    prefix: str = ""  # e.g. "issue", "channel", "doc"
    description: str = ""  # shown in autocomplete meta column

    @abstractmethod
    async def autocomplete(self, query: str, *, limit: int = 10) -> list[ContextCompletionItem]:
        """Return autocomplete items for the given query string."""

    @abstractmethod
    async def expand(self, target: str) -> str | None:
        """Expand *target* to prompt content.  Return ``None`` to skip."""


def register_context_reference_provider(provider: ContextReferenceProvider) -> None:
    """Register a plugin context reference provider."""
    if not isinstance(provider, ContextReferenceProvider):
        raise TypeError("provider must be a ContextReferenceProvider instance")
    prefix = provider.prefix.lower().strip()
    if not prefix:
        raise ValueError("prefix must be a non-empty string")
    if prefix in BUILTIN_PREFIXES:
        raise ValueError(f"prefix '{prefix}' is reserved for built-in references")
    if prefix in _context_reference_providers:
        raise ValueError(f"prefix '{prefix}' is already registered")
    _context_reference_providers[prefix] = provider


def get_context_reference_providers() -> dict[str, ContextReferenceProvider]:
    """Return a snapshot of all registered plugin providers."""
    return dict(_context_reference_providers)


_QUOTED_REFERENCE_VALUE = r'(?:`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\')'
REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+))"
)
# Plugin fallback: any @<word>:<value> the built-in regex did not claim.
_PLUGIN_REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?P<kind>[a-zA-Z][a-zA-Z0-9_-]*):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+)"
)
# ``@file:`` value: quoted path or bare path, each with an optional ``:start[-end]`` range.
_FILE_VALUE_PATTERN = re.compile(
    r'^(?:(?P<quote>`|"|\')(?P<qpath>.+?)(?P=quote)|(?P<path>.+?))(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$'
)

TRAILING_PUNCTUATION = ",.;!?"
_OPENERS = {")": "(", "]": "[", "}": "{"}
_NEEDS_QUOTING = re.compile(r"""[\s()\[\]{}<>"'`]""")
_SENSITIVE_HOME_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh")
_SENSITIVE_HERMES_DIRS = (Path("skills") / ".hub",)
_SENSITIVE_HOME_FILES = tuple(Path(p) for p in (
    ".ssh/authorized_keys", ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/config", ".bashrc", ".zshrc",
    ".profile", ".bash_profile", ".zprofile", ".netrc", ".pgpass", ".npmrc", ".pypirc",
))
_TEXT_EXTENSIONS = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts")
_FENCE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx",
    ".json": "json", ".md": "markdown", ".sh": "bash", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
}


@dataclass(frozen=True)
class ContextReference:
    raw: str
    kind: str
    target: str
    start: int
    end: int
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class ContextReferenceResult:
    message: str
    original_message: str
    references: list[ContextReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    expanded: bool = False
    blocked: bool = False


UrlFetcher = Callable[[str], str | Awaitable[str]] | None
Expansion = tuple[str | None, str | None]  # (warning, block) — exactly one side is set


def format_reference_value(value: str) -> str:
    """Quote a value so ``REFERENCE_PATTERN`` (bare alternative ``\\S+``) reads it back whole.
    Mirrors ``formatRefValue`` in the desktop's directive-text.tsx."""
    if not _NEEDS_QUOTING.search(value):
        return value
    for quote in ("`", '"', "'"):
        if quote not in value:
            return f"{quote}{value}{quote}"
    return value


def parse_context_references(message: str) -> list[ContextReference]:
    refs: list[ContextReference] = []
    if not message:
        return refs
    for match in REFERENCE_PATTERN.finditer(message):
        kind = match.group("simple") or match.group("kind")
        value = _strip_trailing_punctuation(match.group("value") or "")
        if match.group("simple"):
            target, line_start, line_end = "", None, None
        elif kind == "file":
            target, line_start, line_end = _parse_file_reference_value(value)
        else:
            target, line_start, line_end = _strip_reference_wrappers(value), None, None
        refs.append(ContextReference(match.group(0), kind, target, match.start(), match.end(), line_start, line_end))

    # Second pass: plugin-registered prefixes the built-in pattern missed.
    for match in _PLUGIN_REFERENCE_PATTERN.finditer(message) if _context_reference_providers else ():
        kind = match.group("kind")
        if kind in BUILTIN_PREFIXES or kind not in _context_reference_providers:
            continue
        if any(r.kind == kind and r.start == match.start() for r in refs):
            continue
        target = _strip_reference_wrappers(_strip_trailing_punctuation(match.group("value") or ""))
        refs.append(ContextReference(match.group(0), kind, target, match.start(), match.end()))
    return refs


def preprocess_context_references(
    message: str, *, cwd: str | Path, context_length: int, url_fetcher: UrlFetcher = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    """Sync wrapper; safe both without a loop (CLI) and inside a running loop (gateway)."""
    coro = preprocess_context_references_async(
        message, cwd=cwd, context_length=context_length, url_fetcher=url_fetcher, allowed_root=allowed_root
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def preprocess_context_references_async(
    message: str, *, cwd: str | Path, context_length: int, url_fetcher: UrlFetcher = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    refs = parse_context_references(message)
    if not refs:
        return ContextReferenceResult(message=message, original_message=message)
    cwd_path = Path(cwd).expanduser().resolve()
    # Default root = cwd so @ references cannot escape the workspace unless a caller widens it.
    allowed_root_path = Path(allowed_root).expanduser().resolve() if allowed_root is not None else cwd_path
    # Expand concurrently (each ref is independent; several @url: refs would otherwise
    # serialize web_extract round-trips). gather preserves order, so warnings/blocks
    # are assembled in ref order; the token-budget check runs once afterwards.
    tasks = (_expand_reference(ref, cwd_path, url_fetcher=url_fetcher, allowed_root=allowed_root_path) for ref in refs)
    expanded = await asyncio.gather(*tasks)
    warnings = [warning for warning, _ in expanded if warning]
    blocks = [block for _, block in expanded if block]
    injected_tokens = sum(estimate_tokens_rough(block) for block in blocks)
    result = ContextReferenceResult(
        message=message, original_message=message, references=refs, warnings=warnings, injected_tokens=injected_tokens
    )

    hard_limit = max(1, int(context_length * 0.50))
    soft_limit = max(1, int(context_length * 0.25))
    if injected_tokens > hard_limit:
        warnings.append(f"@ context injection refused: {injected_tokens} tokens exceeds the 50% hard limit ({hard_limit}).")
        result.blocked = True
        return result
    if injected_tokens > soft_limit:
        warnings.append(f"@ context injection warning: {injected_tokens} tokens exceeds the 25% soft limit ({soft_limit}).")

    # The `@file:`/`@folder:` tokens stay where the user typed them: the token IS the
    # reference (clients render it as an inline chip); stripping it left a hole in the
    # sentence and forced the desktop to re-derive refs from the attached block.
    final = message
    if warnings:
        final = f"{final}\n\n--- Context Warnings ---\n" + "\n".join(f"- {warning}" for warning in warnings)
    if blocks:
        final = f"{final}\n\n--- Attached Context ---\n\n" + "\n\n".join(blocks)
    result.message = final.strip()
    result.expanded = bool(blocks or warnings)
    return result


# Git-backed reference kinds -> f(ref) -> git argv (the label is "git " + argv).
_GIT_REFERENCE_ARGS: dict[str, Callable[[ContextReference], list[str]]] = {
    "diff": lambda ref: ["diff"],
    "staged": lambda ref: ["diff", "--staged"],
    "git": lambda ref: ["log", f"-{max(1, min(int(ref.target or '1'), 10))}", "-p"],
}


async def _expand_reference(
    ref: ContextReference, cwd: Path, *, url_fetcher: UrlFetcher = None, allowed_root: Path | None = None
) -> Expansion:
    try:
        if ref.kind in ("file", "folder"):
            return _expand_path_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind in _GIT_REFERENCE_ARGS:
            git_args = _GIT_REFERENCE_ARGS[ref.kind](ref)
            return _expand_git_reference(ref, cwd, git_args, "git " + " ".join(git_args))
        if ref.kind == "url":
            content = await _fetch_url_content(ref.target, url_fetcher=url_fetcher)
            if not content:
                return f"{ref.raw}: no content extracted", None
            return None, f"🌐 {ref.raw} ({estimate_tokens_rough(content)} tokens)\n{content}"
    except Exception as exc:
        return f"{ref.raw}: {exc}", None
    provider = _context_reference_providers.get(ref.kind)
    if provider is not None:
        try:
            plugin_content = await provider.expand(ref.target)
            if plugin_content is not None:
                return None, f"📌 {ref.raw} ({estimate_tokens_rough(plugin_content)} tokens)\n{plugin_content}"
        except Exception as exc:
            return f"{ref.raw}: plugin expansion error: {exc}", None
    return f"{ref.raw}: unsupported reference type", None


def _expand_path_reference(ref: ContextReference, cwd: Path, *, allowed_root: Path | None = None) -> Expansion:
    """``@file:`` / ``@folder:``: resolve, allow-check, then inline text / binary stub / listing."""
    is_folder = ref.kind == "folder"
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: {ref.kind} not found", None
    if not (path.is_dir() if is_folder else path.is_file()):
        return f"{ref.raw}: path is not a {ref.kind}", None
    if is_folder:
        listing = _build_folder_listing(path, cwd)
        return None, f"📁 {ref.raw} ({estimate_tokens_rough(listing)} tokens)\n{listing}"
    if _is_binary_file(path):
        # A bare "not supported" warning was a dead end (the model gave up); the file IS
        # on disk where the agent's tools run, so hand it an actionable block instead.
        return None, _binary_reference_block(ref, path)
    text = path.read_text(encoding="utf-8")
    if ref.line_start is not None:
        text = "\n".join(text.splitlines()[max(ref.line_start - 1, 0):ref.line_end or ref.line_start])
    lang = _FENCE_LANGUAGES.get(path.suffix.lower(), "")
    return None, f"📄 {ref.raw} ({estimate_tokens_rough(text)} tokens)\n```{lang}\n{text}\n```"


def _run_quiet(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None) -> subprocess.CompletedProcess:
    """subprocess.run with captured text output, no stdin, and no console flash on Windows."""
    popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=timeout, stdin=subprocess.DEVNULL, **popen_kwargs, **({} if env is None else {"env": env}))


def _expand_git_reference(ref: ContextReference, cwd: Path, args: list[str], label: str) -> Expansion:
    try:
        # Repo-supplied config/attributes must never execute code (GHSA-7x36-8jrh-v4pw).
        result = _run_quiet(["git", *harden_git_argv(args)], cwd, 30, env=noninteractive_git_env())
    except subprocess.TimeoutExpired:
        return f"{ref.raw}: git command timed out (30s)", None
    if result.returncode != 0:
        return f"{ref.raw}: {(result.stderr or '').strip() or 'git command failed'}", None
    content = result.stdout.strip() or "(no output)"
    return None, f"🧾 {label} ({estimate_tokens_rough(content)} tokens)\n```diff\n{content}\n```"


async def _fetch_url_content(url: str, *, url_fetcher: UrlFetcher = None) -> str:
    content = (url_fetcher or _default_url_fetcher)(url)
    if inspect.isawaitable(content):
        content = await content
    return str(content or "").strip()


async def _default_url_fetcher(url: str) -> str:
    from tools.web_tools import web_extract_tool
    docs = json.loads(await web_extract_tool([url], format="markdown")).get("results", [])
    return str(docs[0].get("content") or docs[0].get("raw_content") or "").strip() if docs else ""


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_path(cwd: Path, target: str, *, allowed_root: Path | None = None) -> Path:
    resolved = (cwd / Path(os.path.expanduser(target))).resolve()  # `/` keeps an absolute target as-is
    if allowed_root is not None and not _is_under(resolved, allowed_root):
        raise ValueError("path is outside the allowed workspace")
    return resolved


def _ensure_reference_path_allowed(path: Path) -> None:
    """Refuse credential/internal paths. Fails CLOSED: the gateway feeds untrusted remote text here."""
    from hermes_constants import get_hermes_home
    home, hermes_home = Path(os.path.expanduser("~")).resolve(), get_hermes_home().resolve()
    blocked_exact = {home / rel for rel in _SENSITIVE_HOME_FILES} | {hermes_home / ".env"}
    blocked_dirs = [home / rel for rel in _SENSITIVE_HOME_DIRS] + [hermes_home / rel for rel in _SENSITIVE_HERMES_DIRS]
    if path in blocked_exact:
        raise ValueError("path is a sensitive credential file and cannot be attached")
    if any(_is_under(path, blocked_dir) for blocked_dir in blocked_dirs):
        raise ValueError("path is a sensitive credential or internal Hermes path and cannot be attached")
    # Anchor to the canonical read deny-list (agent/file_safety.get_read_block_error): the
    # narrow list above never caught auth.json, .anthropic_oauth.json, mcp-tokens/, webhook
    # secrets or project .env files, and it grows automatically with that deny-list.
    try:
        from agent.file_safety import get_read_block_error
        blocked = get_read_block_error(str(path)) is not None
    except ValueError:
        raise
    except Exception:
        # If the canonical lookup fails, falling through would re-open the exact hole this
        # guard closes; a spurious block is recoverable, a leaked credential is not.
        raise ValueError("path could not be verified against the credential deny-list and cannot be attached")
    if blocked:
        raise ValueError("path is a sensitive credential or internal Hermes path and cannot be attached")


def _strip_trailing_punctuation(value: str) -> str:
    stripped = value.rstrip(TRAILING_PUNCTUATION)
    # Drop unbalanced closers so "(see @file:x.py)" does not swallow the ")".
    while stripped.endswith((")", "]", "}")) and stripped.count(stripped[-1]) > stripped.count(_OPENERS[stripped[-1]]):
        stripped = stripped[:-1]
    return stripped


def _strip_reference_wrappers(value: str) -> str:
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'" else value


def _parse_file_reference_value(value: str) -> tuple[str, int | None, int | None]:
    m = _FILE_VALUE_PATTERN.match(value)
    start = m and m.group("start")
    if not start:  # no line range: the whole value is the (possibly quoted) path
        return _strip_reference_wrappers(value), None, None
    return m.group("qpath") or m.group("path"), int(start), int(m.group("end") or start)


def _is_binary_file(path: Path) -> bool:
    mime = mimetypes.guess_type(path.name)[0]
    return bool(mime and not mime.startswith("text/") and not path.name.endswith(_TEXT_EXTENSIONS)) or (
        b"\x00" in path.read_bytes()[:4096]
    )


def _build_folder_listing(path: Path, cwd: Path, limit: int = 200) -> str:
    lines = [f"{path.relative_to(cwd)}/"]
    entries = _iter_visible_entries(path, cwd, limit=limit)
    base_depth = len(path.relative_to(cwd).parts)
    for entry in entries:
        indent = "  " * max(len(entry.relative_to(cwd).parts) - base_depth - 1, 0)
        lines.append(f"{indent}- {entry.name}/" if entry.is_dir() else f"{indent}- {entry.name} ({_file_metadata(entry)})")
    if len(entries) >= limit:
        lines.append("- ...")
    return "\n".join(lines)


def _iter_visible_entries(path: Path, cwd: Path, limit: int) -> list[Path]:
    """Files under ``path`` via ``rg --files`` (honours ignore files), else an os.walk fallback."""
    try:
        rg = _run_quiet(["rg", "--files", str(path.relative_to(cwd))], cwd, 10)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        rg = None
    if rg is not None and rg.returncode == 0:
        output: list[Path] = []
        seen_dirs: set[Path] = set()
        for line in [ln.strip() for ln in rg.stdout.splitlines() if ln.strip()][:limit]:
            full = cwd / Path(line)
            for parent in full.parents:
                if parent == cwd or parent in seen_dirs or path not in {parent, *parent.parents}:
                    continue
                seen_dirs.add(parent)
                output.append(parent)
            output.append(full)
        return sorted({p for p in output if p.exists()}, key=lambda p: (not p.is_dir(), str(p)))
    output = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        files = sorted(f for f in files if not f.startswith("."))
        for name in dirs + files:
            output.append(Path(root) / name)
            if len(output) >= limit:
                return output
    return output


def _binary_reference_block(ref: ContextReference, path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        size = format_bytes(path.stat().st_size)
    except OSError:
        size = "unknown size"
    # Under a container backend the host path dangles inside the sandbox: translate staged
    # files to their auto-mounted cache path; fall back to the host path (local backend /
    # translation failure). Run the idempotent TERMINAL_ENV bridge first so in-process
    # gateways that never bridged terminal.* config still see the active backend.
    try:
        from tools.terminal_tool import _ensure_terminal_env_bridged
        _ensure_terminal_env_bridged()
        from tools.credential_files import to_agent_visible_cache_path
        visible = to_agent_visible_cache_path(str(path))
    except Exception:
        visible = str(path)
    return (
        f"📎 {ref.raw} ({mime}, {size}) — binary file, not inlined as text. "
        f"It is available on disk at `{visible}`. Use your tools to work with it "
        f"(read or convert it, extract its text, or view/render it as needed); "
        f"do not tell the user the file type is unsupported."
    )


def _file_metadata(path: Path) -> str:
    if not _is_binary_file(path):
        try:
            return f"{path.read_text(encoding='utf-8').count(chr(10)) + 1} lines"
        except Exception:
            pass
    return f"{path.stat().st_size} bytes"
