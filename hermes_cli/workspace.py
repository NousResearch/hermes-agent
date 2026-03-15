from __future__ import annotations

from typing import Optional

from rich.console import Console

from agent.workspace import index_workspace_knowledgebase, workspace_list, workspace_retrieve, workspace_search, workspace_status
from hermes_cli.config import load_config


def _console(console: Optional[Console]) -> Console:
    return console or Console()


def _print_status(console: Console) -> None:
    data = workspace_status(load_config())
    if not data.get("success"):
        console.print(f"[bold red]{data.get('error', 'Workspace unavailable')}[/]")
        return
    console.print(f"Workspace root: {data['workspace_root']}")
    console.print(f"Knowledgebase root: {data['knowledgebase_root']}")
    console.print(f"Manifest: {data['manifest_path']}")
    console.print(f"Index DB: {data.get('index_path', '(not built)')}")
    console.print(f"Files: {data['file_count']}")
    console.print(f"Chunks: {data.get('chunk_count', 0)}")
    if data.get('embedding_backend'):
        console.print(f"Embedding backend: {data['embedding_backend']}")
    if data.get('dense_backend'):
        console.print(f"Dense backend: {data['dense_backend']}")
    counts = data.get("category_counts") or {}
    if counts:
        for key in sorted(counts):
            console.print(f"  {key}: {counts[key]}")


def _print_index(console: Console) -> None:
    data = index_workspace_knowledgebase(load_config())
    if not data.get("success"):
        console.print(f"[bold red]{data.get('error', 'Index failed')}[/]")
        return
    console.print(f"Indexed {data['file_count']} files into {data.get('chunk_count', 0)} chunks")
    console.print(f"Manifest: {data['manifest_path']}")
    console.print(f"Index DB: {data['index_path']}")
    if data.get('embedding_backend'):
        console.print(f"Embedding backend: {data['embedding_backend']}")
    if data.get('dense_backend'):
        console.print(f"Dense backend: {data['dense_backend']}")


def _print_list(console: Console, path: str = "", recursive: bool = True, limit: int = 20, offset: int = 0) -> None:
    data = workspace_list(load_config(), relative_path=path, recursive=recursive, limit=limit, offset=offset)
    if not data.get("success"):
        console.print(f"[bold red]{data.get('error', 'List failed')}[/]")
        return
    entries = data.get("entries") or []
    if not entries:
        console.print("No workspace files found.")
        return
    for entry in entries:
        console.print(entry["relative_path"])
    if data.get("total_count", len(entries)) > len(entries):
        console.print(f"[dim]Showing {len(entries)} of {data['total_count']} files[/]")


def _print_search(console: Console, query: str, path: str = "", file_glob: str | None = None, limit: int = 10, offset: int = 0) -> None:
    data = workspace_search(query, load_config(), relative_path=path, file_glob=file_glob, limit=limit, offset=offset)
    if not data.get("success"):
        console.print(f"[bold red]{data.get('error', 'Search failed')}[/]")
        return
    matches = data.get("matches") or []
    if not matches:
        console.print("No matches found.")
        return
    for match in matches:
        console.print(f"{match['relative_path']}:{match['line']}  {match['content']}")
    if data.get("total_count", len(matches)) > len(matches):
        console.print(f"[dim]Showing {len(matches)} of {data['total_count']} matches[/]")


def _print_retrieve(console: Console, query: str, limit: int = 8) -> None:
    data = workspace_retrieve(query, load_config(), limit=limit)
    if not data.get("success"):
        console.print(f"[bold red]{data.get('error', 'Retrieve failed')}[/]")
        return
    results = data.get("results") or []
    if not results:
        console.print("No retrieval results found.")
        return
    if data.get('dense_backend') or data.get('rerank_backend'):
        console.print(f"Dense backend: {data.get('dense_backend', '')}  Rerank backend: {data.get('rerank_backend', '')}")
    for result in results:
        rerank_score = result.get('rerank_score')
        rerank_text = f" rerank={rerank_score:.3f}" if isinstance(rerank_score, (int, float)) else ""
        console.print(f"{result['relative_path']}  [rrf={result['rrf_score']:.4f} dense={result['dense_score']:.3f}{rerank_text}]")
        console.print(result["content"])
        console.print()


def workspace_command(args, console: Optional[Console] = None) -> None:
    console = _console(console)
    action = getattr(args, "workspace_action", None) or "status"
    if action == "status":
        _print_status(console)
    elif action == "index":
        _print_index(console)
    elif action == "list":
        _print_list(
            console,
            path=getattr(args, "path", "") or "",
            recursive=getattr(args, "recursive", True),
            limit=getattr(args, "limit", 20),
            offset=getattr(args, "offset", 0),
        )
    elif action == "search":
        query = getattr(args, "query", "") or ""
        if not query.strip():
            console.print("Usage: hermes workspace search <query>")
            return
        _print_search(
            console,
            query=query,
            path=getattr(args, "path", "") or "",
            file_glob=getattr(args, "file_glob", None),
            limit=getattr(args, "limit", 10),
            offset=getattr(args, "offset", 0),
        )
    elif action == "retrieve":
        query = getattr(args, "query", "") or ""
        if not query.strip():
            console.print("Usage: hermes workspace retrieve <query>")
            return
        _print_retrieve(console, query=query, limit=getattr(args, "limit", 8))
    else:
        console.print(f"[bold red]Unknown workspace action: {action}[/]")


def handle_workspace_slash(cmd: str, console: Optional[Console] = None) -> None:
    console = _console(console)
    parts = cmd.strip().split()
    if parts and parts[0].lower() == "/workspace":
        parts = parts[1:]

    if not parts or parts[0] in {"status", "path"}:
        _print_status(console)
        return

    action = parts[0].lower()
    if action == "index":
        _print_index(console)
        return
    if action == "list":
        path = parts[1] if len(parts) > 1 else ""
        _print_list(console, path=path)
        return
    if action == "search":
        query = " ".join(parts[1:]).strip()
        if not query:
            console.print("Usage: /workspace search <query>")
            return
        _print_search(console, query=query)
        return
    if action == "retrieve":
        query = " ".join(parts[1:]).strip()
        if not query:
            console.print("Usage: /workspace retrieve <query>")
            return
        _print_retrieve(console, query=query)
        return

    console.print("Usage: /workspace [status|index|list [path]|search <query>|retrieve <query>]")
