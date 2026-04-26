#!/usr/bin/env python3
"""Cloud storage CLI — thin wrapper around rclone for multiple providers."""

import argparse
import json
import os
import subprocess
import sys

def _run_rclone(args, timeout=300, check=False):
    cmd = ["rclone"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)
        return result
    except FileNotFoundError:
        raise RuntimeError("rclone is not installed. Install with: pkg install rclone (Termux) or apt install rclone (Linux)")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"rclone timed out after {timeout}s")

def _parse_lsf(output):
    items = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) >= 3 and parts[0].isdigit():
            size = parts[0]
            modtime = parts[1]
            name = parts[2]
            items.append({"name": name, "size": size, "modified": modtime, "type": "folder" if name.endswith("/") else "file"})
        else:
            name = line
            items.append({"name": name, "type": "folder" if name.endswith("/") else "file"})
    return items

def cmd_upload(source, dest, recursive=False, progress=False, extra_args=None):
    args = ["copyto" if not recursive else "copy", source, dest]
    if recursive:
        args = ["copy", source, dest, "--recursive"]
    if progress:
        args.append("--progress")
    if extra_args:
        args.extend(extra_args)
    args.append("--checksum")
    result = _run_rclone(args, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr.strip()}")
    return {"uploaded": dest, "local": os.path.abspath(source)}

def cmd_download(source, dest, recursive=False, progress=False, extra_args=None):
    args = ["copyto" if not recursive else "copy", source, dest]
    if recursive:
        args = ["copy", source, dest, "--recursive"]
    if progress:
        args.append("--progress")
    if extra_args:
        args.extend(extra_args)
    result = _run_rclone(args, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr.strip()}")
    return {"downloaded": source, "local": os.path.abspath(dest)}

def cmd_list(remote, dirs_only=False, recursive=False, format_json=False):
    args = ["lsf", remote]
    if dirs_only:
        args.append("--dirs-only")
    if recursive:
        args.append("--recursive")
    if not format_json:
        args.append("-l")
    result = _run_rclone(args, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"List failed: {result.stderr.strip()}")
    if format_json:
        return _parse_lsf(result.stdout)
    return result.stdout.strip()

def cmd_sync(source, dest, dry_run=False, extra_args=None):
    args = ["sync", source, dest]
    if dry_run:
        args.append("--dry-run")
    if extra_args:
        args.extend(extra_args)
    result = _run_rclone(args, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Sync failed: {result.stderr.strip()}")
    return {"synced": True, "dry_run": dry_run, "source": source, "dest": dest}

def cmd_delete(remote, recursive=False):
    args = ["delete", remote]
    if recursive:
        args.append("--rmdirs")
    result = _run_rclone(args, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Delete failed: {result.stderr.strip()}")
    return {"deleted": remote}

def cmd_mkdir(remote):
    result = _run_rclone(["mkdir", remote], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Mkdir failed: {result.stderr.strip()}")
    return {"created": remote}

def cmd_search(remote, query, extra_args=None):
    args = ["lsf", remote, "--include", f"*{query}*"]
    if extra_args:
        args.extend(extra_args)
    result = _run_rclone(args, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Search failed: {result.stderr.strip()}")
    return _parse_lsf(result.stdout)

def cmd_about(remote):
    result = _run_rclone(["about", remote, "--json"], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"About failed: {result.stderr.strip()}")
    return json.loads(result.stdout)

def cmd_info(remote):
    result = _run_rclone(["lsjson", remote], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Info failed: {result.stderr.strip()}")
    return json.loads(result.stdout)

def cmd_empty_trash(remote):
    result = _run_rclone(["cleanup", remote], timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Empty trash failed: {result.stderr.strip()}")
    return {"trash_emptied": remote}

def print_json(data):
    print(json.dumps(data, indent=2, default=str))

def print_table(items):
    if not items:
        print("No items found.")
        return
    if isinstance(items, list) and items and isinstance(items[0], dict):
        for item in items:
            t = "D" if item.get("type") == "folder" else "F"
            size = item.get("size", "")
            name = item.get("name", "?")
            mod = item.get("modified", "")[:16]
            print(f"  [{t}] {name:50s} {str(size) or '':>12s}  {mod}")
    elif isinstance(items, dict):
        for k, v in items.items():
            print(f"  {k:25s} {v}")
    else:
        print(items)

def main():
    parser = argparse.ArgumentParser(description="Cloud storage via rclone")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("upload", help="Upload file or folder")
    p_up.add_argument("source", help="Local file or folder path")
    p_up.add_argument("dest", help="Remote destination (e.g. onedrive:Documents/)")
    p_up.add_argument("--recursive", action="store_true")
    p_up.add_argument("--progress", action="store_true")
    p_up.add_argument("--args", default="", help="Extra rclone args, comma-separated")

    p_down = sub.add_parser("download", help="Download file or folder")
    p_down.add_argument("source", help="Remote source")
    p_down.add_argument("dest", help="Local destination path")
    p_down.add_argument("--recursive", action="store_true")
    p_down.add_argument("--progress", action="store_true")
    p_down.add_argument("--args", default="", help="Extra rclone args, comma-separated")

    p_list = sub.add_parser("list", help="List folder contents")
    p_list.add_argument("remote", help="Remote path")
    p_list.add_argument("--recursive", action="store_true")
    p_list.add_argument("--dirs-only", action="store_true")

    p_sync = sub.add_parser("sync", help="Sync source to dest")
    p_sync.add_argument("source", help="Source path")
    p_sync.add_argument("dest", help="Destination path")
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("--args", default="", help="Extra rclone args, comma-separated")

    p_del = sub.add_parser("delete", help="Delete file or folder")
    p_del.add_argument("remote", help="Remote path")
    p_del.add_argument("--recursive", action="store_true")

    p_mkdir = sub.add_parser("mkdir", help="Create folder on remote")
    p_mkdir.add_argument("remote", help="Remote path")

    p_search = sub.add_parser("search", help="Search remote for files")
    p_search.add_argument("remote", help="Remote root to search")
    p_search.add_argument("query", help="Search string / pattern")
    p_search.add_argument("--args", default="", help="Extra rclone args, comma-separated")

    p_about = sub.add_parser("about", help="Show storage usage for remote")
    p_about.add_argument("remote", help="Remote name")

    p_info = sub.add_parser("info", help="Show metadata for file or folder")
    p_info.add_argument("remote", help="Remote path")

    p_trash = sub.add_parser("empty-trash", help="Empty trash for remote")
    p_trash.add_argument("remote", help="Remote name")

    args = parser.parse_args()
    extra = [a.strip() for a in args.args.split(",") if a.strip()] if hasattr(args, "args") and args.args else []

    try:
        if args.command == "upload":
            result = cmd_upload(args.source, args.dest, args.recursive, args.progress, extra)
        elif args.command == "download":
            result = cmd_download(args.source, args.dest, args.recursive, args.progress, extra)
        elif args.command == "list":
            result = cmd_list(args.remote, args.dirs_only, args.recursive, args.format == "json")
        elif args.command == "sync":
            result = cmd_sync(args.source, args.dest, args.dry_run, extra)
        elif args.command == "delete":
            result = cmd_delete(args.remote, args.recursive)
        elif args.command == "mkdir":
            result = cmd_mkdir(args.remote)
        elif args.command == "search":
            result = cmd_search(args.remote, args.query, extra)
        elif args.command == "about":
            result = cmd_about(args.remote)
        elif args.command == "info":
            result = cmd_info(args.remote)
        elif args.command == "empty-trash":
            result = cmd_empty_trash(args.remote)
        else:
            parser.print_help()
            sys.exit(1)
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {e}"}))
        sys.exit(1)

    if args.format == "json":
        print_json(result)
    else:
        if isinstance(result, str):
            print(result)
        elif isinstance(result, list):
            print_table(result)
        else:
            print_table([result]) if isinstance(result, dict) else print(result)

if __name__ == "__main__":
    main()
