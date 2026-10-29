#!/usr/bin/env python3
"""Project inventory script — identifies bloat categories in a directory.

Usage: python3 inventory.py <project_root>

Outputs JSON with file counts, sizes, and categorized junk.
Safe: read-only, never modifies files.
"""

import os, sys, json, datetime, hashlib
from collections import defaultdict

def file_hash(path):
    """Fix #4: 用内容哈希替代 (文件名+大小) 做重复检测"""
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def inventory(root, max_depth=4):
    stats = {
        'pycache': [], 'empty': [], 'duplicates': {},
        'exports': [], 'old_backups': [], 'logs_old': [],
        'total_files': 0, 'total_size': 0,
        'py_files': 0, 'py_lines': 0
    }
    
    cutoff_30d = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y%m%d')
    
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, '').count(os.sep)
        if depth > max_depth:
            continue
        dirnames[:] = [d for d in dirnames if d not in ['.git', 'node_modules', '.venv', 'venv']]
        
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(fp)
            except OSError:
                continue
            stats['total_files'] += 1
            stats['total_size'] += size
            
            if size == 0 and f not in ['.gitkeep', '__init__.py']:
                stats['empty'].append(fp)
            
            if f.endswith('.pyc'):
                stats['pycache'].append(fp)
            
            if 'exports' in dirpath and f.endswith(('.xlsx', '.docx', '.pdf')):
                stats['exports'].append(fp)
            
            # Fix #4: 用内容哈希检测真正的重复文件
            content_key = file_hash(fp)
            if content_key and content_key not in stats['duplicates']:
                stats['duplicates'][content_key] = []
            if content_key:
                stats['duplicates'][content_key].append(fp)
            
            if f.endswith('.py'):
                stats['py_files'] += 1
                try:
                    with open(fp) as fh:
                        stats['py_lines'] += len(fh.readlines())
                except:
                    pass
            
            if 'memory' in dirpath and f.endswith('.md'):
                try:
                    date_part = f.replace('.md', '').replace('-', '')
                    if len(date_part) == 8 and date_part.isdigit() and date_part < cutoff_30d:
                        stats['logs_old'].append(fp)
                except:
                    pass
    
    stats['duplicates'] = {k: v for k, v in stats['duplicates'].items() if len(v) > 1}
    
    backup_dir = os.path.join(root, 'backups')
    if os.path.isdir(backup_dir):
        backups = sorted([d for d in os.listdir(backup_dir) if os.path.isdir(os.path.join(backup_dir, d))])
        stats['backup_count'] = len(backups)
    else:
        stats['backup_count'] = 0
    
    return stats

def report(stats):
    print("=" * 60)
    print(f"📦 Project Inventory")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Total size:  {stats['total_size']/1024:.0f} KB")
    print(f"   Python:      {stats['py_files']} files, {stats['py_lines']} lines")
    print("=" * 60)
    
    categories = [
        ("__pycache__ / .pyc", stats['pycache'], "Delete"),
        ("Empty files (0KB)", stats['empty'], "Delete"),
        ("Test exports", stats['exports'], "Delete"),
        ("Duplicate files", list(stats['duplicates'].keys()), "Keep 1, delete rest"),
        ("Old logs (>30d)", stats['logs_old'], "Archive"),
        ("Backups", [f"(count: {stats['backup_count']})"], "Keep 5" if stats['backup_count'] > 5 else "OK"),
    ]
    
    for name, items, action in categories:
        count = len(items) if not (name == "Backups") else stats['backup_count']
        if name == "Duplicate files":
            count = sum(len(v) - 1 for v in stats['duplicates'].values())
        print(f"\n{'🗑️' if action == 'Delete' else '📦'} {name}: {count} → {action}")
        if items and name != "Backups":
            for item in (items if isinstance(items, list) else list(items)[:5]):
                print(f"     {item}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 inventory.py <project_root>")
        sys.exit(1)
    root = os.path.abspath(sys.argv[1])
    stats = inventory(root)
    report(stats)
    json_stats = {k: v for k, v in stats.items() if k != 'duplicates'}
    json_stats['duplicates'] = {f"md5:{k[:12]}": len(v) for k, v in stats['duplicates'].items()}
    print("\n--- JSON ---")
    print(json.dumps(json_stats, indent=2, ensure_ascii=False, default=str))
