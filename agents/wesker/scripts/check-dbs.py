#!/usr/bin/env python3
import os
for slug, path in {'research':'~/.hermes/kanban/boards/research/kanban.db','ops':'~/.hermes/kanban/boards/ops/kanban.db','default':'~/.hermes/kanban.db'}.items():
    p = os.path.expanduser(path)
    print(f'{slug}: exists={os.path.isfile(p)}, size={os.path.getsize(p) if os.path.isfile(p) else 0}')
