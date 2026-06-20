import sys
p = 'kanban_db.py'
with open(p, 'r') as f:
    s = f.read()

print(len(s), file=sys.stderr)
print('test', file=sys.stderr)