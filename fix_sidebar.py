#!/usr/bin/env python3
path = 'apps/desktop/src/app/chat/sidebar/index.tsx'
with open(path) as f: lines = f.read().splitlines(True)
new_lines = []
state = 0
for line in lines:
    if '<<<<<<< HEAD' in line:
        state = 1; continue
    elif '=======' in line and state == 1:
        state = 2; continue
    elif '>>>>>>>' in line and state == 2:
        state = 0; continue
    if state == 1 or state == 0:
        new_lines.append(line)
with open(path, 'w') as f: f.writelines(new_lines)
print('patched sidebar/index.tsx; kept HEAD block; lines=', len(new_lines))
