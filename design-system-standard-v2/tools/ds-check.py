#!/usr/bin/env python3
"""
ds-check — ตัวตรวจว่าโปรเจกต์ใช้ Design Token จริงไหม (ไม่พึ่ง lib นอก)

ไล่หา "ค่าที่ฝังตายตัว" (hardcode) ที่ควรมาจาก token แทน:
  - สี hex (#abc, #aabbcc)  → เฉพาะตำแหน่งที่เป็น "ค่า" (มี : = หรือ quote นำหน้า) ไม่จับ id selector
  - rgb()/rgba()/hsl()      → ข้าม rgb(var(--x)) ที่ใช้ token แล้ว
  - oklch(...) literal        → ข้าม oklch(var(--x))
  - ขนาด px >=10 ใน inline style (บรรทัดมี style/sx เท่านั้น) → ข้าม var()/calc()

รุ่น v1.1 · แก้ตาม Codex review (2026-07-05): จับหลาย match/บรรทัด · ตรวจบริบทราย match
  · ข้าม token-wrapped color · px เฉพาะ inline style · รองรับ .ds-allowlist

โหมด: warn (รายงานเฉย · exit 0) | error (พบแล้ว exit 1 · ใช้ใน CI)
ยกเว้น: ไฟล์ token เอง, dist/, node_modules/, *.stories.*, บรรทัด // ds-allow,
        และ substring/แนวไฟล์ใน <project>/.ds-allowlist (สีลูกค้า/แบรนด์/คนนอก)

ใช้:
  python3 ds-check.py <project_dir> [--mode warn|error] [--json]
"""
import sys, os, re, json, argparse

SCAN_EXT = ('.tsx', '.jsx', '.ts', '.css', '.scss')
SKIP_DIR = {'node_modules', 'dist', '.next', '.git', 'build', 'coverage', '.turbo'}
SKIP_FILE = re.compile(r'(tokens?\.(ts|json|css|js)|\.stories\.|design-system)', re.I)
ALLOW_MARK = 'ds-allow'

RULES = [
    ('hex-color',     re.compile(r'#[0-9a-fA-F]{3,8}\b')),
    ('rgb-hsl',       re.compile(r'\b(rgba?|hsla?)\s*\(')),
    ('oklch-literal', re.compile(r'\boklch\s*\(')),
    ('px-in-jsx',     re.compile(r'\b\d{2,}px\b')),
]


def load_allowlist(root):
    path = os.path.join(root, '.ds-allowlist')
    items = []
    if os.path.isfile(path):
        with open(path, encoding='utf-8', errors='ignore') as f:
            for ln in f:
                ln = ln.strip()
                if ln and not ln.startswith('#'):
                    items.append(ln)
    return items


def iter_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIR]
        for fn in fns:
            if fn.endswith(SCAN_EXT) and not SKIP_FILE.search(fn):
                yield os.path.join(dp, fn)


def _value_context(line, start):
    """hex อยู่ในตำแหน่ง 'ค่า' ไหม (มี : = หรือ quote นำหน้าในบรรทัด) — กัน id selector #abc{"""
    pre = line[:start]
    return (':' in pre) or ('=' in pre) or ('"' in pre) or ("'" in pre)


def scan(root, allowlist):
    findings = []
    files_scanned = 0
    for path in iter_files(root):
        rel = os.path.relpath(path, root)
        if any(a in rel for a in allowlist):
            continue
        files_scanned += 1
        try:
            with open(path, encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except OSError:
            continue
        is_css = path.endswith(('.css', '.scss'))
        has_style = None  # lazy per line
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if ALLOW_MARK in line or stripped.startswith(('//', '*', '/*')):
                continue
            if any(a in line for a in allowlist):
                continue
            style_ctx = ('style' in line) or ('sx' in line) or is_css
            for name, rx in RULES:
                for m in rx.finditer(line):
                    text = m.group(0)
                    s, e = m.start(), m.end()
                    after = line[e:e + 6].lstrip()
                    if name in ('rgb-hsl', 'oklch-literal'):
                        if after.startswith('var('):      # rgb(var(--x)) = ใช้ token แล้ว
                            continue
                    elif name == 'hex-color':
                        if not _value_context(line, s):    # ข้าม id selector
                            continue
                    elif name == 'px-in-jsx':
                        if is_css or not style_ctx:        # px เฉพาะ inline style ใน jsx
                            continue
                        pre = line[max(0, s - 6):s]
                        if 'var(' in pre or 'calc(' in pre:
                            continue
                    findings.append({'file': rel, 'line': i,
                                     'rule': name, 'text': text})
    return files_scanned, findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('project_dir')
    ap.add_argument('--mode', choices=['warn', 'error'], default='warn')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()

    allowlist = load_allowlist(a.project_dir)
    files_scanned, findings = scan(a.project_dir, allowlist)
    by_rule = {}
    for f in findings:
        by_rule[f['rule']] = by_rule.get(f['rule'], 0) + 1
    bad_files = {f['file'] for f in findings}
    adoption = 0 if files_scanned == 0 else round(
        (files_scanned - len(bad_files)) / files_scanned * 100, 1)

    if a.json:
        print(json.dumps({'files_scanned': files_scanned,
                          'violations': len(findings), 'by_rule': by_rule,
                          'adoption_pct': adoption, 'allowlist': allowlist,
                          'findings': findings}, ensure_ascii=False, indent=2))
    else:
        print(f"สแกน {files_scanned} ไฟล์ · พบ hardcode {len(findings)} จุด "
              f"ใน {len(bad_files)} ไฟล์ · คะแนนใช้ token {adoption}%"
              + (f" · allowlist {len(allowlist)} รายการ" if allowlist else ""))
        for rule, n in sorted(by_rule.items(), key=lambda x: -x[1]):
            print(f"  - {rule}: {n}")
        for f in findings[:25]:
            print(f"    {f['file']}:{f['line']}  [{f['rule']}]  {f['text']}")
        if len(findings) > 25:
            print(f"    ... อีก {len(findings) - 25} จุด (ใช้ --json ดูครบ)")

    if a.mode == 'error' and findings:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
