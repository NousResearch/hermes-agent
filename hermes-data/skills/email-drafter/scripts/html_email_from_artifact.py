#!/usr/bin/env python3
"""Turn a DRA Content artifact's index.html into an email-safe HTML body + text fallback.

Usage:
    python3 html_email_from_artifact.py SRC_INDEX.html OUT_PREFIX \
        [--greeting GREETING.html] [--width 760] [--img-dir DIR]

Writes OUT_PREFIX.html, OUT_PREFIX.txt and OUT_PREFIX.images.json
(the manifest lists {path, cid, name} for the inline MIME parts to attach).

What it fixes for Gmail:
  * every CSS rule inlined; CSS custom properties resolved to literals
  * box-shadow / page-break-* dropped, duplicate declarations collapsed
  * display:flex (.sig signature row) -> role="presentation" 3-column table
  * tr:nth-child(even) zebra -> explicit background on those <td>s
  * relative <img src> -> src="cid:..." plus an images.json entry
  * outer .sheet div replaced by a fixed-width wrapper table on a tinted canvas

It prints counters that must all be 0 except cid images:
    unresolved var() left / style blocks left / flex left
"""
import re, sys, os, json, html as htmllib

HEAD_FONT = "Helvetica,Arial,sans-serif"
BODY_FONT = "Georgia,'Times New Roman',serif"


def parse_rules(style: str):
    vars_ = {}
    for m in re.finditer(r'--([\w-]+)\s*:\s*([^;}]+)', style):
        vars_[m.group(1)] = m.group(2).strip()

    def resv(text):
        return re.sub(r'var\(--([\w-]+)\)', lambda m: vars_.get(m.group(1), m.group(0)), text)

    rules = {}
    for sel, body in re.findall(r'([^{}]+)\{([^{}]*)\}', style):
        sel = sel.strip()
        if sel.startswith('@') or sel.startswith(':root'):
            continue
        rules.setdefault(sel, []).append(resv(' '.join(body.split())))
    return rules


def make_getter(rules):
    def get(*sels):
        out = []
        for s in sels:
            if s in rules:
                out.extend(rules[s])
        return ';'.join(x.strip().strip(';') for x in out if x.strip())
    return get


def fix_style(style_str: str) -> str:
    out, seen = [], set()
    for p in (x.strip() for x in style_str.split(';')):
        if not p:
            continue
        if p.startswith('box-shadow') or p.startswith('page-break'):
            continue
        key = p.split(':')[0].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return ';'.join(out)


TAG_RE = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)((?:[^>"\']|"[^"]*"|\'[^\']*\')*)>')


def convert(raw: str, img_dir: str | None, width: int, greeting: str):
    style = re.search(r'<style[^>]*>(.*?)</style>', raw, re.S).group(1)
    rules = parse_rules(style)
    get = make_getter(rules)

    tag_styles = {
        'header': get('header'), 'h1': get('h1'), 'h2': get('h2'), 'h3': get('h3'),
        'p': get('p'), 'ul': get('ul,ol') or get('ul'), 'ol': get('ul,ol') or get('ol'),
        'li': get('li'),
        'table': get('table') + ';table-layout:auto',
        'th': get('th'), 'td': get('td'),
        'figure': get('figure'), 'figcaption': get('figcaption'),
        'code': get('code'), 'footer': get('footer'), 'strong': '',
        'a': 'color:#8a1c1c;text-decoration:underline',
        'img': get('figure img') + ';max-width:100%;height:auto;width:100%;display:block',
        'hr': 'border:0;border-top:1px solid #d8d2c6;margin:18px 0',
    }
    cls_styles = {k.lstrip('.'): get(k) for k in
                  ('.sheet', '.firm', '.sub', '.flag', '.note', '.action', '.num')}

    body = raw.split('<body', 1)[1].split('>', 1)[1]
    body = body.replace('</body>', '').replace('</html>', '')

    def repl(m):
        tag, attrs = m.group(1).lower(), m.group(2)
        cm = re.search(r'class\s*=\s*"([^"]*)"', attrs)
        classes = cm.group(1).split() if cm else []
        bits = [tag_styles[tag]] if tag_styles.get(tag) else []
        for c in classes:
            if c in cls_styles:
                bits.append(cls_styles[c])
        sm = re.search(r'style\s*=\s*"([^"]*)"', attrs)
        if sm:
            attrs = attrs[:sm.start()] + attrs[sm.end():]
            bits.append(sm.group(1))
        st = fix_style(';'.join(b for b in bits if b))
        return m.group(0) if not st else '<%s%s style="%s">' % (tag, attrs.rstrip(), st)

    body = TAG_RE.sub(repl, body)

    # images -> cid  (manifest for the MIME builder)
    images = []
    for m in re.finditer(r'<img[^>]*src="([^"]+)"', body):
        src = m.group(1)
        if src.startswith('cid:'):
            continue
        name = os.path.basename(src)
        cid = os.path.splitext(name)[0] + '@note'
        images.append({'path': os.path.join(img_dir or '.', name), 'cid': cid, 'name': name})
        body = body.replace('src="%s"' % src, 'src="cid:%s"' % cid)

    # zebra striping
    def stripe(m):
        rows, out, n = re.split(r'(?=<tr)', m.group(0)), [], 0
        for r in rows:
            if r.lstrip().startswith('<tr'):
                n += 1
                if n % 2 == 0:
                    r = re.sub(r'<td((?:(?!>).)*?)style="',
                               lambda mm: '<td%sstyle="background:#faf8f3;' % mm.group(1), r)
                    r = re.sub(r'<td(?![^>]*style=)', '<td style="background:#faf8f3"', r)
            out.append(r)
        return ''.join(out)
    body = re.sub(r'<table\b.*?</table>', stripe, body, flags=re.S)

    # .sig flex -> table
    def sig(m):
        names = re.findall(r'<div>([^<]*)</div>', m.group(1))
        if not names:
            return m.group(0)
        cells = ''.join(
            '<td style="width:33%;vertical-align:bottom;padding:26px 14px 0 0;">'
            '<div style="border-top:1px solid #999;padding-top:6px;font-family:' + HEAD_FONT + ';'
            'font-size:12px;color:#1a1a1a;">' + htmllib.escape(n.strip()) + '</div></td>'
            for n in names)
        return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
                'style="width:100%;border-collapse:collapse;margin-top:10px;"><tr>' + cells
                + '</tr></table>')
    body = re.sub(r'<div class="sig">(.*?)</div>\s*</footer>',
                  lambda m: sig(m) + '</footer>', body, flags=re.S)

    # drop the outer .sheet wrapper
    body = re.sub(r'^.*?<div class="sheet"[^>]*>', '', body, count=1, flags=re.S)
    body = re.sub(r'</div>\s*$', '', body.rstrip())

    shell = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#e9e5dc;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
 style="width:100%;background:#e9e5dc;border-collapse:collapse;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="%(w)d" cellpadding="0" cellspacing="0" border="0"
 style="width:100%%;max-width:%(w)dpx;background:#fdfcf9;border-collapse:collapse;">
<tr><td style="padding:34px 34px 44px;font-family:%(f)s;font-size:15.5px;line-height:1.6;color:#1a1a1a;">
%(greet)s
%(rule)s
%(note)s
</td></tr></table>
</td></tr></table>
</body></html>
""" % {'w': width, 'f': BODY_FONT, 'greet': greeting,
       'rule': ('<div style="border-top:3px double #d8d2c6;margin:22px 0 26px;font-size:0;line-height:0;">&nbsp;</div>'
                if greeting.strip() else ''),
       'note': body}
    return shell, images


def to_text(shell: str) -> str:
    t = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', shell, flags=re.S)
    t = re.sub(r'</t[dh]>', '  |  ', t)
    t = re.sub(r'</tr>', '\n', t)
    t = re.sub(r'<br\s*/?>', '\n', t)
    t = re.sub(r'</(p|li|h1|h2|h3|div|figcaption|header|footer|table)>', '\n\n', t)
    t = htmllib.unescape(re.sub(r'<[^>]+>', '', t))
    t = re.sub(r'\n[ ]*\|[ ]*\n', '\n', t)
    t = '\n'.join(l.strip() for l in t.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', t).strip() + '\n'


def main(argv):
    src, out = argv[1], argv[2]
    width, img_dir, greeting_path = 760, None, None
    i = 3
    while i < len(argv):
        if argv[i] == '--width':
            width = int(argv[i + 1]); i += 2
        elif argv[i] == '--img-dir':
            img_dir = argv[i + 1]; i += 2
        elif argv[i] == '--greeting':
            greeting_path = argv[i + 1]; i += 2
        else:
            raise SystemExit('unknown arg %r' % argv[i])

    raw = open(src).read()
    greeting = open(greeting_path).read() if greeting_path else ''
    shell, images = convert(raw, img_dir or os.path.dirname(os.path.abspath(src)), width, greeting)

    open(out + '.html', 'w').write(shell)
    open(out + '.txt', 'w').write(to_text(shell))
    json.dump(images, open(out + '.images.json', 'w'), indent=1)

    print('html bytes', len(shell), '| text bytes', len(open(out + ".txt").read()))
    print('unresolved var() left:', shell.count('var(--'))
    print('style blocks left:', shell.count('<style'))
    print('flex left:', shell.count('display:flex'))
    print('cid images:', shell.count('src="cid:'), '| manifest:', len(images))


if __name__ == '__main__':
    main(sys.argv)
