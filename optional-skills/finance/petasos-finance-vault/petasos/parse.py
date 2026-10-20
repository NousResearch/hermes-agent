"""petasos.parse — the generic HTML-table email parsing engine (R2.3, C3).

This module contains ZERO bank knowledge: it walks the registry data only.
Adding or fixing a bank is a banks.yaml edit, never a code edit (P2).

Engine semantics (v1 parity, bank_adapters.py:91–129 at 119eaab):
- first format (across banks, registry order) whose subject matches AND whose
  `requires` all capture wins; within a format the first regex to capture a
  named group wins;
- direction resolved BEFORE amount conversion (`sign_of` needs the raw
  signed string); `defaults` fill missing optional fields; `requires` are
  checked AFTER defaults; the marker group is popped from output.

Email convention (C3): first line `Subject:`; HTML stripped iff `<`…`>` both
present; entities unescaped; whitespace collapsed. Regex hygiene per INV-4
lives in banks.yaml (optional pipes, lookahead terminators, `\\d{2,4}` years);
`to_int` carries the INV-5 ×100 guard (strip the bank's declared decimal tail
FIRST, then thousands separators).
"""
import html as _html
import re


def read_email(raw):
    """(subject, tag-stripped body) per the C3 email convention."""
    lines = (raw or '').splitlines()
    subj = ''
    if lines and lines[0].lower().startswith('subject:'):
        subj = lines[0][len('subject:'):].strip()
        raw = '\n'.join(lines[1:])
    return subj, _strip_html(raw)


def _strip_html(raw):
    if '<' in raw and '>' in raw:
        raw = re.sub(r'<[^>]+>', ' ', raw)
        raw = _html.unescape(raw)
    return re.sub(r'\s+', ' ', raw).strip()


def to_int(text, decimal_separator='.'):
    """'1,234,567.00' → 1234567 (INV-5 ×100 guard): strip the bank's declared
    decimal tail FIRST, then delete thousands separators."""
    s = (text or '').strip()
    if decimal_separator == ',':
        s = re.sub(r',\d{1,2}$', '', s)
    else:
        s = re.sub(r'\.\d{1,2}$', '', s)
    m = re.search(r'-?\d+', s.replace(',', '').replace('.', ''))
    return int(m.group()) if m else None


def normalize_date(text):
    """dd/mm/{yy|yyyy} (optionally with a time tail) → ISO yyyy-mm-dd;
    2-digit years are 20xx (v1 parity). ISO input passes through unchanged."""
    s = (text or '').strip().split(' ')[0]
    m = re.fullmatch(r'(\d{2})/(\d{2})/(\d{2,4})', s)
    if m:
        dd, mm, yy = m.groups()
        if len(yy) == 2:
            yy = '20' + yy
        return f'{yy}-{mm}-{dd}'
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
        return s
    return (text or '').strip()


def parse_format(fmt, bank, subject, body):
    """One format spec against one email. None = no match (subject miss or a
    required field failed to capture)."""
    if not any(k in (subject or '') for k in fmt.get('subject_any', [])):
        return None
    tx = {}
    for rx in fmt.get('fields', []):
        m = re.search(rx, body)
        if m:
            for k, v in m.groupdict().items():
                if v is not None and k not in tx:
                    tx[k] = v.strip()
    dec = (bank or {}).get('amount', {}).get('decimal_separator', '.')
    # direction FIRST — sign_of needs the raw captured string ('+1,234…')
    # before to_int() strips the sign
    d = fmt.get('direction', {})
    marker_group = None
    if 'marker' in d:
        marker_group = d['marker'].get('group')
        g = tx.get(marker_group, '') or ''
        if any(s in g for s in d['marker'].get('out_contains', [])):
            tx['dir'] = 'OUT'
        elif any(s in g for s in d['marker'].get('in_contains', [])):
            tx['dir'] = 'IN'
    elif 'sign_of' in d:
        tx['dir'] = 'IN' if str(tx.get(d['sign_of'], '')).startswith('+') else 'OUT'
    for f in fmt.get('amounts', []):
        if f in tx:
            tx[f] = to_int(tx[f], dec)
    for k, v in fmt.get('defaults', {}).items():
        tx.setdefault(k, v)
    for r in fmt.get('requires', []):
        if tx.get(r) in (None, ''):
            return None
    if marker_group:
        tx.pop(marker_group, None)
    out = {'kind': fmt['kind']}
    out.update(tx)
    return out


def parse_any(registry, subject, body):
    """First format (across all banks, registry order) whose subject matches
    and whose required fields all capture. {} when nothing matches — the
    caller decides visibility (G2)."""
    from .registry import iter_formats
    for _, bank, fmt in iter_formats(registry.banks):
        tx = parse_format(fmt, bank, subject, body)
        if tx:
            return tx
    return {}
