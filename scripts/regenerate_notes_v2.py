import json, os, re
from collections import Counter

BASE = '/home/yeqiuqiu/clawd-architect'
PDF_ANALYSIS = os.path.join(BASE, 'memory', 'pdf_analysis')
TEXT_DIR = os.path.join(PDF_ANALYSIS, 'text')
OUT_DIR = os.path.join(PDF_ANALYSIS, 'notes_v2')
INDEX_JSON = os.path.join(PDF_ANALYSIS, 'index.json')
BATCH1 = os.path.join(PDF_ANALYSIS, 'regen_batch1.json')
BATCH2 = os.path.join(PDF_ANALYSIS, 'regen_batch2.json')

with open(INDEX_JSON, 'r', encoding='utf-8') as f:
    index = json.load(f)
sha_to_paths = {}
for item in index:
    sha = item.get('sha256')
    path = item.get('path')
    if not sha:
        continue
    sha_to_paths.setdefault(sha, []).append(path)


def load_batches():
    items = []
    for path in [BATCH1, BATCH2]:
        with open(path, 'r', encoding='utf-8') as f:
            items.extend(json.load(f))
    return items

STOPWORDS = set("""
a an the and or if of for to in on by with without as at from is are was were be been being this that these those it its into over under up down not no yes do does did done can could would should must may might will shall
we our you your they their he she his her them us i me my mine ours theirs ours
""".split())

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

KEYWORDS_RISK = re.compile(r'\b(risk|fail|failure|error|limit|latency|bias|leak|leakage|spoof|attack|edge case|manipulat|mitigat|downtime|rate limit|inaccurate)\b', re.I)
KEYWORDS_ACTION = re.compile(r'\b(should|must|recommend|need to|build|implement|ensure|validate|design|deploy|monitor|test)\b', re.I)
KEYWORDS_DATA = re.compile(r'\b(schema|fields|field|json|table|columns|endpoint|api|payload|message|format|interface|websocket|rest)\b', re.I)


def normalize_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^\d+$', line) or line in {'•', '-', '*'}:
            i += 1
            continue
        if not line:
            cleaned.append('')
            i += 1
            continue
        if line.endswith('-') and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            line = line[:-1] + nxt
            i += 2
            cleaned.append(line)
            continue
        cleaned.append(line)
        i += 1

    paragraphs = []
    buf = []
    for line in cleaned:
        if not line:
            if buf:
                paragraphs.append(' '.join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(' '.join(buf))
    return '\n\n'.join(paragraphs)


def clean_text(text: str) -> str:
    text = re.sub(r'(?:\s\d{1,3}){3,}', ' ', text)  # remove long numeric sequences
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def summarize_sentences(text: str, n=4, max_len=220):
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    words = [w.lower() for w in re.findall(r'[A-Za-z0-9_]+', text)]
    freq = Counter(w for w in words if w not in STOPWORDS)
    scored = []
    for i, s in enumerate(sentences):
        if len(s) < 30:
            continue
        if not re.match(r'^[A-Z0-9]', s):
            continue
        words_s = [w.lower() for w in re.findall(r'[A-Za-z0-9_]+', s)]
        if not words_s:
            continue
        score = sum(freq.get(w, 0) for w in words_s if w not in STOPWORDS) / max(1, len(words_s))
        scored.append((score, i, s))
    scored.sort(reverse=True)
    top = []
    for score, i, s in scored:
        s = clean_text(s)
        if len(s) > max_len:
            s = s[:max_len].rsplit(' ', 1)[0] + '…'
        if s not in [t[1] for t in top]:
            top.append((i, s))
        if len(top) >= n:
            break
    return [s for _, s in sorted(top, key=lambda x: x[0])]


def extract_endpoints(text: str, n=8):
    candidates = set()
    for m in re.finditer(r'(?<!\w)(/[-A-Za-z0-9._]+(?:/[-A-Za-z0-9._]+)*)', text):
        ep = m.group(1)
        if len(ep) < 4:
            continue
        if ep.startswith('//'):
            continue
        if 'http' in ep.lower():
            continue
        if '.' in ep:
            continue
        if 'file_' in ep.lower():
            continue
        if re.search(r'\d{6,}', ep):
            continue
        if ep.count('/') == 1 and len(ep) < 5:
            continue
        candidates.add(ep.rstrip('.,;:'))
    endpoints = sorted(candidates, key=lambda x: (len(x), x))
    return endpoints[:n]


def extract_sentences_by_keyword(text: str, regex, n=4):
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    selected = []
    for s in sentences:
        if regex.search(s):
            s = clean_text(s)
            if len(s) < 30:
                continue
            if not re.match(r'^[A-Z0-9]', s):
                continue
            if s not in selected:
                selected.append(s)
        if len(selected) >= n:
            break
    return selected


def infer_title(title_hint: str):
    title = title_hint or 'Untitled'
    title = title.replace('.pdf', '').replace('.PDF', '')
    return title.strip()


def build_note(sha, title_hint, raw_text, source_path, text_path):
    norm = normalize_text(raw_text)
    title = infer_title(title_hint)

    paragraphs = [p for p in norm.split('\n\n') if p.strip()]
    first_para = paragraphs[0] if paragraphs else ''
    what = []
    if first_para:
        first_sents = [s.strip() for s in SENT_SPLIT.split(first_para) if s.strip()]
        what = [clean_text(s) for s in first_sents if re.match(r'^[A-Z0-9]', s)][:3]

    key_block = ''
    for p in paragraphs[1:]:
        if re.search(r'\b(key|summary|takeaway|finding|recommendation|design|approach|overview)\b', p, re.I):
            key_block += ' ' + p
            if len(key_block) > 1200:
                break
    key_takeaways = summarize_sentences(key_block, n=4) if key_block else summarize_sentences(norm, n=4)

    data_sent = extract_sentences_by_keyword(norm, KEYWORDS_DATA, n=4)
    endpoints = extract_endpoints(norm, n=8)
    risks = extract_sentences_by_keyword(norm, KEYWORDS_RISK, n=4)
    actions = extract_sentences_by_keyword(norm, KEYWORDS_ACTION, n=4)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**sha256**: `{sha}`")
    lines.append("")
    src_parts = []
    if source_path:
        src_parts.append(source_path)
    if text_path:
        src_parts.append(text_path)
    src = " | ".join(src_parts) if src_parts else "unknown"
    lines.append(f"**source**: {src}")
    lines.append("")

    lines.append("## What it is")
    if what:
        lines.extend([f"- {s}" for s in what])
    else:
        lines.append("- Summary unavailable from extracted text.")
    lines.append("")

    lines.append("## Key takeaways")
    if key_takeaways:
        lines.extend([f"- {s}" for s in key_takeaways])
    else:
        lines.append("- Key takeaways not explicitly stated in the source; infer from context.")
    lines.append("")

    if data_sent or endpoints:
        lines.append("## Data/schema or interfaces")
        if endpoints:
            lines.append("**Endpoints / interfaces mentioned**:")
            lines.append("- " + ", ".join(endpoints))
        if data_sent:
            lines.append("**Data / schema notes**:")
            lines.extend([f"- {s}" for s in data_sent])
        lines.append("")

    lines.append("## Reliability/risks")
    if risks:
        lines.extend([f"- {s}" for s in risks])
    else:
        lines.append("- Risks are not explicitly detailed; consider rate limits, data gaps, and inference error.")
    lines.append("")

    lines.append("## Actionable implications")
    if actions:
        lines.extend([f"- {s}" for s in actions])
    else:
        lines.append("- Implement the core recommendations and validate against described constraints and metrics.")
    lines.append("")

    lines.append("## Open questions")
    lines.append("- Which assumptions should be validated with fresh data or stakeholder review?")
    lines.append("")

    return "\n".join(lines)


def main():
    items = load_batches()
    os.makedirs(OUT_DIR, exist_ok=True)
    for item in items:
        sha = item['sha']
        title = item.get('title', sha)
        text_path = os.path.join(TEXT_DIR, f"{sha}.txt")
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            text = ""
        source_path = None
        if sha in sha_to_paths:
            source_path = sha_to_paths[sha][0]
        note = build_note(sha, title, text, source_path, text_path)
        out_path = os.path.join(OUT_DIR, f"{sha}.md")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(note)

if __name__ == '__main__':
    main()
