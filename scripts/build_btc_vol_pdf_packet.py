from __future__ import annotations

import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from html import unescape

from bs4 import BeautifulSoup
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    ListFlowable,
    ListItem,
    Preformatted,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'institutional'
OUT = ART / 'pdf-packet'
INVESTOR = OUT / '01-investor-and-diligence-pdfs'
INTERNAL = OUT / '02-internal-team-pdfs'
COMBINED = OUT / 'combined'

CONTROL = 'SCREEN-ONLY · NOT EXECUTABLE · INTERNAL/INVESTOR DILIGENCE ONLY · NOT A QUOTE · NOT AN OFFER'

DOCS = [
    ('investor', '00-packet-index', ART/'investor-packet/packet_index.md', 'Packet Index'),
    ('investor', '01-one-page-tear-sheet', ART/'investor-packet/one-page-tear-sheet.md', 'One Page Tear Sheet'),
    ('investor', '02-investor-memo', ART/'investor-packet/btc-vol-desk-investor-memo.md', 'Investor Memo'),
    ('investor', '03-static-site-export', ART/'investor-packet/site/index.html', 'Static Site Text Export'),
    ('investor', '04-latest-monitor-report', ART/'investor-packet/evidence/latest-report.md', 'Latest Monitor Report'),
    ('investor', '05-backtest-research-evidence', ART/'investor-packet/evidence/backtest-report.md', 'Backtest Research Evidence'),
    ('investor', '06-legal-wrapper-draft', ART/'investor-packet/evidence/legal-wrapper-package-v1.md', 'Legal Wrapper Draft'),

    ('internal', '00-master-plan-v2', ART/'idea1-master-plan-v2-evidence-first-hardening.md', 'Master Plan v2 Evidence-First Hardening'),
    ('internal', '01-licensed-source-acquisition-package', ART/'historical/licensed_source_package/licensed-source-acquisition-package-v1.md', 'Licensed Source Acquisition Package'),
    ('internal', '02-source-intake-validation', ART/'historical/licensed_source_package/source-intake-skeleton-validation-v1.md', 'Source Intake Skeleton Validation'),
    ('internal', '03-source-blocker-audit', ART/'historical/source_audit/source_blocker_audit_current.md', 'Source Blocker Audit'),
    ('internal', '04-rfq-verification-plan', ART/'idea1-rfq-verification-plan-v0.md', 'RFQ Verification Plan'),
    ('internal', '05-rfq-verification-board', ART/'idea1-rfq-verification-board-v0.md', 'RFQ Verification Board'),
    ('internal', '06-launch-model-decision-matrix', ART/'idea1-launch-model-decision-matrix-v0.md', 'Launch Model Decision Matrix'),
    ('internal', '07-investor-deep-dive', ART/'idea1-investor-deep-dive-v0.md', 'Investor Deep Dive'),
    ('internal', '08-data-evidence-appendix', ART/'idea1-data-evidence-appendix.md', 'Data Evidence Appendix'),
    ('internal', '09-client-map-commercial-wedge', ART/'idea1-client-map-commercial-wedge.md', 'Client Map / Commercial Wedge'),
    ('internal', '10-treasury-case-study', ART/'idea1-treasury-case-study-10000btc-v0.md', 'Treasury Case Study'),
    ('internal', '11-miner-production-hedge-case-study', ART/'idea1-miner-production-hedge-case-study-v0.md', 'Miner Production Hedge Case Study'),
    ('internal', '12-board-policy-template', ART/'idea1-btc-treasury-board-policy-template-v0.md', 'BTC Treasury Board Policy Template'),
    ('internal', '13-miner-hedge-calculator-spec', ART/'idea1-miner-hedge-calculator-spec-v0.md', 'Miner Hedge Calculator Spec'),
    ('internal', '14-data-pipeline-hardening-spec', ART/'idea1-btc-vol-monitor-data-pipeline-hardening-spec-v0.md', 'Data Pipeline Hardening Spec'),
    ('internal', '15-legal-perimeter-memo', ART/'idea1-legal-perimeter-memo-v0.md', 'Legal Perimeter Memo'),
    ('internal', '16-investor-deck-outline', ART/'idea1-investor-deck-outline-v0.md', 'Investor Deck Outline'),
    ('internal', '17-30-day-execution-sprint', ART/'idea1-30day-execution-sprint-v0.md', '30-Day Execution Sprint'),
]


def clean_md(text: str) -> str:
    text = re.sub(r'```[a-zA-Z0-9_-]*\n', '```\n', text)
    text = text.replace('\t', '    ')
    return text


def html_to_md_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding='utf-8', errors='replace'), 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.extract()
    lines = []
    for el in soup.find_all(['h1','h2','h3','h4','p','li','div','span','code','strong']):
        txt = ' '.join(el.get_text(' ', strip=True).split())
        if not txt:
            continue
        if el.name == 'h1': lines.append('# '+txt)
        elif el.name == 'h2': lines.append('\n## '+txt)
        elif el.name == 'h3': lines.append('\n### '+txt)
        elif el.name == 'li': lines.append('- '+txt)
        else: lines.append(txt)
    # de-dupe consecutive identical lines
    out=[]
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return '\n'.join(out)


def read_doc(path: Path) -> str:
    if path.suffix.lower() in {'.html', '.htm'}:
        return html_to_md_text(path)
    return clean_md(path.read_text(encoding='utf-8', errors='replace'))


def make_styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(name='DocTitle', parent=base['Title'], fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=colors.HexColor('#0b1b2b'), spaceAfter=14))
    base.add(ParagraphStyle(name='Control', parent=base['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#8a5a00'), backColor=colors.HexColor('#fff4d6'), borderColor=colors.HexColor('#d8a531'), borderWidth=0.5, borderPadding=5, spaceAfter=12))
    base.add(ParagraphStyle(name='H1x', parent=base['Heading1'], fontSize=17, leading=21, textColor=colors.HexColor('#102a43'), spaceBefore=14, spaceAfter=8))
    base.add(ParagraphStyle(name='H2x', parent=base['Heading2'], fontSize=13, leading=17, textColor=colors.HexColor('#1f4e79'), spaceBefore=12, spaceAfter=6))
    base.add(ParagraphStyle(name='Bodyx', parent=base['BodyText'], fontSize=9.2, leading=12.5, alignment=TA_LEFT, spaceAfter=5))
    base.add(ParagraphStyle(name='Bulletx', parent=base['BodyText'], fontSize=8.9, leading=12, leftIndent=14, firstLineIndent=-8, spaceAfter=3))
    base.add(ParagraphStyle(name='Codex', parent=base['Code'], fontName='Courier', fontSize=7.2, leading=9, textColor=colors.HexColor('#1f2933'), backColor=colors.HexColor('#f2f4f7'), borderPadding=4, spaceAfter=6))
    return base


def esc(s: str) -> str:
    return (s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))


def parse_markdown_to_flowables(text: str, styles) -> list:
    flow = []
    code = []
    in_code = False
    table_buf = []

    def flush_code():
        nonlocal code
        if code:
            blob = '\n'.join(code)
            if len(blob) > 6500:
                blob = blob[:6500] + '\n...[truncated in PDF rendering; see source artifact]'
            flow.append(Preformatted(blob, styles['Codex'], maxLineLength=95))
            code = []

    def flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        rows=[]
        for line in table_buf[:28]:
            cells=[c.strip() for c in line.strip('|').split('|')]
            if all(re.fullmatch(r':?-{2,}:?', c or '') for c in cells):
                continue
            rows.append([Paragraph(esc(c[:180]), styles['Bodyx']) for c in cells[:5]])
        if rows:
            tbl=Table(rows, repeatRows=1)
            tbl.setStyle(TableStyle([
                ('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#ccd6e0')),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f2fb')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
            ]))
            flow.append(tbl); flow.append(Spacer(1, 6))
        table_buf=[]

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith('```'):
            flush_table()
            if in_code:
                in_code=False; flush_code()
            else:
                in_code=True; code=[]
            continue
        if in_code:
            code.append(line)
            continue
        if '|' in line and line.strip().startswith('|'):
            table_buf.append(line)
            continue
        flush_table()
        stripped=line.strip()
        if not stripped:
            flow.append(Spacer(1, 4)); continue
        if stripped.startswith('# '):
            flow.append(Paragraph(esc(stripped[2:].strip()), styles['H1x']))
        elif stripped.startswith('## '):
            flow.append(Paragraph(esc(stripped[3:].strip()), styles['H2x']))
        elif stripped.startswith('### '):
            flow.append(Paragraph('<b>'+esc(stripped[4:].strip())+'</b>', styles['Bodyx']))
        elif re.match(r'^[-*]\s+', stripped):
            flow.append(Paragraph('• '+esc(re.sub(r'^[-*]\s+', '', stripped)), styles['Bulletx']))
        elif re.match(r'^\d+\.\s+', stripped):
            flow.append(Paragraph(esc(stripped), styles['Bulletx']))
        else:
            # light inline cleanup
            s = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', esc(stripped))
            s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
            flow.append(Paragraph(s, styles['Bodyx']))
    flush_table(); flush_code()
    return flow


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#6b7280'))
    canvas.drawString(0.55*inch, 0.35*inch, CONTROL)
    canvas.drawRightString(7.95*inch, 0.35*inch, f'Page {doc.page}')
    canvas.restoreState()


def build_pdf(title: str, source: Path, out: Path):
    styles = make_styles()
    out.parent.mkdir(parents=True, exist_ok=True)
    story = [Paragraph(title, styles['DocTitle']), Paragraph(CONTROL, styles['Control'])]
    story.append(Paragraph(f'Source artifact: <font name="Courier">{esc(str(source.relative_to(ROOT)))}</font>', styles['Bodyx']))
    story.append(Spacer(1, 10))
    story.extend(parse_markdown_to_flowables(read_doc(source), styles))
    doc = SimpleDocTemplate(str(out), pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch, title=title)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def build_cover(readme_path: Path, pdf_path: Path, rows: list[dict]):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    readme = f"""# BTC Vol Desk PDF Handoff Packet

Generated: {now}

{CONTROL}

## Investor / Diligence PDFs

These are the cleaner documents to show an investor for diligence context. They remain internal-diligence-only unless counsel approves external distribution.

"""
    for r in rows:
        if r['audience']=='investor': readme += f"- `{r['pdf']}` — {r['title']}\n"
    readme += "\n## Internal Team PDFs\n\nThese are for internal execution, legal, source acquisition, RFQ workflow, and implementation planning.\n\n"
    for r in rows:
        if r['audience']=='internal': readme += f"- `{r['pdf']}` — {r['title']}\n"
    readme += "\n## Non-negotiable gates\n\n- Screen-only, not executable.\n- Not a quote, not a fund offering, not client-facing advice.\n- External use remains blocked until counsel-approved wrapper.\n- Quote-verified status requires real two-counterparty evidence and attestations.\n- Licensed historical source coverage remains incomplete; fixtures do not count as covered.\n"
    readme_path.write_text(readme, encoding='utf-8')
    build_pdf('BTC Vol Desk PDF Handoff Packet - README', readme_path, pdf_path)


def merge_pdfs(paths: list[Path], out: Path):
    writer=PdfWriter()
    for p in paths:
        reader=PdfReader(str(p))
        for page in reader.pages:
            writer.add_page(page)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f: writer.write(f)


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    INVESTOR.mkdir(parents=True); INTERNAL.mkdir(parents=True); COMBINED.mkdir(parents=True)
    manifest=[]; investor_pdfs=[]; internal_pdfs=[]
    for audience, slug, src, title in DOCS:
        if not src.exists():
            manifest.append({'audience':audience,'slug':slug,'title':title,'source':str(src),'missing':True})
            continue
        dest_dir=INVESTOR if audience=='investor' else INTERNAL
        pdf=dest_dir/f'{slug}.pdf'
        build_pdf(title, src, pdf)
        rel=str(pdf.relative_to(OUT))
        row={'audience':audience,'slug':slug,'title':title,'source':str(src.relative_to(ROOT)),'pdf':rel,'bytes':pdf.stat().st_size,'pages':len(PdfReader(str(pdf)).pages)}
        manifest.append(row)
        (investor_pdfs if audience=='investor' else internal_pdfs).append(pdf)
    readme_md=OUT/'README.md'; readme_pdf=OUT/'00_README.pdf'
    build_cover(readme_md, readme_pdf, [m for m in manifest if not m.get('missing')])
    merge_pdfs([readme_pdf]+investor_pdfs, COMBINED/'BTC_Vol_Desk_INVESTOR_DILIGENCE_PACKET.pdf')
    merge_pdfs([readme_pdf]+internal_pdfs, COMBINED/'BTC_Vol_Desk_INTERNAL_TEAM_PACKET.pdf')
    merge_pdfs([readme_pdf]+investor_pdfs+internal_pdfs, COMBINED/'BTC_Vol_Desk_FULL_PACKET.pdf')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    zip_path = ART/'btc-vol-desk-pdf-handoff-packet.zip'
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(OUT))
    print(json.dumps({
        'ok': True,
        'out_dir': str(OUT),
        'zip_path': str(zip_path),
        'zip_bytes': zip_path.stat().st_size,
        'pdf_count': len([m for m in manifest if not m.get('missing')]) + 4,
        'missing': [m for m in manifest if m.get('missing')],
        'combined': [str((COMBINED/name).relative_to(ROOT)) for name in ['BTC_Vol_Desk_INVESTOR_DILIGENCE_PACKET.pdf','BTC_Vol_Desk_INTERNAL_TEAM_PACKET.pdf','BTC_Vol_Desk_FULL_PACKET.pdf']],
    }, indent=2))

if __name__ == '__main__':
    main()
