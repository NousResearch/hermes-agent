"""Parse docx/xlsx/pdf/pptx from Hermes cache and output text summaries.

Usage:
  python3 parse_documents.py /path/to/cache/*.docx ...
  python3 parse_documents.py /Users/ray/.hermes/cache/documents/doc_*.docx

Handles:
  - docx: zipfile + xml.etree (no python-docx needed)
  - xlsx: zipfile + xml.etree direct XML read (no openpyxl needed)
  - pdf:  pymupdf (fitz)
  - pptx: zipfile + xml.etree

Outputs structured summaries to stdout — one section per document.
"""

import zipfile
import xml.etree.ElementTree as ET
import os
import sys

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def parse_docx(path):
    z = zipfile.ZipFile(path)
    tree = ET.fromstring(z.read("word/document.xml"))
    paragraphs = []
    for p in tree.iter(f"{{{W_NS}}}p"):
        texts = []
        for t in p.iter(f"{{{W_NS}}}t"):
            if t.text:
                texts.append(t.text)
        if texts:
            paragraphs.append("".join(texts))
    return "\n".join(paragraphs[:150])


def parse_xlsx(path):
    z = zipfile.ZipFile(path)
    shared = []
    try:
        ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in ss.iter(f"{{{S_NS}}}si"):
            t = si.find(f"{{{S_NS}}}t")
            shared.append(t.text if t is not None and t.text else "")
    except Exception:
        pass

    wb = ET.fromstring(z.read("xl/workbook.xml"))
    sheets = []
    for s in wb.iter(f"{{{S_NS}}}sheet"):
        sheets.append((s.get("name"), s.get("sheetId")))

    output = []
    for sname, sid in sheets:
        try:
            sx = ET.fromstring(z.read(f"xl/worksheets/sheet{int(sid)}.xml"))
        except Exception:
            try:
                sx = ET.fromstring(z.read(f"xl/worksheets/sheet{sid}.xml"))
            except Exception:
                continue
        rows = list(sx.iter(f"{{{S_NS}}}row"))
        output.append(f"\n--- Sheet: {sname} ({len(rows)} rows) ---")
        for row in rows[:60]:
            cells = []
            for c in row.iter(f"{{{S_NS}}}c"):
                v = c.find(f"{{{S_NS}}}v")
                if v is not None and v.text:
                    val = v.text
                    if c.get("t") == "s":
                        try:
                            val = shared[int(val)]
                        except Exception:
                            pass
                    cells.append(val.strip() if val else "")
                else:
                    cells.append("")
            line = " | ".join(cells)
            if line.strip():
                output.append(line[:400])
    return "\n".join(output)


def parse_pdf(path):
    try:
        import fitz
    except ImportError:
        return "ERROR: pymupdf not installed. Run: pip install pymupdf"

    doc = fitz.open(path)
    output = [f"页数: {doc.page_count}"]
    for i in range(min(doc.page_count, 10)):
        text = doc[i].get_text().strip()
        if text:
            output.append(f"\n--- 第{i+1}页 ---")
            for line in text.split("\n")[:60]:
                if line.strip():
                    output.append(f"  {line[:250]}")
        else:
            output.append(f"\n--- 第{i+1}页 [纯图片] ---")
    doc.close()
    return "\n".join(output)


def parse_pptx(path):
    z = zipfile.ZipFile(path)
    pres = ET.fromstring(z.read("ppt/presentation.xml"))
    slides = pres.findall(f".//{{{P_NS}}}sldId")
    output = [f"总页数: {len(slides)}"]
    for i in range(1, min(len(slides) + 1, 12)):
        try:
            sx = ET.fromstring(z.read(f"ppt/slides/slide{i}.xml"))
            texts = []
            for t in sx.iter(f"{{{A_NS}}}t"):
                if t.text:
                    texts.append(t.text)
            txt = " ".join(texts)
            if txt.strip():
                output.append(f"\n--- Slide {i} ---\n{txt[:400]}")
            else:
                output.append(f"\n--- Slide {i} [无文字] ---")
        except Exception:
            pass
    return "\n".join(output)


PARSERS = {
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
    ".pdf": parse_pdf,
    ".pptx": parse_pptx,
}


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python3 parse_documents.py <file>...")
        sys.exit(1)

    for path in paths:
        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        size_kb = os.path.getsize(path) / 1024

        print(f"\n{'='*70}")
        print(f"  {name} ({ext}, {size_kb:.0f} KB)")
        print(f"{'='*70}")

        parser = PARSERS.get(ext)
        if parser:
            try:
                result = parser(path)
                print(result[:8000])
            except Exception as e:
                print(f"ERROR parsing {name}: {e}")
        else:
            print(f"Unsupported format: {ext}")


if __name__ == "__main__":
    main()
