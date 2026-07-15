#!/usr/bin/env python3
"""ตรวจความพร้อมชั้น H/U/F ใน .project/DesignSystem.md โดยใช้ stdlib เท่านั้น."""

import argparse
import json
import re
import sys
from pathlib import Path


HEADINGS = {
    "H0": "## H0 เอกสารโปรเจกต์ที่อ่าน",
    "H1": "## H1 บัตรประจำตัวโครงการ",
    "H2": "## H2 เหตุผลการเลือกสี",
    "H3": "## H3 Pain → Design Response",
    "H4": "## H4 สองภาษา",
    "H5": "## H5 Direction Check",
    "H6": "## H6 ความสอดคล้องข้ามเอกสาร",
    "H7": "## H7 ลายเซ็นความเป็นคน",
    "D17": "## ข้อห้ามที่เกี่ยว (D17)",
    "U1": "## U1 กฎ UX 7 ข้อ",
    "U2": "## U2 มาตรฐาน Flow",
    "U3": "## U3 ทดสอบ 5 วินาที",
    "U4": "## U4 Anti-patterns",
    "F1": "## F1 Branding",
    "F2": "## F2 Target",
    "F3": "## F3 Persona + Floor",
    "F4": "## F4 Motivation",
    "F5": "## F5 Emotion 6 แกน",
    "F6": "## F6 Function → Component",
    "F7": "## F7 Mood & Tone",
}

LAYER_CODES = {
    "H": ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"),
    "U": ("U1", "U2", "U3", "U4"),
    "F": ("F1", "F2", "F3", "F4", "F5", "F6", "F7"),
    "D": ("D17",),
}

ARCHETYPES = (
    "Ruler", "Sage", "Caregiver", "Explorer", "Hero", "Creator",
    "Everyman", "Lover", "Jester", "Magician", "Outlaw", "Innocent",
)

TEMPLATE = """# Design System

> กรอกทุกช่อง TODO ให้เป็นข้อมูลจริงก่อนเริ่มสร้าง token ชั้น A

## H0 เอกสารโปรเจกต์ที่อ่าน
- เอกสารที่อ่าน/แหล่ง: TODO
- ผล conflict: TODO (เขียน "ไม่พบขัดกัน" หรือ "พบขัดกัน" พร้อมรายการ)

## H1 บัตรประจำตัวโครงการ
| รายการ | ค่า |
|---|---|
| ชื่อโครงการ/แบรนด์ | TODO |
| วิสัยทัศน์/positioning | TODO |
| พันธกิจ/สิ่งที่ทำ | TODO |
| ภาษาหลัก | TODO |
| เฟสปัจจุบัน | TODO |

## H2 เหตุผลการเลือกสี
| สี | ความหมาย | เชื่อมพันธกิจยังไง | ใช้ที่ไหน |
|---|---|---|---|
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## H3 Pain → Design Response
| Pain | Design Response |
|---|---|
| TODO | TODO |
| TODO | TODO |
| TODO | TODO |

## H4 สองภาษา
- ภาษานำ: TODO
- ปุ่มสลับภาษา: TODO
- microcopy/ข้อความ 2 ภาษา: TODO
- ฟอนต์ TH/ไทย: TODO
- font EN/อังกฤษ: TODO

## H5 Direction Check
| ข้อ | คำตอบ | เหตุผล |
|---|---|---|
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |
| 4 | TODO | TODO |

## H6 ความสอดคล้องข้ามเอกสาร
- แหล่งที่เทียบ: TODO
- ผลตรวจ: TODO (ใส่ตาราง conflict หรือข้อความ "ไม่พบขัดกัน")

## H7 ลายเซ็นความเป็นคน
- TODO: ลายเซ็นข้อที่ 1
- TODO: ลายเซ็นข้อที่ 2

## ข้อห้ามที่เกี่ยว (D17)
- TODO: ใส่ข้อห้ามที่เกี่ยว หรือเขียนว่า "ไม่มี"

## U1 กฎ UX 7 ข้อ
| กฎ | ใช้กับหน้าไหน |
|---|---|
| Hick | TODO |
| Fitts | TODO |
| Jakob | TODO |
| Miller | TODO |
| Tesler | TODO |
| Doherty | TODO |
| Aesthetic | TODO |

## U2 มาตรฐาน Flow
Flow สำคัญ: TODO
- entry: TODO
- next: TODO
- progress: TODO
- error recovery: TODO
- success: TODO
- no dead end: TODO

## U3 ทดสอบ 5 วินาที
หน้าสำคัญ: TODO
1. อยู่ไหน: TODO
2. ทำอะไรได้: TODO
3. อะไรสำคัญ: TODO
4. กดอะไรต่อ: TODO
5. ถอยทางไหน: TODO
โครงหน้า 6 ส่วน:
- Context/บริบท: TODO
- Primary action/แอ็กชันหลัก: TODO
- Key information/ข้อมูลสำคัญ: TODO
- Details/รายละเอียด: TODO
- Support/ตัวช่วย: TODO
- Exit/ทางออก: TODO

## U4 Anti-patterns
- พระเอกของหน้า: TODO
- ลำดับสายตา 1-2-3: TODO

## F1 Branding
- Archetype: TODO
- ปรับ: TODO%

## F2 Target
- กลุ่มเป้าหมายหลัก: TODO
- บริบทใช้งาน: TODO

## F3 Persona + Floor
- Persona: TODO
- ฟอนต์ขั้นต่ำ: TODOpx
- contrast: TODO
- target size: TODOpx

## F4 Motivation
- แรงจูงใจหลัก: TODO
- CTA ที่ตอบ: TODO

## F5 Emotion 6 แกน
| แกน | คะแนน 1-5 | ค่า token ที่แปลงแล้วอย่างน้อย 3 ค่า |
|---|---|---|
| ทางการ | TODO | TODO |
| อบอุ่น | TODO | TODO |
| พลัง | TODO | TODO |
| ชัด | TODO | TODO |
| หนาแน่น | TODO | TODO |

## F6 Function → Component
| ฟังก์ชันหลัก | ชุด component |
|---|---|
| TODO | TODO |
| TODO | TODO |

## F7 Mood & Tone
Mood: TODO

| Touchpoint | Do | Don't |
|---|---|---|
| เว็บ | TODO | TODO |
| แอดมิน | TODO | TODO |
"""


class ThaiArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, "ใช้คำสั่งไม่ถูกต้อง: ค่าหรือรูปแบบอาร์กิวเมนต์ไม่ถูกต้อง\nดูรูปแบบด้วย --help หรือสร้างแม่แบบด้วย --init\n")


def split_sections(text):
    sections = {}
    current = None
    by_heading = {heading: code for code, heading in HEADINGS.items()}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in by_heading:
            current = by_heading[stripped]
            sections[current] = []
        elif stripped.startswith("## "):
            current = None
        elif current is not None:
            sections[current].append(line)
    return sections


def table_rows(lines):
    candidates = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        candidates.append((index, cells))

    separator_indexes = set()
    header_indexes = set()
    for index, cells in candidates:
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            separator_indexes.add(index)
            header_indexes.add(index - 1)
    return [cells for index, cells in candidates
            if index not in separator_indexes and index not in header_indexes]


def nonempty_value(text, label_pattern):
    match = re.search(label_pattern + r"\s*[:|]\s*(.+)", text, re.IGNORECASE)
    return bool(match and match.group(1).strip(" |-*\t"))


def issue(code, detail):
    return {"code": code, "message": "[%s] %s" % (code, detail)}


def validate_h0(lines):
    body = "\n".join(lines)
    document_match = re.search(
        r"(?:เอกสารที่อ่าน|ชื่อไฟล์|แหล่ง)\s*[:|]\s*([^\n|]+)", body, re.I
    )
    document_value = document_match.group(1).strip(" -*\t") if document_match else ""
    has_document = bool(document_value and not re.search(r"^(?:ไม่มี|ไม่พบ)(?:\s|$)", document_value))
    no_documents = bool(re.search(r"(?:เอกสารที่อ่าน|ชื่อไฟล์|แหล่ง)\s*[:|]\s*(?:ไม่มี|ไม่พบ)", body, re.I))

    if not has_document and not no_documents:
        return "ต้องระบุเอกสารที่อ่านอย่างน้อย 1 รายการ หรือเขียนว่าไม่มีเอกสาร"
    if no_documents:
        owner_answers = ("พันธกิจ", "กลุ่มเป้าหมาย", "ปัญหาหลัก", "เฟสงาน")
        missing = [label for label in owner_answers if not nonempty_value(body, re.escape(label))]
        if missing:
            return "ไม่มีเอกสารจึงต้องมีคำตอบเจ้าของ 4 ข้อ: พันธกิจ/กลุ่มเป้าหมาย/ปัญหาหลัก/เฟสงาน"

    if "ไม่พบขัดกัน" in body:
        return None
    conflict = re.search(r"พบขัดกัน\s*[:：-]\s*(\S.{4,})", body)
    if conflict:
        return None
    return "ต้องระบุผล conflict ว่าไม่พบขัดกัน หรือพบขัดกันพร้อมรายการ"


def validate_h1(lines):
    rows = table_rows(lines)
    values = {row[0].lower(): row[1].strip() for row in rows if len(row) >= 2}
    required = ("ชื่อโครงการ/แบรนด์", "วิสัยทัศน์/positioning", "พันธกิจ/สิ่งที่ทำ", "ภาษาหลัก", "เฟสปัจจุบัน")
    missing = [name for name in required if not values.get(name.lower())]
    if missing:
        return "ตารางบัตรโครงการขาดค่าที่ต้องมี: %s" % ", ".join(missing)
    short = [name for name in required[1:3] if len(values[name.lower()]) < 20]
    if short:
        return "%s ต้องยาวอย่างน้อย 20 ตัวอักษร" % " และ ".join(short)
    return None


def validate_table(lines, minimum, columns, label):
    rows = table_rows(lines)
    valid = [row for row in rows if len(row) >= columns and all(cell.strip() for cell in row[:columns])]
    if len(valid) < minimum:
        return "%s ต้องมี >= %d แถวข้อมูล — พบ %d" % (label, minimum, len(valid))
    return None


def validate_h4(lines):
    body = "\n".join(lines)
    checks = (
        nonempty_value(body, r"ภาษานำ"),
        bool(re.search(r"(?:ปุ่มสลับ|สลับภาษา)", body, re.I)),
        bool(re.search(r"(?:microcopy|ข้อความ\s*2\s*ภาษา)", body, re.I)),
        bool(re.search(r"(?:ฟอนต์|font)[^\n]*(?:TH|ไทย)|(?:TH|ไทย)[^\n]*(?:ฟอนต์|font)", body, re.I)),
        bool(re.search(r"(?:ฟอนต์|font)[^\n]*(?:EN|อังกฤษ)|(?:EN|อังกฤษ)[^\n]*(?:ฟอนต์|font)", body, re.I)),
    )
    return None if all(checks) else "ต้องมีภาษานำ ปุ่มสลับภาษา microcopy/ข้อความ 2 ภาษา และคู่ฟอนต์ TH/EN ให้ครบ"


def validate_h5(lines):
    answered = 0
    for row in table_rows(lines):
        if len(row) >= 3 and re.fullmatch(r"(?:ผ่าน|ไม่ผ่าน)", row[1].strip()) and row[2].strip():
            answered += 1
    if answered == 0:
        for line in lines:
            if (re.match(r"\s*(?:[-*]|\d+[.)])\s+", line)
                    and re.search(r"(?:ผ่าน|ไม่ผ่าน)", line)
                    and re.search(r"(?:เหตุผล|เพราะ|—|:)\s*\S+", line)):
                answered += 1
    if answered < 4:
        return "Direction Check ต้องตอบผ่าน/ไม่ผ่านพร้อมเหตุผลครบ 4 ข้อ — พบ %d" % answered
    return None


def validate_h6(lines):
    body = "\n".join(lines)
    source_match = re.search(r"(?:แหล่งที่เทียบ|ชื่อไฟล์|เอกสาร)\s*[:|]\s*([^\n|]+)", body, re.I)
    source_value = source_match.group(1).strip(" -*\t") if source_match else ""
    named_source = bool(
        source_value
        and not re.search(r"^(?:ไม่มี|ไม่พบขัดกัน|พบขัดกัน)(?:\s|$)", source_value)
    )
    named_file = bool(re.search(r"\b[^\s|]+\.(?:md|txt|pdf|docx?|ya?ml|json|css|tsx?|jsx?|html?)\b", body, re.I))
    if (named_source or named_file) and ("ไม่พบขัดกัน" in body or table_rows(lines)):
        return None
    return "ต้องระบุชื่อแหล่งที่เทียบอย่างน้อย 1 แหล่ง และมีตาราง conflict หรือข้อความ \"ไม่พบขัดกัน\""


def validate_h7(lines):
    body = " ".join(line.strip(" -*\t") for line in lines).strip()
    entries = [line for line in lines if re.match(r"\s*(?:[-*]|\d+[.)])\s+", line)]
    entries.extend(" | ".join(row) for row in table_rows(lines))
    if len(entries) < 2 or len(body) < 60:
        return "ลายเซ็นความเป็นคนต้องมีอย่างน้อย 2 รายการและเนื้อหารวมอย่างน้อย 60 ตัวอักษร"
    return None


def validate_content(lines, label):
    body = " ".join(line.strip(" -*\t") for line in lines).strip()
    return None if body else "%s ต้องมีเนื้อหา" % label


def validate_u1(lines):
    laws = ("Hick", "Fitts", "Jakob", "Miller", "Tesler", "Doherty", "Aesthetic")
    entries = [" ".join(row) for row in table_rows(lines)]
    entries.extend(line for line in lines if re.match(r"\s*(?:[-*]|\d+[.)])\s+", line))
    found = []
    for law in laws:
        if any(law.lower() in entry.lower() and "หน้า" in entry for entry in entries):
            found.append(law)
    if len(found) < 7:
        return "ต้องอ้างกฎ UX พร้อมหน้าที่ใช้ครบ 7 ข้อ — พบ %d" % len(found)
    return None


def validate_u2(lines):
    body = "\n".join(lines).lower()
    required = ("entry", "next", "progress", "error recovery", "success", "no dead end")
    found = sum(1 for term in required if nonempty_value(body, re.escape(term)))
    has_named_flow = any(
        re.search(r"\bflow\b", line, re.I) and re.search(r"(?:[:：]|\s)\s*\S+", line)
        for line in lines
    )
    if found < 6 or not has_named_flow:
        return "Flow สำคัญต้องระบุครบ 6 อย่าง — พบ %d" % found
    return None


def validate_u3(lines):
    body = "\n".join(lines)
    questions = ("อยู่ไหน", "ทำอะไรได้", "อะไรสำคัญ", "กดอะไรต่อ", "ถอยทางไหน")
    found = sum(1 for term in questions if nonempty_value(body, re.escape(term)))
    has_page = nonempty_value(body, r"หน้าสำคัญ")
    page_parts = (
        ("Context", "บริบท"),
        ("Primary action", "แอ็กชันหลัก"),
        ("Key information", "ข้อมูลสำคัญ"),
        ("Details", "รายละเอียด"),
        ("Support", "ตัวช่วย"),
        ("Exit", "ทางออก"),
    )
    found_parts = sum(
        1 for terms in page_parts
        if any(nonempty_value(body, re.escape(term)) for term in terms)
    )
    if found < 5 or not has_page or found_parts < 6:
        return "ทดสอบ 5 วินาทีต้องตอบครบ 5 คำถามและโครงหน้า 6 ส่วน — พบคำตอบ %d, ส่วนหน้า %d" % (found, found_parts)
    return None


def validate_u4(lines):
    body = "\n".join(lines)
    if "หน้า" not in body or "พระเอก" not in body or not re.search(r"1\D+2\D+3", body):
        return "ต้องระบุพระเอกของหน้าและลำดับสายตา 1-2-3 อย่างน้อย 1 หน้า"
    return None


def validate_f1(lines):
    body = "\n".join(lines)
    archetype = next((name for name in ARCHETYPES if re.search(r"\b%s\b" % name, body, re.I)), None)
    percentages = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", body)]
    if not archetype or not percentages or percentages[0] > 20:
        return "ต้องเลือก archetype 1 จาก 12 แบบและระบุเปอร์เซ็นต์ปรับไม่เกิน 20%"
    return None


def validate_f2(lines):
    body = "\n".join(lines)
    if not nonempty_value(body, r"กลุ่มเป้าหมายหลัก") or not nonempty_value(body, r"บริบทใช้งาน"):
        return "ต้องระบุกลุ่มเป้าหมายหลักและบริบทใช้งาน"
    return None


def validate_f3(lines):
    body = "\n".join(lines)
    persona = nonempty_value(body, r"Persona")
    font = re.search(r"(?:ฟอนต์|font)[^\n]*?\d+(?:\.\d+)?\s*px", body, re.I)
    contrast = re.search(r"contrast[^\n]*?\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)?", body, re.I)
    target = re.search(r"target(?:\s+size)?[^\n]*?\d+(?:\.\d+)?\s*px", body, re.I)
    if not all((persona, font, contrast, target)):
        return "ต้องมี persona และค่าขั้นต่ำเป็นตัวเลข: ฟอนต์ px, contrast, target size px"
    return None


def validate_f4(lines):
    body = "\n".join(lines)
    if not nonempty_value(body, r"แรงจูงใจหลัก") or not nonempty_value(body, r"CTA(?: ที่ตอบ)?"):
        return "ต้องระบุแรงจูงใจหลัก 1-2 ตัวและ CTA ที่ตอบแรงจูงใจ"
    return None


def token_values_with_units(text):
    return re.findall(
        r"\d+(?:\.\d+)?\s*(?:px|ms|%|:1|×|°|deg)(?=\s|[,;|)]|$)",
        text,
        re.I,
    )


def validate_f5(lines):
    axes = ("ทางการ", "อบอุ่น", "พลัง", "ชัด", "หนาแน่น")
    valid = 0
    for axis in axes:
        for row in table_rows(lines):
            if not row or axis not in row[0] or len(row) < 3:
                continue
            score = re.search(r"(?<!\d)([1-5])(?!\d)", row[1])
            token_values = token_values_with_units(" ".join(row[2:]))
            if score and len(token_values) >= 3:
                valid += 1
                break
    if valid < 5:
        return "ตารางอารมณ์ต้องมี >= 5 แกน พร้อมคะแนน 1-5 และค่า token ที่มีหน่วย >= 3 ค่า — พบ %d แกน" % valid
    return None


def validate_f7(lines):
    body = "\n".join(lines)
    mood_match = re.search(r"Mood\s*:\s*([^\n]+)", body, re.I)
    moods = []
    if mood_match:
        moods = [part.strip() for part in re.split(r"[,，·/]+|\s{2,}", mood_match.group(1)) if part.strip()]
        if len(moods) == 1:
            moods = [part for part in moods[0].split() if part]
    rows = [row for row in table_rows(lines) if len(row) >= 2 and row[0].strip() and row[1].strip()]
    touchpoints = sum(1 for term in ("เว็บ", "แอดมิน", "เอกสาร") if term in body)
    if not 5 <= len(moods) <= 8 or len(rows) < 2 or touchpoints < 2:
        return "ต้องมีคำ mood 5-8 คำ ตาราง do/don't >= 2 แถว และ touchpoint >= 2 จากเว็บ/แอดมิน/เอกสาร — พบ mood %d คำ, ตาราง %d แถว, touchpoint %d" % (len(moods), len(rows), touchpoints)
    return None


def validate_f6(lines):
    table_error = validate_table(lines, 2, 2, "Function → Component")
    if table_error is None:
        return None
    list_rows = 0
    for line in lines:
        if not re.match(r"\s*(?:[-*]|\d+[.)])\s+", line):
            continue
        parts = re.split(r"\s*(?:→|->)\s*", line, maxsplit=1)
        if len(parts) == 2 and parts[0].strip(" -*\t") and parts[1].strip():
            list_rows += 1
    if list_rows >= 2:
        return None
    return "Function → Component ต้องมี >= 2 แถวข้อมูล — พบ %d" % list_rows


VALIDATORS = {
    "H0": validate_h0,
    "H1": validate_h1,
    "H2": lambda lines: validate_table(lines, 2, 4, "ตารางเหตุผลสี"),
    "H3": lambda lines: validate_table(lines, 3, 2, "ตาราง Pain → Design Response"),
    "H4": validate_h4,
    "H5": validate_h5,
    "H6": validate_h6,
    "H7": validate_h7,
    "D17": lambda lines: validate_content(lines, "ข้อห้ามที่เกี่ยว (D17)"),
    "U1": validate_u1,
    "U2": validate_u2,
    "U3": validate_u3,
    "U4": validate_u4,
    "F1": validate_f1,
    "F2": validate_f2,
    "F3": validate_f3,
    "F4": validate_f4,
    "F5": validate_f5,
    "F6": validate_f6,
    "F7": validate_f7,
}


def validate(text, layer):
    sections = split_sections(text)
    codes = tuple(code for name in ("H", "U", "F", "D") for code in LAYER_CODES[name]) if layer == "all" else LAYER_CODES[layer]
    errors = []
    for code in codes:
        if code not in sections:
            errors.append(issue(code, "ไม่พบหัวข้อตายตัว: %s" % HEADINGS[code]))
            continue
        lines = sections[code]
        if any("TODO" in line for line in lines):
            errors.append(issue(code, "ยังมี TODO ในหัวข้อนี้"))
            continue
        detail = VALIDATORS[code](lines)
        if code == "F5" and not detail and re.search(r"(?:Premium|Luxe)", text, re.I):
            luxe_rows = [row for row in table_rows(lines) if row and re.search(r"(?:Luxe|หรู)", row[0], re.I)]
            luxe_valid = any(
                len(row) >= 3
                and re.search(r"(?<!\d)([1-5])(?!\d)", row[1])
                and len(token_values_with_units(" ".join(row[2:]))) >= 3
                for row in luxe_rows
            )
            if not luxe_valid:
                detail = "มี Premium/Luxe จึงต้องมีแกน Luxe พร้อมคะแนนและค่า token ที่มีหน่วย >= 3 ค่า"
        if detail:
            errors.append(issue(code, detail))
    return codes, errors


def print_result(payload, as_json):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif payload.get("ok"):
        print("ผ่านทุกข้อของชั้น %s (%d/%d ข้อ)" % (
            payload["layer"], len(payload["checked"]), len(payload["checked"])))
    else:
        for error in payload.get("errors", []):
            print(error["message"])


def parse_args(argv=None):
    parser = ThaiArgumentParser(description="ตรวจชั้นแบรนด์ H/U/F และข้อห้าม D ก่อนเริ่ม token ชั้น A")
    parser.add_argument("--file", default=".project/DesignSystem.md", help="ไฟล์ DesignSystem.md")
    parser.add_argument("--layer", choices=("H", "U", "F", "D", "all"), default="all")
    parser.add_argument("--json", action="store_true", help="แสดงผล JSON")
    parser.add_argument("--init", action="store_true", help="สร้างแม่แบบเมื่อไฟล์ยังไม่มี")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.file)

    if args.init:
        if path.exists():
            payload = {"ok": False, "action": "init", "file": str(path),
                       "errors": [issue("INIT", "มีไฟล์อยู่แล้ว จึงไม่เขียนทับ: %s" % path)]}
            print_result(payload, args.json)
            return 1
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(TEMPLATE, encoding="utf-8")
        except OSError as exc:
            payload = {"ok": False, "action": "init", "file": str(path),
                       "errors": [issue("INIT", "สร้างแม่แบบไม่ได้: %s" % exc)]}
            print_result(payload, args.json)
            return 2
        payload = {"ok": True, "action": "init", "file": str(path), "errors": []}
        if args.json:
            print_result(payload, True)
        else:
            print("สร้างแม่แบบแล้ว: %s\nกรอก TODO ให้ครบ แล้วรันตัวตรวจอีกครั้ง" % path)
        return 0

    if not path.is_file():
        payload = {"ok": False, "layer": args.layer, "file": str(path),
                   "errors": [issue("FILE", "ไม่พบไฟล์: %s — สร้างแม่แบบด้วย --init" % path)]}
        print_result(payload, args.json)
        return 2
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        payload = {"ok": False, "layer": args.layer, "file": str(path),
                   "errors": [issue("FILE", "อ่านไฟล์ไม่ได้: %s" % exc)]}
        print_result(payload, args.json)
        return 1

    checked, errors = validate(text, args.layer)
    payload = {"ok": not errors, "layer": args.layer, "file": str(path),
               "checked": list(checked), "errors": errors}
    print_result(payload, args.json)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
