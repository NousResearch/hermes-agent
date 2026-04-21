from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable


_ARTICLE_BLOCK_RE = re.compile(r"^=== ARTICLE CODE \d+ ===$", re.MULTILINE)
_ARTICLE_HEADER_RE = re.compile(r"^Article\s+([^\n]+)", re.MULTILINE)
_ARTICLE_NUMBER_RE = re.compile(r"Num[ée]ro article:\s*([^\n]+)")
_START_RE = re.compile(r"Date d[ée]but vigueur:\s*(\d{2}/\d{2}/\d{4})")
_END_RE = re.compile(r"Date fin vigueur:\s*(\d{2}/\d{2}/\d{4})")
_URL_RE = re.compile(r"https://www\.legifrance\.gouv\.fr/[^\s)]+")
_TEXT_ID_RE = re.compile(r"\b(?:LEGIARTI|LEGITEXT)\d+\b")
_SPACES_RE = re.compile(r"\s+")


def normalize_fact_date(value: str) -> str:
    return date.fromisoformat(str(value).strip()).isoformat()


def canonical_article_ref(value: str) -> str:
    normalized = str(value or "").strip().upper()
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    normalized = _SPACES_RE.sub(" ", normalized)
    return normalized


def parse_ddmmyyyy(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%d/%m/%Y").date()


def split_article_blocks(text: str) -> list[str]:
    if "=== ARTICLE CODE 1 ===" not in text:
        return []

    matches = list(_ARTICLE_BLOCK_RE.finditer(text))
    if not matches:
        return []

    blocks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[start:end].strip())
    return blocks


def extract_block_article_refs(block: str) -> set[str]:
    refs: set[str] = set()

    header_match = _ARTICLE_HEADER_RE.search(block)
    if header_match:
        refs.add(canonical_article_ref(header_match.group(1)))

    number_match = _ARTICLE_NUMBER_RE.search(block)
    if number_match:
        refs.add(canonical_article_ref(number_match.group(1)))

    return {ref for ref in refs if ref}


def extract_block_vigueur_window(block: str) -> tuple[str | None, str | None]:
    start_match = _START_RE.search(block)
    end_match = _END_RE.search(block)
    return (
        start_match.group(1) if start_match else None,
        end_match.group(1) if end_match else None,
    )


def extract_block_source_url(block: str) -> str | None:
    match = _URL_RE.search(block)
    return match.group(0) if match else None


def extract_block_source_ids(block: str) -> list[str]:
    return sorted(set(_TEXT_ID_RE.findall(block)))


def block_matches_article_ref(block: str, article_ref: str) -> bool:
    target = canonical_article_ref(article_ref)
    return target in extract_block_article_refs(block)


def block_matches_fact_date(block: str, fact_date: str) -> bool:
    target = date.fromisoformat(normalize_fact_date(fact_date))
    raw_start, raw_end = extract_block_vigueur_window(block)
    start = parse_ddmmyyyy(raw_start)
    end = parse_ddmmyyyy(raw_end)

    if start is None and end is None:
        return False
    if start is not None and target < start:
        return False
    if end is not None and target > end:
        return False
    return True


def build_rechercher_code_args(
    article_ref: str,
    code_name: str,
    max_candidate_blocks: int = 20,
) -> dict[str, object]:
    return {
        "search": canonical_article_ref(article_ref),
        "code_name": str(code_name).strip(),
        "champ": "NUM_ARTICLE",
        "type_recherche": "EXACTE",
        "sort": "DATE_DESC",
        "page_number": 1,
        "page_size": max_candidate_blocks,
    }


@dataclass(frozen=True)
class CodeArticleVersion:
    article_ref: str
    code_name: str
    fact_date: str
    vigueur_start: str | None
    vigueur_end: str | None
    text: str
    source_url: str | None
    source_ids: tuple[str, ...]
    selection_mode: str


def select_code_article_version(
    raw_text: str,
    article_ref: str,
    code_name: str,
    fact_date: str,
) -> CodeArticleVersion | None:
    normalized_fact_date = normalize_fact_date(fact_date)
    target_article = canonical_article_ref(article_ref)

    matching_blocks: list[str] = [
        block
        for block in split_article_blocks(raw_text)
        if block_matches_article_ref(block, target_article)
    ]
    if not matching_blocks:
        return None

    valid_blocks = [
        block for block in matching_blocks
        if block_matches_fact_date(block, normalized_fact_date)
    ]
    if not valid_blocks:
        return None

    def _sort_key(block: str) -> tuple[date, date]:
        raw_start, raw_end = extract_block_vigueur_window(block)
        start = parse_ddmmyyyy(raw_start) or date.min
        end = parse_ddmmyyyy(raw_end) or date.max
        return (start, end)

    selected = sorted(valid_blocks, key=_sort_key, reverse=True)[0]
    vigueur_start, vigueur_end = extract_block_vigueur_window(selected)

    return CodeArticleVersion(
        article_ref=target_article,
        code_name=str(code_name).strip(),
        fact_date=normalized_fact_date,
        vigueur_start=vigueur_start,
        vigueur_end=vigueur_end,
        text=selected.strip(),
        source_url=extract_block_source_url(selected),
        source_ids=tuple(extract_block_source_ids(selected)),
        selection_mode="exact_num_article+vigueur_window",
    )


def iter_candidate_article_refs(blocks: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for block in blocks:
        for ref in extract_block_article_refs(block):
            if ref not in seen:
                seen.add(ref)
                ordered.append(ref)
    return ordered
