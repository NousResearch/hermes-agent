#!/usr/bin/env python3
"""Search PubMed and print readable results.

Examples:
    python3 search_pubmed.py "crispr base editing" --max 5
    python3 search_pubmed.py --author "Jennifer Doudna" --max 5
    python3 search_pubmed.py "checkpoint inhibitors" --mesh "Neoplasms" --since 2022
    python3 search_pubmed.py --pmid 41256272,41248061
    python3 search_pubmed.py --abstract 41256272
    python3 search_pubmed.py --related 41256272 --max 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "HermesAgent"


def _api_params() -> dict[str, str]:
    params = {"tool": TOOL_NAME}
    email = (os.getenv("NCBI_EMAIL") or "").strip()
    api_key = (os.getenv("NCBI_API_KEY") or "").strip()
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def _request_json(endpoint: str, params: dict[str, str]) -> dict:
    return json.loads(_request_bytes(endpoint, params).decode("utf-8"))


def _request_xml(endpoint: str, params: dict[str, str]) -> ET.Element:
    return ET.fromstring(_request_bytes(endpoint, params))


def _request_bytes(endpoint: str, params: dict[str, str]) -> bytes:
    payload = dict(_api_params())
    payload.update(params)
    url = f"{BASE_URL}/{endpoint}?{urllib.parse.urlencode(payload)}"
    request = urllib.request.Request(url, headers={"User-Agent": f"{TOOL_NAME}/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else float(attempt + 1)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _build_term(args: argparse.Namespace) -> str:
    if args.raw_query:
        return args.raw_query

    parts: list[str] = []
    if args.query:
        parts.append(f"({args.query})")
    if args.author:
        parts.append(f"({args.author}[Author])")
    if args.title:
        parts.append(f"({args.title}[Title])")
    if args.journal:
        parts.append(f"({args.journal}[Journal])")
    if args.mesh:
        parts.append(f"({args.mesh}[MeSH Terms])")

    if not parts:
        raise ValueError(
            "Provide a query, --author, --title, --journal, --mesh, "
            "--raw-query, --pmid, --abstract, or --related."
        )
    return " AND ".join(parts)


def search_ids(args: argparse.Namespace) -> list[str]:
    sort_map = {
        "relevance": "relevance",
        "date": "pub date",
    }
    params = {
        "db": "pubmed",
        "term": _build_term(args),
        "retmode": "json",
        "retmax": str(args.max),
        "sort": sort_map[args.sort],
    }
    if args.since:
        params["mindate"] = str(args.since)
        params["datetype"] = "pdat"
    if args.until:
        params["maxdate"] = str(args.until)
        params["datetype"] = "pdat"

    data = _request_json("esearch.fcgi", params)
    return list(data.get("esearchresult", {}).get("idlist", []))


def fetch_summaries(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    data = _request_json(
        "esummary.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        },
    )
    result = data.get("result", {})
    ordered = []
    for pmid in ids:
        record = result.get(pmid)
        if isinstance(record, dict):
            ordered.append(record)
    return ordered


def fetch_related_ids(pmid: str, max_results: int) -> list[str]:
    data = _request_json(
        "elink.fcgi",
        {
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "linkname": "pubmed_pubmed",
            "retmode": "json",
        },
    )
    linksets = data.get("linksets", [])
    if not linksets:
        return []
    dbs = linksets[0].get("linksetdbs", [])
    if not dbs:
        return []
    links = dbs[0].get("links", [])
    return [str(link) for link in links[:max_results]]


def fetch_abstracts(ids: list[str]) -> list[dict[str, str]]:
    if not ids:
        return []
    root = _request_xml(
        "efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        },
    )
    records: list[dict[str, str]] = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("./MedlineCitation")
        article_node = citation.find("./Article") if citation is not None else None
        title = _collect_text(article_node.find("./ArticleTitle")) if article_node is not None else ""
        pmid = _collect_text(citation.find("./PMID")) if citation is not None else ""
        journal = _collect_text(article_node.find("./Journal/Title")) if article_node is not None else ""
        pub_date = _extract_pub_date(article_node) if article_node is not None else ""
        abstract = _extract_abstract(article_node)
        records.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "pub_date": pub_date,
                "abstract": abstract,
            }
        )
    return records


def _extract_pub_date(article_node: ET.Element | None) -> str:
    if article_node is None:
        return ""
    pub_date = article_node.find("./Journal/JournalIssue/PubDate")
    if pub_date is None:
        return ""
    parts = [
        _collect_text(pub_date.find("./Year")),
        _collect_text(pub_date.find("./Month")),
        _collect_text(pub_date.find("./Day")),
    ]
    parts = [part for part in parts if part]
    medline_date = _collect_text(pub_date.find("./MedlineDate"))
    return " ".join(parts) or medline_date


def _extract_abstract(article_node: ET.Element | None) -> str:
    if article_node is None:
        return ""
    abstract_node = article_node.find("./Abstract")
    if abstract_node is None:
        return ""

    blocks: list[str] = []
    for item in abstract_node.findall("./AbstractText"):
        label = (item.attrib.get("Label") or "").strip()
        text = _collect_text(item).strip()
        if not text:
            continue
        if label:
            blocks.append(f"{label}: {text}")
        else:
            blocks.append(text)
    return "\n\n".join(blocks)


def _collect_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _article_ids(record: dict) -> dict[str, str]:
    article_ids: dict[str, str] = {}
    for item in record.get("articleids", []):
        id_type = str(item.get("idtype") or "").strip().lower()
        value = str(item.get("value") or "").strip()
        if id_type and value:
            article_ids[id_type] = value
    return article_ids


def print_summaries(records: list[dict]) -> None:
    if not records:
        print("No results found.")
        return

    for index, record in enumerate(records, start=1):
        article_ids = _article_ids(record)
        authors = [author.get("name", "") for author in record.get("authors", []) if author.get("name")]
        author_text = ", ".join(authors[:6])
        if len(authors) > 6:
            author_text += ", et al."

        pmid = str(record.get("uid") or "")
        title = str(record.get("title") or "").strip()
        journal = str(record.get("fulljournalname") or record.get("source") or "").strip()
        pub_date = str(record.get("pubdate") or record.get("epubdate") or "").strip()
        doi = article_ids.get("doi", "")
        pmc = article_ids.get("pmc", "")

        print(f"{index}. {title}")
        print(f"   PMID: {pmid}")
        print(f"   Journal: {journal or 'Unknown'} | Date: {pub_date or 'Unknown'}")
        if author_text:
            print(f"   Authors: {author_text}")
        extras = []
        if doi:
            extras.append(f"DOI: {doi}")
        if pmc:
            extras.append(f"PMC: {pmc}")
        if extras:
            print(f"   IDs: {' | '.join(extras)}")
        if pmid:
            print(f"   Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        print()


def print_abstracts(records: list[dict[str, str]]) -> None:
    if not records:
        print("No abstract found.")
        return

    for index, record in enumerate(records, start=1):
        print(f"{index}. {record['title']}")
        print(f"   PMID: {record['pmid']}")
        if record["journal"] or record["pub_date"]:
            print(
                f"   Journal: {record['journal'] or 'Unknown'}"
                f" | Date: {record['pub_date'] or 'Unknown'}"
            )
        print()
        print(record["abstract"] or "[No abstract text available]")
        print()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search PubMed via NCBI E-utilities.")
    parser.add_argument("query", nargs="?", help="General PubMed search query")
    parser.add_argument("--author", help="Restrict to author field")
    parser.add_argument("--title", help="Restrict to title field")
    parser.add_argument("--journal", help="Restrict to journal field")
    parser.add_argument("--mesh", help="Add a MeSH term filter")
    parser.add_argument("--raw-query", help="Exact PubMed query string to send as-is")
    parser.add_argument("--since", type=int, help="Minimum publication year")
    parser.add_argument("--until", type=int, help="Maximum publication year")
    parser.add_argument("--sort", choices=["relevance", "date"], default="relevance")
    parser.add_argument("--max", type=int, default=5, help="Maximum number of results")
    parser.add_argument("--pmid", help="Comma-separated PMIDs to summarize directly")
    parser.add_argument("--abstract", help="Comma-separated PMIDs to fetch full abstracts for")
    parser.add_argument("--related", help="Seed PMID for related-paper expansion")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        if args.abstract:
            ids = [item.strip() for item in args.abstract.split(",") if item.strip()]
            print_abstracts(fetch_abstracts(ids))
            return 0

        if args.related:
            related_ids = fetch_related_ids(args.related.strip(), args.max)
            print_summaries(fetch_summaries(related_ids))
            return 0

        if args.pmid:
            ids = [item.strip() for item in args.pmid.split(",") if item.strip()]
            print_summaries(fetch_summaries(ids))
            return 0

        ids = search_ids(args)
        print_summaries(fetch_summaries(ids))
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except urllib.error.HTTPError as exc:
        print(f"HTTP error from NCBI: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error talking to NCBI: {exc.reason}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Failed to parse JSON response: {exc}", file=sys.stderr)
        return 1
    except ET.ParseError as exc:
        print(f"Failed to parse XML response: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
