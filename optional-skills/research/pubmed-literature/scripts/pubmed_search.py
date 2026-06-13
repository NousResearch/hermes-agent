#!/usr/bin/env python3
"""
PubMed Literature CLI — search and analyze biomedical literature via NCBI E-utilities.

Usage:
    python3 pubmed_search.py search "CRISPR gene therapy" --num 5
    python3 pubmed_search.py detail 38271494
    python3 pubmed_search.py author "Yoshua Bengio" --num 10
    python3 pubmed_search.py journal "Nature" --since 2024 --num 5

No API key required. Uses NCBI E-utilities (free, public).
"""

import json
import sys
import datetime
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "Mozilla/5.0"


def api_request(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode()


def parse_esearch(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    ids = [id_elem.text for id_elem in root.findall(".//Id") or []]
    total = root.findtext(".//Count") or "0"
    return {"total": int(total), "ids": ids}


def parse_esummary(xml_text: str) -> list:
    root = ET.fromstring(xml_text)
    results = []
    for docsum in root.findall(".//DocSum") or []:
        item = {
            "uid": (docsum.findtext("Id") or ""),
            "title": "",
            "source": "",
            "pubdate": "",
            "authors": [],
        }
        for child in docsum:
            name = child.get("Name", "")
            if name == "Title":
                item["title"] = child.text or ""
            elif name == "Source":
                item["source"] = child.text or ""
            elif name == "PubDate":
                item["pubdate"] = child.text or ""
            elif name in (
                "PMCID",
                "CitedByCount",
                "PmcCitedByCount",
                "pubmed_citedby_count",
            ):
                try:
                    item["cited_by_count"] = int(child.text or 0)
                except (ValueError, TypeError):
                    item["cited_by_count"] = 0
            elif name == "AuthorList":
                for author in child.findall(".//Item") or []:
                    if author.text:
                        item["authors"].append(author.text)
        results.append(item)
    return results


def parse_abstract(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    article = root.find(".//Article")
    if article is None:
        return {"error": "No article found"}
    title = article.findtext("ArticleTitle") or ""
    abstract_parts = []
    for elem in article.findall(".//AbstractText") or []:
        text = elem.text or ""
        label = elem.get("Label", "")
        if label:
            text = f"{label}: {text}"
        if text.strip():
            abstract_parts.append(text.strip())
    abstract = "\n".join(abstract_parts)

    authors = []
    for author in article.findall(".//Author") or []:
        last = author.findtext("LastName") or ""
        fore = author.findtext("ForeName") or ""
        if last or fore:
            authors.append(f"{last} {fore}".strip())

    journal = article.findtext(".//Journal/Title") or ""
    pubdate_parts = []
    for tag in ["Year", "Month", "Day", "MedlineDate"]:
        v = article.findtext(f".//Journal/JournalIssue/PubDate/{tag}") or ""
        if v:
            pubdate_parts.append(v)

    mesh = []
    for mh in root.findall(".//MeshHeading") or []:
        dn = mh.findtext("DescriptorName") or ""
        if dn:
            mesh.append(dn)

    keywords = []
    for kw in root.findall(".//Keyword") or []:
        k = kw.text or ""
        if k:
            keywords.append(k)

    return {
        "title": title,
        "abstract": abstract[:2000] if abstract else "",
        "authors": authors,
        "journal": journal,
        "pubdate": " ".join(pubdate_parts),
        "mesh_terms": mesh[:20],
        "keywords": keywords[:10],
    }


def cmd_search(args):
    params = {
        "db": "pubmed",
        "term": args.query,
        "retmax": args.num,
        "retmode": "json",
        "sort": "relevance",
    }
    if args.since:
        params["mindate"] = str(args.since)
        params["maxdate"] = str(datetime.date.today().year)
        params["datetype"] = "pdat"
    if args.author:
        params["term"] += f" AND {args.author}[Author]"
    if args.journal:
        params["term"] += f" AND {args.journal}[Journal]"

    url = f"{ESEARCH}?{urllib.parse.urlencode(params)}"
    raw = api_request(url)

    if params["retmode"] == "json":
        data = json.loads(raw)
        ids = data.get("esearchresult", {}).get("idlist", [])
        total = int(data.get("esearchresult", {}).get("count", 0))
    else:
        data = parse_esearch(raw)
        ids = data["ids"]
        total = data["total"]

    output = {"query": args.query, "total": total, "results": []}

    if ids:
        # Fetch summaries
        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        url2 = f"{ESUMMARY}?{urllib.parse.urlencode(summary_params)}"
        raw2 = api_request(url2)
        summaries = parse_esummary(raw2)
        output["results"] = summaries[: args.num]

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_detail(args):
    params = {
        "db": "pubmed",
        "id": args.pmid,
        "retmode": "xml",
        "rettype": "abstract",
    }
    url = f"{EFETCH}?{urllib.parse.urlencode(params)}"
    raw = api_request(url)
    article = parse_abstract(raw)
    article["pmid"] = args.pmid
    print(json.dumps(article, indent=2, ensure_ascii=False))


def cmd_author(args):
    params = {
        "db": "pubmed",
        "term": f"{args.name}[Author]",
        "retmax": args.num,
        "retmode": "json",
        "sort": "relevance",
    }
    if args.since:
        params["mindate"] = str(args.since)
        params["maxdate"] = str(datetime.date.today().year)
        params["datetype"] = "pdat"

    url = f"{ESEARCH}?{urllib.parse.urlencode(params)}"
    raw = api_request(url)
    data = json.loads(raw)
    ids = data.get("esearchresult", {}).get("idlist", [])
    total = int(data.get("esearchresult", {}).get("count", 0))

    output = {"author": args.name, "total": total, "results": []}
    if ids:
        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        url2 = f"{ESUMMARY}?{urllib.parse.urlencode(summary_params)}"
        raw2 = api_request(url2)
        summaries = parse_esummary(raw2)
        output["results"] = summaries[: args.num]

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_journal(args):
    params = {
        "db": "pubmed",
        "term": f"{args.name}[Journal]",
        "retmax": args.num,
        "retmode": "json",
        "sort": "relevance",
    }
    if args.since:
        params["mindate"] = str(args.since)
        params["maxdate"] = str(datetime.date.today().year)
        params["datetype"] = "pdat"

    url = f"{ESEARCH}?{urllib.parse.urlencode(params)}"
    raw = api_request(url)
    data = json.loads(raw)
    ids = data.get("esearchresult", {}).get("idlist", [])
    total = int(data.get("esearchresult", {}).get("count", 0))

    output = {"journal": args.name, "total": total, "results": []}
    if ids:
        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        url2 = f"{ESUMMARY}?{urllib.parse.urlencode(summary_params)}"
        raw2 = api_request(url2)
        summaries = parse_esummary(raw2)
        output["results"] = summaries[: args.num]

    print(json.dumps(output, indent=2, ensure_ascii=False))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "search":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("query")
        p.add_argument("--num", type=int, default=10)
        p.add_argument("--since", type=int)
        p.add_argument("--author")
        p.add_argument("--journal")
        args = p.parse_args(sys.argv[2:])
        cmd_search(args)
    elif command == "detail":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("pmid")
        args = p.parse_args(sys.argv[2:])
        cmd_detail(args)
    elif command == "author":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("name")
        p.add_argument("--num", type=int, default=10)
        p.add_argument("--since", type=int)
        args = p.parse_args(sys.argv[2:])
        cmd_author(args)
    elif command == "journal":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("name")
        p.add_argument("--num", type=int, default=10)
        p.add_argument("--since", type=int)
        args = p.parse_args(sys.argv[2:])
        cmd_journal(args)
    elif command in ("--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
