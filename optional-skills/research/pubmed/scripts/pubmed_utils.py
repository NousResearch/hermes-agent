#!/usr/bin/env python3
"""
pubmed_utils.py — Utility scripts for PubMed literature search workflows.

Outbound network calls (read-only GET requests only):
  - https://eutils.ncbi.nlm.nih.gov  (NCBI E-utilities — PubMed/PMC public API)

No data is transmitted outbound. All requests fetch public read-only data.
No API key required. Rate limit: 3 requests/second.

Commands:
    python3 pubmed_utils.py search "malaria drug resistance" --max 10
    python3 pubmed_utils.py abstract 38765432
    python3 pubmed_utils.py abstract 38765432,38123456
    python3 pubmed_utils.py trend "praziquantel schistosomiasis"
    python3 pubmed_utils.py author "Nwaka S"
    python3 pubmed_utils.py related 38765432
    python3 pubmed_utils.py export "tuberculosis bedaquiline" --max 20

Author: Bennytimz
"""

import sys
import re
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_PARAMS = "tool=hermes&email=hermes-pubmed@nousresearch.com"
RATE_LIMIT  = 0.35


def http_get(url: str, timeout: int = 15) -> Optional[bytes]:
    """GET request to NCBI E-utilities public API — read-only, no data transmitted.
    Target: https://eutils.ncbi.nlm.nih.gov
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json, text/xml, text/plain"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("Rate limited by NCBI. Waiting 2 seconds...", file=sys.stderr)
            time.sleep(2)
            return http_get(url, timeout)
        print(f"HTTP {e.code}: {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def esearch(query: str, db: str = "pubmed", retmax: int = 10,
            sort: str = "relevance") -> dict:
    """Search NCBI database and return result dict.
    GET request to https://eutils.ncbi.nlm.nih.gov — read-only, no data transmitted.
    """
    encoded = urllib.parse.quote(query)
    url     = (f"{EUTILS_BASE}/esearch.fcgi?db={db}&term={encoded}"
               f"&retmax={retmax}&retmode=json&sort={sort}&{TOOL_PARAMS}")
    raw = http_get(url)
    if not raw:
        return {}
    try:
        return json.loads(raw).get("esearchresult", {})
    except Exception:
        return {}


def efetch_xml(pmids: list, db: str = "pubmed") -> str:
    """Fetch full records in XML format for a list of IDs.
    GET request to https://eutils.ncbi.nlm.nih.gov — read-only, no data transmitted.
    """
    id_str = ",".join(pmids)
    url    = (f"{EUTILS_BASE}/efetch.fcgi?db={db}&id={id_str}"
              f"&retmode=xml&rettype=abstract&{TOOL_PARAMS}")
    raw = http_get(url)
    return raw.decode("utf-8", errors="replace") if raw else ""


def parse_articles(xml: str) -> list:
    """Parse PubMed XML into list of article dicts."""
    articles = []
    for block in xml.split("<PubmedArticle>")[1:]:

        def extract(pattern, text=block, flags=re.DOTALL):
            m = re.search(pattern, text, flags)
            return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else "N/A"

        author_matches = re.findall(
            r"<Author[^>]*>.*?<LastName>(.*?)</LastName>.*?(?:<ForeName>(.*?)</ForeName>)?",
            block, re.DOTALL
        )
        authors = [f"{ln} {fn}".strip() for ln, fn in author_matches if ln]

        abstract_parts = re.findall(
            r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.DOTALL
        )
        abstract = " ".join(
            re.sub(r"<[^>]+>", "", p).strip() for p in abstract_parts
        ) if abstract_parts else "No abstract available"

        mesh = re.findall(r"<DescriptorName[^>]*>(.*?)</DescriptorName>", block)
        mesh_clean = [re.sub(r"<[^>]+>", "", m).strip() for m in mesh]

        articles.append({
            "pmid":     extract(r"<PMID[^>]*>(\d+)</PMID>"),
            "title":    extract(r"<ArticleTitle>(.*?)</ArticleTitle>"),
            "journal":  extract(r"<ISOAbbreviation>(.*?)</ISOAbbreviation>"),
            "year":     extract(r"<PubDate>.*?<Year>(\d{4})</Year>"),
            "volume":   extract(r"<Volume>(\d+)</Volume>"),
            "issue":    extract(r"<Issue>(\d+)</Issue>"),
            "pages":    extract(r"<MedlinePgn>(.*?)</MedlinePgn>"),
            "doi":      extract(r'<ArticleId IdType="doi">(.*?)</ArticleId>'),
            "pmc":      extract(r'<ArticleId IdType="pmc">(.*?)</ArticleId>'),
            "authors":  authors,
            "abstract": abstract,
            "mesh":     mesh_clean[:8],
        })
    return articles


def print_article(article: dict, show_abstract: bool = True,
                  index: int = None) -> None:
    """Pretty-print a single article."""
    prefix = f"[{index}] " if index is not None else ""
    authors = article["authors"]
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" et al. ({len(authors)} authors)"

    vol = article["volume"]
    iss = article["issue"]
    pg  = article["pages"]
    cit = f"{article['journal']} {article['year']}"
    if vol != "N/A":
        cit += f";{vol}"
    if iss != "N/A":
        cit += f"({iss})"
    if pg != "N/A":
        cit += f":{pg}"

    print(f"{prefix}{article['title']}")
    print(f"  {author_str}")
    print(f"  {cit}")
    print(f"  PMID: {article['pmid']}  |  DOI: {article['doi']}")
    if article["pmc"] != "N/A":
        print(f"  Full text: https://www.ncbi.nlm.nih.gov/pmc/articles/{article['pmc']}/")
    print(f"  PubMed: https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/")

    if show_abstract and article["abstract"] != "No abstract available":
        short = article["abstract"][:600]
        if len(article["abstract"]) > 600:
            short += "..."
        print(f"\n  Abstract: {short}")

    print()


def cmd_search(query: str, retmax: int = 10, sort: str = "relevance",
               show_abstract: bool = True) -> None:
    """Search PubMed and display results."""
    print(f"\nPubMed search: {query!r}  |  Max: {retmax}\n")

    result = esearch(query, retmax=retmax, sort=sort)
    if not result:
        print("Search failed.")
        return

    count = result.get("count", "0")
    ids   = result.get("idlist", [])

    print(f"Total: {count}  |  Showing: {len(ids)}\n")

    if not ids:
        print("No results. Try broader terms or check spelling.")
        spell_url = (f"{EUTILS_BASE}/espell.fcgi?db=pubmed"
                     f"&term={urllib.parse.quote(query)}&{TOOL_PARAMS}")
        spell_raw = http_get(spell_url)
        if spell_raw:
            corrected = re.search(r"<CorrectedQuery>(.*?)</CorrectedQuery>",
                                  spell_raw.decode())
            if corrected:
                print(f"Did you mean: {corrected.group(1)}")
        return

    time.sleep(RATE_LIMIT)
    xml      = efetch_xml(ids)
    articles = parse_articles(xml)

    for i, art in enumerate(articles, 1):
        print_article(art, show_abstract=show_abstract, index=i)
        time.sleep(0.1)

    if int(count) > retmax:
        print(f"Showing {retmax} of {count}. Use --max {min(retmax*2,100)} to see more.")


def cmd_abstract(pmids_str: str) -> None:
    """Fetch full abstracts for given PMIDs."""
    pmids    = [p.strip() for p in pmids_str.split(",") if p.strip()]
    xml      = efetch_xml(pmids)
    articles = parse_articles(xml)

    if not articles:
        print(f"No records found for PMIDs: {pmids_str}")
        return

    print(f"\nFetched {len(articles)} article(s)\n")
    print("=" * 70)

    for art in articles:
        authors    = art["authors"]
        author_str = "\n          ".join(authors[:8])
        if len(authors) > 8:
            author_str += f"\n          ... and {len(authors)-8} more"

        vol = art["volume"]
        iss = art["issue"]
        pg  = art["pages"]
        cit = f"{art['journal']} {art['year']}"
        if vol != "N/A":
            cit += f";{vol}"
        if iss != "N/A":
            cit += f"({iss})"
        if pg != "N/A":
            cit += f":{pg}"

        print(f"TITLE   : {art['title']}")
        print(f"AUTHORS : {author_str}")
        print(f"CITATION: {cit}")
        print(f"PMID    : {art['pmid']}")
        print(f"DOI     : {art['doi']}")
        if art["pmc"] != "N/A":
            print(f"PMC     : {art['pmc']}")
            print(f"Full text: https://www.ncbi.nlm.nih.gov/pmc/articles/{art['pmc']}/")
        print(f"PubMed  : https://pubmed.ncbi.nlm.nih.gov/{art['pmid']}/")
        print()
        print("ABSTRACT:")
        print(art["abstract"])
        if art["mesh"]:
            print()
            print(f"MeSH    : {', '.join(art['mesh'])}")
        print()
        print("=" * 70)
        print()


def cmd_trend(query: str, start_year: int = 2015, end_year: int = 2024) -> None:
    """Show publication trend over years."""
    print(f"\nPublication trend: {query!r}  ({start_year}-{end_year})\n")

    counts = {}
    for year in range(start_year, end_year + 1):
        result = esearch(f"{query} AND {year}[pdat]", retmax=0)
        counts[year] = int(result.get("count", 0))
        time.sleep(RATE_LIMIT)

    max_count = max(counts.values()) if counts else 1

    print(f"  {'Year':<6}  {'Papers':>7}  Chart")
    print("  " + "-"*50)
    for year, n in counts.items():
        bar = chr(9608) * int(n / max_count * 35) if max_count > 0 else ""
        print(f"  {year:<6}  {n:>7,}  {bar}")

    total     = sum(counts.values())
    peak_year = max(counts, key=counts.get)
    delta     = counts[end_year] - counts[start_year]

    print()
    print(f"Total ({start_year}-{end_year}): {total:,} papers")
    print(f"Peak year             : {peak_year} ({counts[peak_year]:,} papers)")
    print(f"Trend                 : {'Growing' if delta>0 else 'Declining'} ({delta:+d} papers)")


def cmd_author(author_name: str, retmax: int = 10) -> None:
    """Find papers by a specific author."""
    result = esearch(f"{author_name}[Author]", retmax=retmax, sort="pub date")

    print(f"\nPapers by: {author_name}")
    print(f"Total: {result.get('count', 0)}\n")

    ids = result.get("idlist", [])
    if not ids:
        print("No papers found. Try format: 'Surname AB[Author]'")
        return

    time.sleep(RATE_LIMIT)
    xml      = efetch_xml(ids)
    articles = parse_articles(xml)

    for i, art in enumerate(articles, 1):
        au = ", ".join(art["authors"][:3]) + (" et al." if len(art["authors"])>3 else "")
        print(f"  [{i}] {art['title'][:78]}")
        print(f"       {au}")
        print(f"       {art['journal']} {art['year']} | PMID: {art['pmid']}")
        print()


def cmd_related(pmid: str, retmax: int = 5) -> None:
    """Find papers related to a given PMID."""
    print(f"\nPapers related to PMID: {pmid}\n")

    # GET request to NCBI E-utilities — read-only, no data transmitted
    url = (f"{EUTILS_BASE}/elink.fcgi?dbfrom=pubmed&db=pubmed"
           f"&id={pmid}&cmd=neighbor_score&retmode=json&{TOOL_PARAMS}")
    raw = http_get(url)
    if not raw:
        print("Could not fetch related articles.")
        return

    try:
        data     = json.loads(raw)
        related_ids = []
        for ls in data.get("linksets", []):
            for link in ls.get("linksetdbs", []):
                if link.get("linkname") == "pubmed_pubmed":
                    related_ids = [str(i["id"]) for i in link.get("links", [])[:retmax]]
                    break
    except Exception as e:
        print(f"Parse error: {e}")
        return

    if not related_ids:
        print("No related articles found.")
        return

    time.sleep(RATE_LIMIT)
    xml      = efetch_xml(related_ids)
    articles = parse_articles(xml)

    print(f"Top {len(articles)} related articles:\n")
    for i, art in enumerate(articles, 1):
        print_article(art, show_abstract=False, index=i)


def cmd_export(query: str, retmax: int = 20,
               fmt: str = "medline", output: str = None) -> None:
    """Export search results in MEDLINE format."""
    print(f"\nExporting: {query!r}  |  Format: {fmt}  |  Max: {retmax}\n")

    result = esearch(query, retmax=retmax)
    ids    = result.get("idlist", [])
    count  = result.get("count", "0")

    if not ids:
        print("No results found.")
        return

    print(f"Found {count}, exporting {len(ids)}...")
    time.sleep(RATE_LIMIT)

    rettype = "medline" if fmt == "medline" else "abstract"
    # GET request to NCBI E-utilities — read-only, no data transmitted
    url = (f"{EUTILS_BASE}/efetch.fcgi?db=pubmed&id={','.join(ids)}"
           f"&rettype={rettype}&retmode=text&{TOOL_PARAMS}")
    raw = http_get(url)

    if not raw:
        print("Export failed.")
        return

    content = raw.decode("utf-8", errors="replace")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved {len(ids)} records to: {output}")
        print("Import into Zotero (File > Import), Mendeley, or EndNote")
    else:
        print(content[:2000])
        if len(content) > 2000:
            print(f"\n... truncated. Save: python3 pubmed_utils.py export \"{query}\" --output refs.nbib")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PubMed literature search utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  search   <query>          Search PubMed with abstracts
  abstract <pmid[,pmid...]> Fetch full abstracts by PMID
  trend    <query>          Publication volume trend by year
  author   <name>           All papers by an author
  related  <pmid>           Papers related to a PMID
  export   <query>          Export citations in MEDLINE format

Examples:
  python3 pubmed_utils.py search "praziquantel schistosomiasis" --max 5
  python3 pubmed_utils.py search "malaria artemisinin" --sort pub_date
  python3 pubmed_utils.py abstract 38765432
  python3 pubmed_utils.py abstract 38765432,38123456
  python3 pubmed_utils.py trend "bedaquiline tuberculosis" --start 2012 --end 2024
  python3 pubmed_utils.py author "Nwaka S"
  python3 pubmed_utils.py related 38765432
  python3 pubmed_utils.py export "HIV antiretroviral Africa" --max 20 --output refs.nbib

Search syntax:
  Field tags : term[Title], term[Abstract], Author[Author], term[MeSH]
  Date range : 2020:2024[pdat]
  Pub types  : Randomized Controlled Trial[pt], Review[pt], Meta-Analysis[pt]
  Open access: Free Full Text[sb]
  Boolean    : AND, OR, NOT
        """
    )

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--max", type=int, default=10, dest="retmax")
    p.add_argument("--sort", default="relevance",
                   choices=["relevance", "pub_date", "author", "journal"])
    p.add_argument("--no-abstract", action="store_true")

    p = sub.add_parser("abstract")
    p.add_argument("pmids")

    p = sub.add_parser("trend")
    p.add_argument("query")
    p.add_argument("--start", type=int, default=2015, dest="start_year")
    p.add_argument("--end", type=int, default=2024, dest="end_year")

    p = sub.add_parser("author")
    p.add_argument("name")
    p.add_argument("--max", type=int, default=10, dest="retmax")

    p = sub.add_parser("related")
    p.add_argument("pmid")
    p.add_argument("--max", type=int, default=5, dest="retmax")

    p = sub.add_parser("export")
    p.add_argument("query")
    p.add_argument("--max", type=int, default=20, dest="retmax")
    p.add_argument("--format", default="medline",
                   choices=["medline", "abstract"], dest="fmt")
    p.add_argument("--output", default=None)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "search":
        cmd_search(args.query, retmax=args.retmax, sort=args.sort,
                   show_abstract=not args.no_abstract)
    elif args.command == "abstract":
        cmd_abstract(args.pmids)
    elif args.command == "trend":
        cmd_trend(args.query, start_year=args.start_year, end_year=args.end_year)
    elif args.command == "author":
        cmd_author(args.name, retmax=args.retmax)
    elif args.command == "related":
        cmd_related(args.pmid, retmax=args.retmax)
    elif args.command == "export":
        cmd_export(args.query, retmax=args.retmax, fmt=args.fmt, output=args.output)


if __name__ == "__main__":
    main()
