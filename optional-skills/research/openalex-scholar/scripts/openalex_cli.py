#!/usr/bin/env python3
"""
OpenAlex Scholar CLI — search and analyze academic research via the free OpenAlex API.

Usage:
    python3 openalex_cli.py search "quantum computing"
    python3 openalex_cli.py search "transformer neural network" --since 2020 --min-citations 100 --open-access
    python3 openalex_cli.py author "Geoffrey Hinton"
    python3 openalex_cli.py work W4255852847
    python3 openalex_cli.py doi 10.1038/nature12373
    python3 openalex_cli.py concept "machine learning"
    python3 openalex_cli.py institution "Stanford University"
    python3 openalex_cli.py source "Nature"
    python3 openalex_cli.py citations W4255852847

No API key required. Uses OpenAlex REST API (docs.openalex.org).
"""

import json
import os
import sys
import urllib.parse
import urllib.request

API_BASE = "https://api.openalex.org"
USER_AGENT = "OpenAlexScholarSkill/1.0 (mailto:research@example.com)"


def api_get(path: str, params: dict = None) -> dict:
    """Make a GET request to the OpenAlex API."""
    url = f"{API_BASE}{path}"
    if params:
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def print_json(data):
    """Pretty-print JSON output."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_search(args):
    """Search scholarly works."""
    params = {
        "search": args.query,
        "per_page": str(args.per_page),
        "sort": args.sort,
        "mailto": "openalex@hermes-agent.dev",
    }

    filters = []
    if args.since:
        filters.append(f"from_publication_date:{args.since}-01-01")
    if args.until:
        filters.append(f"to_publication_date:{args.until}-12-31")
    if args.open_access:
        filters.append("is_oa:true")
    if args.min_citations:
        filters.append(f"cited_by_count:>{args.min_citations}")
    if args.type:
        filters.append(f"type:{args.type}")
    if args.language:
        filters.append(f"language:{args.language}")
    if args.author_id:
        filters.append(f"authorships.author.id:{args.author_id}")
    if args.concept_id:
        filters.append(f"concepts.id:{args.concept_id}")
    if args.institution_id:
        filters.append(f"institutions.id:{args.institution_id}")
    if filters:
        params["filter"] = ",".join(filters)

    data = api_get("/works", params)
    meta = data.get("meta", {})
    results = data.get("results", [])

    output = {
        "total_results": meta.get("count", 0),
        "page": meta.get("page", 1),
        "per_page": meta.get("per_page", args.per_page),
        "results": [],
    }
    for r in results:
        authors = [a.get("author", {}).get("display_name", "?")
                   for a in r.get("authorships", [])[:5]]
        output["results"].append({
            "id": r.get("id", "").replace("https://openalex.org/", ""),
            "title": r.get("title", ""),
            "year": r.get("publication_year"),
            "cited_by": r.get("cited_by_count", 0),
            "type": r.get("type"),
            "oa": r.get("open_access", {}).get("is_oa", False),
            "authors": authors,
            "doi": r.get("doi"),
        })

    print_json(output)


def cmd_work(args):
    """Get a specific work by OpenAlex ID."""
    data = api_get(f"/works/{args.work_id}")
    authors = [a.get("author", {}).get("display_name", "?")
               for a in data.get("authorships", [])]
    concepts = [c.get("display_name") for c in data.get("concepts", [])[:10]]

    output = {
        "id": data.get("id", "").replace("https://openalex.org/", ""),
        "title": data.get("title", ""),
        "year": data.get("publication_year"),
        "cited_by_count": data.get("cited_by_count", 0),
        "type": data.get("type"),
        "doi": data.get("doi"),
        "language": data.get("language"),
        "oa": data.get("open_access", {}).get("is_oa", False),
        "oa_status": data.get("open_access", {}).get("oa_status"),
        "authors": authors,
        "concepts": concepts,
        "referenced_works_count": len(data.get("referenced_works", [])),
        "is_retracted": data.get("is_retracted", False),
        "abstract_inverted_index": data.get("abstract_inverted_index") is not None,
    }
    print_json(output)


def cmd_doi(args):
    """Look up a work by DOI."""
    doi_clean = args.doi.replace("https://doi.org/", "").replace("doi:", "")
    data = api_get(f"/works/doi:{doi_clean}")
    authors = [a.get("author", {}).get("display_name", "?")
               for a in data.get("authorships", [])]

    output = {
        "id": data.get("id", "").replace("https://openalex.org/", ""),
        "title": data.get("title", ""),
        "doi": f"https://doi.org/{doi_clean}",
        "year": data.get("publication_year"),
        "cited_by_count": data.get("cited_by_count", 0),
        "type": data.get("type"),
        "authors": authors,
    }
    print_json(output)


def cmd_author(args):
    """Get author profile and top works."""
    # Search for the author first
    search_data = api_get("/authors", {
        "search": args.name,
        "per_page": "5",
        "mailto": "openalex@hermes-agent.dev",
    })
    results = search_data.get("results", [])
    if not results:
        print(json.dumps({"error": f"No author found: {args.name}"}))
        return

    author = results[0]
    author_id = author.get("id", "").rsplit("/", 1)[-1]
    insts = [i.get("display_name", "?")
             for i in author.get("last_known_institutions", [])]

    output = {
        "id": author_id,
        "display_name": author.get("display_name"),
        "cited_by_count": author.get("cited_by_count", 0),
        "works_count": author.get("works_count", 0),
        "h_index": author.get("summary_stats", {}).get("h_index"),
        "i10_index": author.get("summary_stats", {}).get("i10_index"),
        "2yr_mean_citedness": author.get("summary_stats", {}).get("2yr_mean_citedness"),
        "institutions": insts,
        "concepts": [c.get("display_name") for c in author.get("concepts", [])[:5]],
    }

    # Get top-cited works
    works_data = api_get(f"/works", {
        "filter": f"authorships.author.id:{author_id}",
        "sort": "cited_by_count:desc",
        "per_page": "5",
        "mailto": "openalex@hermes-agent.dev",
    })
    output["top_works"] = [
        {"title": w.get("title"), "cited_by": w.get("cited_by_count", 0), "year": w.get("publication_year")}
        for w in works_data.get("results", [])[:5]
    ]

    print_json(output)


def cmd_concept(args):
    """Get concept details and hierarchy."""
    search_data = api_get("/concepts", {
        "search": args.name,
        "per_page": "1",
        "mailto": "openalex@hermes-agent.dev",
    })
    results = search_data.get("results", [])
    if not results:
        print(json.dumps({"error": f"No concept found: {args.name}"}))
        return

    concept = results[0]
    output = {
        "id": concept.get("id", "").rsplit("/", 1)[-1],
        "display_name": concept.get("display_name"),
        "description": concept.get("description"),
        "level": concept.get("level"),
        "works_count": concept.get("works_count"),
        "cited_by_count": concept.get("cited_by_count", 0),
        "ancestors": [a.get("display_name") for a in concept.get("ancestors", [])],
        "descendants": [d.get("display_name") for d in concept.get("descendants", [])[:10]],
    }
    print_json(output)


def cmd_institution(args):
    """Get institution profile."""
    search_data = api_get("/institutions", {
        "search": args.name,
        "per_page": "1",
        "mailto": "openalex@hermes-agent.dev",
    })
    results = search_data.get("results", [])
    if not results:
        print(json.dumps({"error": f"No institution found: {args.name}"}))
        return

    inst = results[0]
    output = {
        "id": inst.get("id", "").rsplit("/", 1)[-1],
        "display_name": inst.get("display_name"),
        "country_code": inst.get("country_code"),
        "type": inst.get("type"),
        "homepage_url": inst.get("homepage_url"),
        "works_count": inst.get("works_count"),
        "cited_by_count": inst.get("cited_by_count", 0),
        "2yr_mean_citedness": inst.get("summary_stats", {}).get("2yr_mean_citedness"),
        "h_index": inst.get("summary_stats", {}).get("h_index"),
        "concepts": [c.get("display_name") for c in inst.get("x_concepts", [])[:5]],
    }
    print_json(output)


def cmd_source(args):
    """Get journal/conference/source info."""
    search_data = api_get("/sources", {
        "search": args.name,
        "per_page": "1",
        "mailto": "openalex@hermes-agent.dev",
    })
    results = search_data.get("results", [])
    if not results:
        print(json.dumps({"error": f"No source found: {args.name}"}))
        return

    src = results[0]
    output = {
        "id": src.get("id", "").rsplit("/", 1)[-1],
        "display_name": src.get("display_name"),
        "type": src.get("type"),
        "issn": src.get("issn_l"),
        "publisher": src.get("host_organization_name"),
        "works_count": src.get("works_count"),
        "cited_by_count": src.get("cited_by_count", 0),
        "is_oa": src.get("is_oa", False),
        "homepage_url": src.get("homepage_url"),
    }
    print_json(output)


def cmd_citations(args):
    """Citation analysis for a work."""
    data = api_get(f"/works/{args.work_id}")

    # Get works that cite this work
    citing_data = api_get("/works", {
        "filter": f"cited_by:{args.work_id}",
        "sort": "cited_by_count:desc",
        "per_page": "10",
        "mailto": "openalex@hermes-agent.dev",
    })

    output = {
        "source_work": {
            "id": data.get("id", "").replace("https://openalex.org/", ""),
            "title": data.get("title", ""),
            "cited_by_count": data.get("cited_by_count", 0),
            "referenced_works_count": len(data.get("referenced_works", [])),
        },
        "top_citing_works": [
            {"title": w.get("title"), "cited_by": w.get("cited_by_count", 0), "year": w.get("publication_year")}
            for w in citing_data.get("results", [])[:10]
        ],
        "total_citing_results": citing_data.get("meta", {}).get("count", 0),
    }
    print_json(output)


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "search":
        import argparse
        p = argparse.ArgumentParser(description="Search scholarly works")
        p.add_argument("query", help="Search query")
        p.add_argument("--per-page", type=int, default=10, help="Results per page (max 200)")
        p.add_argument("--sort", default="relevance_score:desc", help="Sort field:order")
        p.add_argument("--since", type=int, help="Publication year start")
        p.add_argument("--until", type=int, help="Publication year end")
        p.add_argument("--open-access", action="store_true", help="Open access only")
        p.add_argument("--min-citations", type=int, help="Minimum citation count")
        p.add_argument("--type", help="Work type (article, review, book-chapter, etc.)")
        p.add_argument("--language", help="Language code (en, fr, de, es, etc.)")
        p.add_argument("--author-id", help="Filter by OpenAlex author ID")
        p.add_argument("--concept-id", help="Filter by OpenAlex concept ID")
        p.add_argument("--institution-id", help="Filter by OpenAlex institution ID")
        args = p.parse_args(sys.argv[2:])
        cmd_search(args)

    elif command == "work":
        import argparse
        p = argparse.ArgumentParser(description="Get work details by OpenAlex ID")
        p.add_argument("work_id", help="OpenAlex work ID (e.g. W4255852847)")
        args = p.parse_args(sys.argv[2:])
        cmd_work(args)

    elif command == "doi":
        import argparse
        p = argparse.ArgumentParser(description="Look up work by DOI")
        p.add_argument("doi", help="DOI (e.g. 10.1038/nature12373)")
        args = p.parse_args(sys.argv[2:])
        cmd_doi(args)

    elif command == "author":
        import argparse
        p = argparse.ArgumentParser(description="Get author profile")
        p.add_argument("name", help="Author name to search")
        args = p.parse_args(sys.argv[2:])
        cmd_author(args)

    elif command == "concept":
        import argparse
        p = argparse.ArgumentParser(description="Explore concept hierarchy")
        p.add_argument("name", help="Concept name to search")
        args = p.parse_args(sys.argv[2:])
        cmd_concept(args)

    elif command == "institution":
        import argparse
        p = argparse.ArgumentParser(description="Get institution profile")
        p.add_argument("name", help="Institution name")
        args = p.parse_args(sys.argv[2:])
        cmd_institution(args)

    elif command == "source":
        import argparse
        p = argparse.ArgumentParser(description="Get journal/conference info")
        p.add_argument("name", help="Source name (journal, repository, conference)")
        args = p.parse_args(sys.argv[2:])
        cmd_source(args)

    elif command == "citations":
        import argparse
        p = argparse.ArgumentParser(description="Citation analysis")
        p.add_argument("work_id", help="OpenAlex work ID")
        args = p.parse_args(sys.argv[2:])
        cmd_citations(args)

    elif command in ("--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()