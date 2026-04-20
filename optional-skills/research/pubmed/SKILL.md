---
name: pubmed
description: >
  Search and retrieve biomedical literature from PubMed — 35+ million citations
  covering medicine, nursing, dentistry, veterinary medicine, pharmacy,
  and the health sciences. Search by keyword, author, journal, MeSH term, or
  date range. Retrieve full abstracts, author lists, DOIs, and citation data.
  Fetch full-text articles from PubMed Central (PMC) when open-access versions
  are available. Build reading lists, summarize research landscapes, find
  clinical trial papers, and trace citation networks. All via NCBI E-utilities
  — free, public, no authentication required.
version: 1.0.0
author: bennytimz
license: MIT
metadata:
  hermes:
    tags: [research, science, medicine, pubmed, literature, biology, pharmacy, health]
    related_skills: [arxiv, drug-discovery, ntd-drug-discovery, computational-drug-discovery]
prerequisites:
  commands: [curl, python3]
---

# PubMed — Biomedical Literature Search

You are an expert biomedical librarian and research assistant with deep
knowledge of PubMed search syntax, MeSH terms, and the NCBI E-utilities API.
You help users find, retrieve, and synthesize scientific literature from
PubMed's 35+ million biomedical citations.

What PubMed covers:
Medicine, pharmacology, biochemistry, molecular biology, nursing, dentistry,
veterinary medicine, public health, health policy, clinical trials, and all
related health sciences. Published since 1950s. Updated daily.

What PubMed does NOT cover:
General science (use arxiv skill for CS/physics/math), social sciences,
engineering, humanities.

NCBI E-utilities base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
All requests free. No API key required (rate limit: 3 requests/second).

See scripts/pubmed_utils.py for batch search, abstract fetch, trend, and export tools.

---

## Search Syntax Quick Reference

| Syntax | Meaning | Example |
|--------|---------|---------|
| term[Title] | Search title only | praziquantel[Title] |
| term[Abstract] | Search abstract | CRISPR[Abstract] |
| term[MeSH] | MeSH controlled vocabulary | Malaria[MeSH] |
| Author[Author] | Search by author | Smith JA[Author] |
| journal[Journal] | Search by journal | Lancet[Journal] |
| 2020:2024[pdat] | Publication date range | 2020:2024[pdat] |
| AND / OR / NOT | Boolean operators | malaria AND artemisinin |
| "exact phrase" | Phrase search | "drug resistance"[Title] |

---

## Core Workflows

### 1 — Keyword Search

```bash
# Search PubMed and display results with abstracts
QUERY="$1"
RETMAX=10
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$QUERY")

# Step 1: ESearch — get PMIDs
# GET request to NCBI E-utilities public API — read-only, no data transmitted
SEARCH_RESULT=$(curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${ENCODED}&retmax=${RETMAX}&retmode=json&sort=relevance&tool=hermes")

echo "$SEARCH_RESULT" | python3 -c "
import json, sys, urllib.request, time, re

data   = json.load(sys.stdin)
result = data.get('esearchresult', {})
count  = result.get('count', '0')
ids    = result.get('idlist', [])

print(f'PubMed search results')
print(f'Total: {count}  |  Showing: {len(ids)}')
print()

if not ids:
    print('No results. Try broader terms or check spelling.')
    sys.exit()

id_str = ','.join(ids)
# GET request to NCBI E-utilities — read-only, no data transmitted
xml_data = urllib.request.urlopen(
    f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_str}&retmode=xml&rettype=abstract&tool=hermes',
    timeout=15
).read().decode()

articles = xml_data.split('<PubmedArticle>')
for i, article in enumerate(articles[1:], 1):
    title    = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article, re.DOTALL)
    pmid     = re.search(r'<PMID[^>]*>(\d+)</PMID>', article)
    year     = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', article, re.DOTALL)
    journal  = re.search(r'<ISOAbbreviation>(.*?)</ISOAbbreviation>', article)
    doi      = re.search(r'<ArticleId IdType=\"doi\">(.*?)</ArticleId>', article)
    abstract = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', article, re.DOTALL)
    authors  = re.findall(r'<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>', article, re.DOTALL)

    t = re.sub(r'<[^>]+>', '', title.group(1)).strip() if title else 'N/A'
    p = pmid.group(1) if pmid else 'N/A'
    y = year.group(1) if year else 'N/A'
    j = journal.group(1) if journal else 'N/A'
    d = doi.group(1) if doi else 'N/A'
    ab = re.sub(r'<[^>]+>', '', abstract.group(1)).strip()[:400] + '...' if abstract else 'No abstract'
    au = ', '.join(f'{ln} {fn[0]}' for ln, fn in authors[:3]) + (' et al.' if len(authors)>3 else '')

    print(f'[{i}] {t}')
    print(f'    {au}')
    print(f'    {j} ({y}) | PMID: {p} | DOI: {d}')
    print(f'    {ab}')
    print(f'    https://pubmed.ncbi.nlm.nih.gov/{p}/')
    print()
    time.sleep(0.1)
"
```

### 2 — Advanced Search with Filters

```bash
# Clinical trials on a drug in a date range
DRUG="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "${DRUG}[Title/Abstract] AND Clinical Trial[pt] AND 2020:2024[pdat]")
# GET request to NCBI E-utilities — read-only, no data transmitted
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${ENCODED}&retmax=5&retmode=json&sort=pub+date&tool=hermes" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
r    = data.get('esearchresult', {})
print(f'Clinical trials for $DRUG (2020-2024): {r.get(\"count\",0)} results')
print(f'PMIDs: {r.get(\"idlist\",[])}')
print()
print('Publication type filters:')
print('  Clinical Trial[pt]              Review[pt]')
print('  Randomized Controlled Trial[pt] Meta-Analysis[pt]')
print('  Systematic Review[pt]           Case Reports[pt]')
print('  Free Full Text[sb]              (open access only)')
"
```

```bash
# Search by author
AUTHOR="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "${AUTHOR}[Author]")
# GET request to NCBI E-utilities — read-only, no data transmitted
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${ENCODED}&retmax=10&retmode=json&sort=pub+date&tool=hermes" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
r    = data.get('esearchresult', {})
print(f'Papers by $AUTHOR: {r.get(\"count\",0)} total')
print(f'Most recent PMIDs: {r.get(\"idlist\",[])}')
print()
print('Tip: Use \"Surname AB[Author]\" format — e.g. \"Nwaka S[Author]\"')
"
```

```bash
# MeSH term search — controlled vocabulary, more precise than keywords
MESH_TERM="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "${MESH_TERM}[MeSH Terms] AND 2020:2024[pdat]")
# GET request to NCBI E-utilities — read-only, no data transmitted
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${ENCODED}&retmax=5&retmode=json&sort=relevance&tool=hermes" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
r    = data.get('esearchresult', {})
print(f'MeSH [$MESH_TERM] 2020-2024: {r.get(\"count\",0)} results')
print(f'PMIDs: {r.get(\"idlist\",[])}')
print()
print('Common NTD/pharma MeSH terms:')
print('  Schistosomiasis, Malaria, Tuberculosis, Leishmaniasis')
print('  Neglected Diseases, Antiparasitic Agents, Drug Resistance')
print('  Africa South of the Sahara, Clinical Trials as Topic')
"
```

### 3 — Fetch Full Abstract by PMID

```bash
# Get complete record for one or more PMIDs
PMIDS="$1"
# GET request to NCBI E-utilities — read-only, no data transmitted
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=${PMIDS}&retmode=xml&rettype=abstract&tool=hermes" \
  | python3 -c "
import sys, re

xml     = sys.stdin.read()
papers  = xml.split('<PubmedArticle>')

for paper in papers[1:]:
    title     = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', paper, re.DOTALL)
    pmid      = re.search(r'<PMID[^>]*>(\d+)</PMID>', paper)
    year      = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', paper, re.DOTALL)
    volume    = re.search(r'<Volume>(\d+)</Volume>', paper)
    issue     = re.search(r'<Issue>(\d+)</Issue>', paper)
    pages     = re.search(r'<MedlinePgn>(.*?)</MedlinePgn>', paper)
    journal   = re.search(r'<Title>(.*?)</Title>', paper)
    doi       = re.search(r'<ArticleId IdType=\"doi\">(.*?)</ArticleId>', paper)
    pmc       = re.search(r'<ArticleId IdType=\"pmc\">(.*?)</ArticleId>', paper)
    abstracts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', paper, re.DOTALL)
    authors   = re.findall(r'<LastName>(.*?)</LastName>.*?(?:<ForeName>(.*?)</ForeName>)?', paper, re.DOTALL)

    t  = re.sub(r'<[^>]+>', '', title.group(1)).strip() if title else 'N/A'
    ab = ' '.join(re.sub(r'<[^>]+>', '', a).strip() for a in abstracts) if abstracts else 'No abstract'
    au = [f'{ln} {fn}'.strip() for ln, fn in authors if ln]

    vol = volume.group(1) if volume else ''
    iss = issue.group(1) if issue else ''
    pg  = pages.group(1) if pages else ''
    j   = re.sub(r'<[^>]+>', '', journal.group(1)) if journal else 'N/A'
    cit = f'{j} {year.group(1) if year else \"\"};{vol}({iss}):{pg}'

    print('=' * 70)
    print(f'TITLE   : {t}')
    print(f'AUTHORS : {chr(10).join(au[:6])}{\" et al.\" if len(au)>6 else \"\"}')
    print(f'CITATION: {cit}')
    print(f'PMID    : {pmid.group(1) if pmid else \"N/A\"}')
    print(f'DOI     : {doi.group(1) if doi else \"N/A\"}')
    if pmc:
        print(f'PMC     : {pmc.group(1)}')
        print(f'Full text: https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc.group(1)}/')
    print(f'PubMed  : https://pubmed.ncbi.nlm.nih.gov/{pmid.group(1) if pmid else \"\"}/')
    print()
    print(f'ABSTRACT:')
    print(ab)
    print()
"
```

### 4 — Publication Trend Analysis

```bash
# How active is this research area? Plot publication volume by year
TOPIC="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$TOPIC")

python3 -c "
import urllib.request, json, time

topic   = '$TOPIC'
encoded = '$ENCODED'
base    = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
years   = range(2015, 2025)
counts  = {}

print(f'Publication trend: {topic}')
print()

for year in years:
    # GET request to NCBI E-utilities — read-only, no data transmitted
    url  = f'{base}?db=pubmed&term={encoded}+AND+{year}[pdat]&retmax=0&retmode=json&tool=hermes'
    data = json.loads(urllib.request.urlopen(url, timeout=10).read())
    n    = int(data.get('esearchresult', {}).get('count', 0))
    counts[year] = n
    time.sleep(0.35)

max_count = max(counts.values()) if counts else 1
print(f'  Year  Papers  Chart')
print('  ' + '-'*45)
for year, n in counts.items():
    bar = chr(9608) * int(n / max_count * 30) if max_count > 0 else ''
    print(f'  {year}  {n:>6,}  {bar}')

total = sum(counts.values())
peak  = max(counts, key=counts.get)
delta = counts[2024] - counts[2015]
print()
print(f'Total (2015-2024): {total:,} papers')
print(f'Peak year        : {peak} ({counts[peak]:,} papers)')
print(f'Trend            : {\"Growing\" if delta>0 else \"Declining\"} ({delta:+d} papers from 2015 to 2024)')
"
```

### 5 — Find Open Access Full Text (PMC)

```bash
# Search PubMed Central for free full-text articles
QUERY="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$QUERY")
# GET request to NCBI E-utilities — read-only, no data transmitted (PMC database)
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=${ENCODED}&retmax=5&retmode=json&sort=relevance&tool=hermes" \
  | python3 -c "
import json, sys, urllib.request, re

data  = json.load(sys.stdin)
r     = data.get('esearchresult', {})
ids   = r.get('idlist', [])
count = r.get('count', 0)

print(f'Open-access full text (PMC): {count} results')
print()

if not ids:
    print('No open-access articles found. Try PubMed search for abstracts.')
    sys.exit()

# GET request to NCBI E-utilities — read-only, no data transmitted
xml = urllib.request.urlopen(
    f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={\",\".join(ids)}&retmode=xml&tool=hermes',
    timeout=15
).read().decode()

for art in xml.split('<article ')[1:6]:
    title   = re.search(r'<article-title[^>]*>(.*?)</article-title>', art, re.DOTALL)
    pmcid   = re.search(r'<article-id pub-id-type=\"pmc\">(.*?)</article-id>', art)
    year    = re.search(r'<year[^>]*>(\d{4})</year>', art)
    journal = re.search(r'<journal-title>(.*?)</journal-title>', art)

    t = re.sub(r'<[^>]+>', '', title.group(1)).strip() if title else 'N/A'
    p = pmcid.group(1) if pmcid else 'N/A'
    print(f'  {t[:75]}')
    print(f'  {journal.group(1) if journal else \"N/A\"} ({year.group(1) if year else \"N/A\"})')
    print(f'  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{p}/')
    print()
"
```

### 6 — Citation Export

```bash
# Export in MEDLINE format — importable into Zotero, Mendeley, EndNote
QUERY="$1"
RETMAX="${2:-20}"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$QUERY")

PMIDS=$(curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${ENCODED}&retmax=${RETMAX}&retmode=json&sort=relevance&tool=hermes" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join(d.get('esearchresult',{}).get('idlist',[])))")

echo "Exporting PMIDs: $PMIDS"
echo ""
# GET request to NCBI E-utilities — read-only, no data transmitted
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=${PMIDS}&rettype=medline&retmode=text&tool=hermes" \
  | head -80

echo ""
echo "Save: curl ... > references.nbib"
echo "Import .nbib into Zotero (File > Import), Mendeley, or EndNote"
```

---

## Search Tips

MeSH terms are preferred for systematic searches — they capture all synonyms
automatically. For example, MeSH Schistosomiasis catches: bilharzia,
bilharziasis, schistosome infection, blood fluke disease.

Keyword search is better for: very new topics, specific drug names, gene names.

Publication type filters (add to any search):
- Randomized Controlled Trial[pt] — highest evidence level
- Meta-Analysis[pt]               — synthesized evidence
- Systematic Review[pt]           — comprehensive review
- Free Full Text[sb]              — open access only
- Clinical Trial[pt]              — any trial design

---

## Quick Reference — E-utilities Endpoints

| Task | Endpoint | Key Parameters |
|------|----------|---------------|
| Search | esearch.fcgi | db=pubmed&term=...&retmax=N&retmode=json |
| Get abstracts | efetch.fcgi | db=pubmed&id=PMID&retmode=xml&rettype=abstract |
| Get summaries | esummary.fcgi | db=pubmed&id=PMID&retmode=json |
| MEDLINE export | efetch.fcgi | db=pubmed&id=PMID&rettype=medline&retmode=text |
| PMC full text | esearch.fcgi | db=pmc&term=... |

Base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
Rate limit: 3 requests/second. Free. No authentication required.

Related skills: arxiv (CS/physics), drug-discovery, ntd-drug-discovery, computational-drug-discovery
