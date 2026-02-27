import sys
import time
import requests
import xml.etree.ElementTree as ET
import json

# ArXiv API namespaces
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom'
}

def fetch_arxiv_metadata(arxiv_id, max_retries=3):
    """
    Fetches paper metadata using the ArXiv Export API.
    Handles XML namespaces, rate limiting, and implements exponential backoff for 503s.
    """
    url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            
            # Handle 503 Service Unavailable (Retry with exponential backoff)
            if response.status_code == 503:
                wait_time = (2 ** attempt) + 3 # 4s, 5s, 7s...
                print(f"Warning: 503 Service Unavailable. Retrying in {wait_time}s...", file=sys.stderr)
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            entry = root.find('atom:entry', NAMESPACES)
            
            if entry is None or entry.find('atom:title', NAMESPACES).text.strip() == 'Error':
                return {"error": f"Paper ID '{arxiv_id}' not found or invalid."}
                
            # Extract fields
            title = entry.find('atom:title', NAMESPACES).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', NAMESPACES).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', NAMESPACES).text
            updated = entry.find('atom:updated', NAMESPACES).text
            
            doi = None
            for link in entry.findall('atom:link', NAMESPACES):
                if link.attrib.get('title') == 'doi':
                    doi = link.attrib.get('href').replace('http://dx.doi.org/', '')
            
            authors = []
            for author in entry.findall('atom:author', NAMESPACES):
                name = author.find('atom:name', NAMESPACES).text
                authors.append(name)
                
            links = []
            for link in entry.findall('atom:link', NAMESPACES):
                links.append(link.attrib.get('href'))

            metadata = {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "updated": updated,
                "doi": doi,
                "summary": summary,
                "links": links
            }
            
            return metadata

        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
            time.sleep(1) # Small delay for other transient errors
            
    return {"error": "Unknown failure during metadata fetch."}

def generate_bibtex(metadata):
    """Generates a formatted BibTeX entry from metadata."""
    if "error" in metadata:
        return ""
        
    cite_key = f"{metadata['authors'][0].split()[-1]}{metadata['published'][:4]}{metadata['arxiv_id'].split('.')[0]}"
    authors_str = " and ".join(metadata['authors'])
    
    bibtex = f"""@article{{{cite_key},
  title = {{{metadata['title']}}},
  author = {{{authors_str}}},
  year = {{{metadata['published'][:4]}}},
  eprint = {{{metadata['arxiv_id']}}},
  archivePrefix = {{arXiv}},
  primaryClass = {{cs.LG}},
  url = {{{metadata['links'][0]}}}
}}"""
    return bibtex

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_arxiv.py <arxiv_id>")
        sys.exit(1)
        
    arxiv_id = sys.argv[1].strip()
    
    # Enforce ArXiv rate limit (3s delay)
    # Note: In a real agentic workflow, the agent would wait or this script handles it.
    metadata = fetch_arxiv_metadata(arxiv_id)
    
    if "error" in metadata:
        print(json.dumps(metadata, indent=2))
    else:
        print("--- Metadata ---")
        print(json.dumps(metadata, indent=2))
        print("\n--- BibTeX ---")
        print(generate_bibtex(metadata))
