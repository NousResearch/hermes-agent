---
name: quadtree-refs
description: Query the quadtree literature reference DB at /opt/data/references/quadtree-references.json. Search by keyword, tag, year, or author.
version: 1.0.0
---

# quadtree-refs

Gordon's personal reference collection for quadtree research.

## File
`/opt/data/references/quadtree-references.json` — array of reference objects.

## Lookup (ad-hoc)

Use `execute_code` or `terminal + python3` to query:

```python
import json
with open("/opt/data/references/quadtree-references.json") as f:
    refs = json.load(f)

# Search by tag
[q for q in refs if "balance" in q["tags"]]

# Search by keyword in description
[q for q in refs if "depth" in q["description"].lower()]

# All entries sorted by year
sorted(refs, key=lambda r: r["year"])
```

## Adding a reference

Edit the JSON file directly. Each entry needs:
```json
{
  "id": "unique-slug",
  "title": "...",
  "authors": "...",
  "year": YYYY,
  "venue": "...",
  "url": "...",
  "description": "2-3 sentence summary",
  "tags": ["tag1", "tag2"]
}
```

## Coverage gaps (not yet in DB)

- Point quadtree deletion/reinsertion complexity
- Color image / multi-band quadtrees
- Parallel/ concurrent quadtree construction
- Distributed quadtrees (e.g., for P2P or map-reduce)
- Specific metrics: traversal cost, query sensitivity to depth cap
