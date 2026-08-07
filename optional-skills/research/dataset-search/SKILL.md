---
name: dataset-search
description: Search and filter HuggingFace datasets.
version: 1.0.0
author: Kewe63
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [datasets, huggingface, machine-learning, research]
    related_skills: [drug-discovery, bioinformatics, clinical-trials, patent-research]
prerequisites:
  commands: [python3]
---

# Dataset Search Skill

Search and analyse [HuggingFace Datasets](https://huggingface.co/datasets) — free, no API key required. Filter by task category, modality, size bucket, or language; retrieve per-dataset detail (description, card data, citation, configs).

## When to Use

- Discover existing datasets for a downstream ML task before training a model
- Check whether a candidate dataset (namespaced like `keremberke/chest-xray-classification`) is real and how popular it is
- Pull a dataset's `cardData` fields (license, task categories, language) for compliance / catalogue work
- Survey popular datasets in a niche (by `likes` count) when a problem has no obvious training corpus

Prefer `web_search` for "what is the best dataset for X" recommendation framings; this skill targets the wire-data fit (filterable, machine-quotable JSON).

## Prerequisites

- Python 3.10+ (stdlib only — `urllib`, `argparse`, `json`)
- Outbound HTTPS to `huggingface.co` (port 443)
- No API key, no signup

## How to Run

Commands run via the bundled helper script. `${HERMES_SKILL_DIR}` is substituted at scan time by the skill loader; copy-paste the resolved path or run from inside the skill directory.

```bash
# Free-text search across dataset cards
python3 "${HERMES_SKILL_DIR}/scripts/dataset_search.py" search "chest xray"

# Filter by HuggingFace task category
python3 "${HERMES_SKILL_DIR}/scripts/dataset_search.py" search "text classification" \
    --task text-classification --limit 10

# Filter by modality (image / text / audio / ...)
python3 "${HERMES_SKILL_DIR}/scripts/dataset_search.py" search "image" --modality image

# Filter by size bucket — limits results to small / medium / large datasets only
python3 "${HERMES_SKILL_DIR}/scripts/dataset_search.py" search "image" \
    --modality image --size small

# Top-N datasets by likes count
python3 "${HERMES_SKILL_DIR}/scripts/dataset_search.py" popular --limit 10

# Per-dataset detail (description, card data, citation, configs)
python3 "${HERMES_SKILL_DIR}/scripts/dataset_search.py" detail \
    "keremberke/chest-xray-classification"
```

## Quick Reference

| Command | Positional | Key flags | Returns |
|---------|-----------|-----------|---------|
| `search` | `<query>` | `--task`, `--modality`, `--size`, `--lang`, `--limit` | `{query, total, results: [...]}` |
| `popular` | — | `--limit` | `{popular: true, results: [...]}` |
| `detail` | `<dataset_id>` | — | full dataset dict |

`--size` choices: `small`, `medium`, `large`. Maps to `cardData.size_categories` entries of the same name (HuggingFace standard taxonomy). Omitted = no size filtering.

`--modality` examples: `image`, `text`, `audio`, `video`, `3d`. Free-form — the script does not validate against the catalogue.

`--limit` default: `10`. The HuggingFace API caps at `100`; the script does not clamp internally — pass a value ≤ 100.

`detail` always returns first 2,000 chars of `description` and first 500 chars of `citation` to keep payloads bounded.

## Procedure

1. Start with `search "<topic>"` to surface candidate dataset IDs; `--limit` at 20 is a good initial probe.
2. Add filters one at a time (`--task`, then `--modality`, then `--size`) so each step prunes results visibly.
3. For each promising dataset, run `detail "<owner>/<dataset>"` to capture `license`, `task_categories`, `language`, and `size_categories` from `cardData`.
4. Report the dataset's `downloads` and `likes` counts alongside the description so the user can gauge traction.

## Pitfalls

- HuggingFace API rate-limits unauthenticated calls at ~10 req/min/IP; if results return empty unexpectedly, wait 60s before retrying.
- Namespaced dataset IDs (`owner/name`) **must preserve the slash** — the script passes `safe='/'` to `urllib.parse.quote` to avoid the API returning 400. Encoded variants like `%2F` are rejected.
- `--size` filtering is client-side over `cardData.size_categories` — datasets that don't publish a size bucket are excluded when a size filter is set. Drop `--size` to surface them.
- The HuggingFace API does **not** paginate; specifying `--limit 100` returns whatever fit, no follow-up. For larger sweeps, page by varying `--search` / filters.
- `cardData` is user-supplied — fields like `task_categories` and `language` can be missing or empty. Treat empty arrays as "unknown", not "no".
- `citation` may be a long BibTeX block; the script truncates to 500 chars.

## Verification

Run the bundled tests (no network required — all HTTP is mocked):

```bash
scripts/run_tests.sh tests/skills/test_dataset_search_skill.py -q
```

Spot-check a real call before quoting results to the user:

```bash
python3 "${HERMES_SKILL_DIR}/scripts/dataset_search.py" search "hypertension" --limit 3 | head -40
```

The output should be JSON with `query`, `total`, and `results[].id`, `likes`, `downloads`, `tags`, `siblings`.
