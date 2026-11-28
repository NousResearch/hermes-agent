"""Atomic, retry-safe JSON ingestion through the ledger's existing record builders."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

SOURCE_FIELDS = {'key', 'url', 'title', 'source_type', 'questions', 'summary', 'evidence',
                 'publisher', 'author', 'published_date', 'accessed_date', 'independence_group',
                 'independence_note', 'limitations', 'extraction_status'}
CLAIM_FIELDS = {'key', 'text', 'question', 'status', 'sources', 'counter_sources', 'evidence',
                'importance', 'caveat', 'reasoning'}


def _record(value: dict, allowed: set[str], required: set[str]) -> None:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f'Invalid record fields; required={sorted(required)}, allowed={sorted(allowed)}')
    if not isinstance(value['key'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', value['key']):
        raise ValueError('Record keys must be simple identifiers (letters, digits, underscore or dash)')


def _strings(values, label: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f'{label} must be an array of nonempty strings')
    return values


def _source(data: dict, record: dict) -> dict:
    from evidence_ledger import SOURCE_TYPES, make_source
    _record(record, SOURCE_FIELDS, {'key', 'url', 'title', 'source_type', 'questions', 'summary', 'evidence'})
    if record['source_type'] not in SOURCE_TYPES:
        raise ValueError('Unknown source_type')
    text_fields = SOURCE_FIELDS - {'questions', 'evidence', 'limitations'}
    if any(not isinstance(record[k], str) for k in text_fields & record.keys()):
        raise ValueError('Source text fields must be strings')
    if not record['title'].strip() or not record['summary'].strip():
        raise ValueError('Sources need a title and compact summary')
    questions = _strings(record['questions'], 'questions')
    if not questions or not isinstance(record['evidence'], list) or not record['evidence']:
        raise ValueError('Sources need questions and excerpt/locator evidence')
    if record.get('extraction_status', 'extracted') not in {'pending', 'extracted', 'failed'}:
        raise ValueError('Unknown extraction_status')
    args = argparse.Namespace(**{k: record.get(k) for k in (
        'url', 'title', 'publisher', 'author', 'published_date', 'accessed_date',
        'source_type', 'summary', 'independence_group', 'independence_note')},
        question=questions, extraction_status=record.get('extraction_status', 'extracted'),
        excerpt=[], limitation=_strings(record.get('limitations', []), 'limitations'),
        evidence=[json.dumps(item) for item in record['evidence']])
    return {**make_source(data, args), 'key': record['key']}


def _claim(data: dict, record: dict, sources: dict) -> dict:
    from evidence_ledger import CLAIM_STATUSES, make_claim
    _record(record, CLAIM_FIELDS, {'key', 'text', 'question', 'status', 'sources', 'evidence'})
    if any(not isinstance(record[k], str) for k in {'text', 'question', 'status', 'importance', 'caveat', 'reasoning'} & record.keys()):
        raise ValueError('Claim text fields must be strings')
    if not record['text'].strip() or record['status'] not in CLAIM_STATUSES:
        raise ValueError('Claims need text and a recognized status')
    if record.get('importance', 'major') not in {'major', 'minor'}:
        raise ValueError('Unknown claim importance')
    supporting = _strings(record['sources'], 'sources')
    counter = _strings(record.get('counter_sources', []), 'counter_sources')
    if set(supporting + counter) - sources.keys():
        raise ValueError('Claim references an unknown source key')
    evidence = _evidence_ids(record['evidence'], sources)
    args = argparse.Namespace(text=record['text'], question=record['question'], status=record['status'],
        importance=record.get('importance', 'major'), reasoning=record.get('reasoning'), caveat=record.get('caveat'),
        source=[str(sources[k]['id']) for k in supporting],
        counter_source=[str(sources[k]['id']) for k in counter], evidence=evidence)
    return {**make_claim(data, args), 'key': record['key']}


def _evidence_ids(values, sources: dict) -> list[str]:
    result = []
    for key in _strings(values, 'evidence'):
        source_key, separator, index = key.rpartition(':')
        if not separator or source_key not in sources or not index.isdigit() or int(index) < 1:
            raise ValueError(f'Unknown evidence key {key!r}; use source_key:1')
        items = sources[source_key].get('evidence', [])
        if int(index) > len(items):
            raise ValueError(f'Unknown evidence key {key!r}')
        result.append(items[int(index) - 1]['id'])
    return result


def _keys(records: list, prefix: str) -> dict:
    result = {}
    for item in records:
        key = item.get('key', f"{prefix}{item['id']}")
        if key in result:
            raise ValueError(f'Duplicate stored key {key!r}')
        result[key] = item
    return result


def ingest(path: Path, payload: dict) -> dict:
    from evidence_ledger import read_ledger, validate, write_ledger
    if not isinstance(payload, dict) or set(payload) != {'batch_id', 'sources', 'claims'}:
        raise ValueError('Batch needs exactly batch_id, sources and claims')
    batch_id = payload['batch_id']
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError('batch_id must be a nonempty string')
    if not all(isinstance(payload[k], list) for k in ('sources', 'claims')):
        raise ValueError('Batch sources and claims must be arrays')
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    data = read_ledger(path)
    receipt = data.get('batches', {}).get(batch_id)
    if receipt:
        if receipt['sha256'] != fingerprint:
            raise ValueError('batch_id was already used with different contents')
        return receipt
    staged = copy.deepcopy(data)
    source_keys = _keys(staged['sources'], 'S')
    claim_keys = _keys(staged['claims'], 'C')
    added_sources, added_claims = {}, {}
    for record in payload['sources']:
        source = _source(staged, record)
        if source['key'] in source_keys:
            raise ValueError('Source key already exists; reference it instead of adding it again')
        staged['sources'].append(source)
        source_keys[source['key']] = source
        added_sources[source['key']] = source['id']
    for record in payload['claims']:
        claim = _claim(staged, record, source_keys)
        if claim['key'] in claim_keys:
            raise ValueError('Claim key already exists; use a new key for a revised claim')
        staged['claims'].append(claim)
        claim_keys[claim['key']] = claim
        added_claims[claim['key']] = claim['id']
    errors, _ = validate(staged)
    if errors:
        raise ValueError('; '.join(errors))
    receipt = {'sha256': fingerprint, 'sources': added_sources, 'claims': added_claims}
    staged.setdefault('batches', {})[batch_id] = receipt
    write_ledger(path, staged)
    return receipt


def command(args) -> None:
    payload = json.loads(Path(args.payload).expanduser().read_text(encoding='utf-8'))
    print(json.dumps(ingest(Path(args.ledger).expanduser(), payload), ensure_ascii=False))
